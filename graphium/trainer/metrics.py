"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


from typing import Union, Callable, Optional, Dict, Any, Literal, List, Tuple

import sys

import torch
from torch import Tensor
import torch.distributed as dist
import operator as op
from copy import deepcopy
from loguru import logger

from torch.nn.modules.loss import _Loss
from torchmetrics.utilities.distributed import reduce
import torchmetrics.functional.regression.mae
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics import Metric

from graphium.utils.tensor import nan_mean

# NOTE(hadim): the below is a fix to be able to import previously saved Graphium model that are incompatible
# with the current version of torchmetrics.
# In the future, we should NOT save any torchmetrics objects during serialization.
# See https://github.com/valence-discovery/graphium/issues/106
sys.modules["torchmetrics.functional.regression.mean_absolute_error"] = torchmetrics.functional.regression.mae

EPS = 1e-5


class Thresholder:
    def __init__(
        self,
        threshold: float,
        operator: Union[Literal["greater", "gt", ">", "lower", "lt", "<"], Callable] = "greater",
        th_on_preds: bool = True,
        th_on_target: bool = False,
    ):
        # Basic params
        self.threshold = threshold
        self.th_on_target = th_on_target
        self.th_on_preds = th_on_preds
        self.operator, self.op_str = self._get_operator(operator)

    def compute(self, preds: Tensor, target: Tensor):
        # Apply the threshold on the predictions
        if self.th_on_preds:
            preds = self.operator(preds, self.threshold)

        # Apply the threshold on the targets
        if self.th_on_target:
            target = self.operator(target, self.threshold)

        return preds, target

    def __call__(self, preds: Tensor, target: Tensor):
        return self.compute(preds, target)

    def __repr__(self):
        r"""
        Control how the class is printed
        """

        return f"{self.op_str}{self.threshold}"

    @staticmethod
    def _get_operator(operator):
        """Operator can either be a string, or a callable"""
        if isinstance(operator, str):
            op_name = operator.lower()
            if op_name in ["greater", "gt", ">"]:
                op_str = ">"
                operator = op.gt
            elif op_name in ["lower", "lt", "<"]:
                op_str = "<"
                operator = op.lt
            else:
                raise ValueError(f"operator `{op_name}` not supported")
        elif callable(operator):
            op_str = operator.__name__
            if op_str == "lt":
                op_str = "<"
            elif op_str == "gt":
                op_str = ">"
        elif operator is None:
            pass
        else:
            raise TypeError(f"operator must be either `str` or `callable`, provided: `{type(operator)}`")

        return operator, op_str

    def __getstate__(self):
        """Serialize the class for pickling."""
        state = {}
        state["threshold"] = self.threshold
        state["th_on_target"] = self.th_on_target
        state["th_on_preds"] = self.th_on_preds

        # Set the operator state.
        # If it's a callable, it's up to the user to ensure it unserializes well
        operator = self.operator
        if operator == op.lt:
            operator = "lower"
        elif operator == op.gt:
            operator = "greater"
        state["operator"] = operator

        return state

    def __setstate__(self, state: dict):
        state["operator"], state["op_str"] = self._get_operator(state["operator"])
        self.__dict__.update(state)
        self.__init__(**state)

    def __eq__(self, obj) -> bool:
        is_eq = [
            self.threshold == obj.threshold,
            self.th_on_target == obj.th_on_target,
            self.th_on_preds == obj.th_on_preds,
            self.operator == obj.operator,
            self.op_str == obj.op_str,
        ]
        return all(is_eq)


def _filter_nans(preds: Tensor, target: Tensor, target_nan_mask: Union[Literal[None, "none", "ignore"], int]) -> Tuple[Tensor, Tensor]:
    """Handle the NaNs according to the chosen options"""

    if target_nan_mask is None: # No NaN handling
        return preds, target
    
    if target.dtype in [torch.int, torch.int16, torch.int32, torch.int64, torch.int8]:
        target_nans = (torch.iinfo(target.dtype).min == target) | (torch.iinfo(target.dtype).max == target)
    else:
        target_nans = torch.isnan(target)
    if ~target_nans.any(): # No NaNs
        return preds, target
    elif isinstance(target_nan_mask, (int, float)): # Replace NaNs
        target = target.clone()
        target[target_nans] = target_nan_mask
    elif target_nan_mask == "ignore": # Remove NaNs
        target = target[~target_nans]
        preds = preds[~target_nans]
    else:
        raise ValueError(f"Invalid option `{target_nan_mask}`")
    return preds, target

class MetricWrapper:
    r"""
    Allows to initialize a metric from a name or Callable, and initialize the
    `Thresholder` in case the metric requires a threshold.
    """

    def __init__(
        self,
        metric: Union[str, torchmetrics.Metric, torch.nn.modules.loss._Loss],
        threshold_kwargs: Optional[Dict[str, Any]] = None,
        target_nan_mask: Union[Literal[None, "none", "ignore"], int] = None,
        multitask_handling: Literal[None, "none", "flatten", "mean-per-label"] = None,
        squeeze_targets: bool = False,
        target_to_int: bool = False,
        **kwargs,
    ):
        r"""
        Parameters
            metric:
                The metric to use. See `METRICS_DICT`

            threshold_kwargs:
                If `None`, no threshold is applied.
                Otherwise, we use the class `Thresholder` is initialized with the
                provided argument, and called before the `compute`

            target_nan_mask:

                - None: Do not change behaviour if there are NaNs

                - int, float: Value used to replace NaNs. For example, if `target_nan_mask==0`, then
                  all NaNs will be replaced by zeros

                - 'ignore': The NaN values will be removed from the tensor before computing the metrics.
                  Must be coupled with the `multitask_handling='flatten'` or `multitask_handling='mean-per-label'`.

            multitask_handling:
                - None: Do not process the tensor before passing it to the metric.
                  Cannot use the option `multitask_handling=None` when `target_nan_mask=ignore`.
                  Use either 'flatten' or 'mean-per-label'.

                - 'flatten': Flatten the tensor to produce the equivalent of a single task

                - 'mean-per-label': Loop all the labels columns, process them as a single task,
                    and average the results over each task
                  *This option might slow down the computation if there are too many labels*

            squeeze_targets:
                If true, targets will be squeezed prior to computing the metric.
                Required in classifigression task.

            target_to_int:
                If true, targets will be converted to integers prior to computing the metric.

            kwargs:
                Other arguments to call with the metric
        """

        metric_class, self.metric_name = self._get_metric_class(metric)
        self.thresholder = None
        if threshold_kwargs is not None:
            self.thresholder = Thresholder(**threshold_kwargs)

        self.target_nan_mask = self._parse_target_nan_mask(target_nan_mask)
        self.multitask_handling = self._parse_multitask_handling(multitask_handling, self.target_nan_mask)
        self.squeeze_targets = squeeze_targets
        self.target_to_int = target_to_int
        self.kwargs = kwargs

        self.metric, self.kwargs = self._initialize_metric(metric_class, self.target_nan_mask, self.multitask_handling, **self.kwargs)

    @staticmethod
    def _initialize_metric(metric, target_nan_mask, multitask_handling, **kwargs):
        r"""
        Initialize the metric with the provided kwargs
        """
    
        if not isinstance(metric, type):
            if callable(metric):
                metric = MetricToConcatenatedTorchMetrics(
                    metric_fn=metric,
                    target_nan_mask=target_nan_mask, 
                    multitask_handling=multitask_handling, 
                    **kwargs)
                return metric, kwargs
            elif all(hasattr(metric, method) for method in ["update", "compute", "reset", "to"]):
                return metric, kwargs
            else:
                raise ValueError(f"metric must be a callable, or a class with 'update', 'compute', 'reset', 'to', provided: `{type(metric)}`")
        
        metric = metric(**kwargs)
        if not all(hasattr(metric, method) for method in ["update", "compute", "reset", "to"]):
            raise ValueError(f"metric must be a callable, or a class with 'update', 'compute', 'reset', 'to', provided: `{type(metric)}`")

        return metric, kwargs


    @staticmethod
    def _parse_target_nan_mask(target_nan_mask):
        """
        Parse the `target_nan_mask` parameter
        """

        if (target_nan_mask is None) or isinstance(target_nan_mask, (int, float)):
            # None, int, float, are accepted
            pass
        elif isinstance(target_nan_mask, Tensor) and (target_nan_mask.numel() == 1):
            # Only single element tensors are accepted
            target_nan_mask = target_nan_mask.flatten()[0]
        elif isinstance(target_nan_mask, str):
            # Only a few str options are accepted
            target_nan_mask = target_nan_mask.lower()
            accepted_str = ["ignore", "none"]
            assert (
                target_nan_mask in accepted_str
            ), f"Provided {target_nan_mask} not in accepted_str={accepted_str}"

            if target_nan_mask == "none":
                target_nan_mask = None
        else:
            raise ValueError(f"Unrecognized option `target_nan_mask={target_nan_mask}`")
        return target_nan_mask

    @staticmethod
    def _parse_multitask_handling(multitask_handling, target_nan_mask):
        """
        Parse the `multitask_handling` parameter
        """

        if multitask_handling is None:
            # None is accepted
            pass
        elif isinstance(multitask_handling, str):
            # Only a few str options are accepted
            multitask_handling = multitask_handling.lower()
            accepted_str = ["flatten", "mean-per-label", "none"]
            assert (
                multitask_handling in accepted_str
            ), f"Provided {multitask_handling} not in accepted_str={accepted_str}"

            if multitask_handling == "none":
                multitask_handling = None
        else:
            raise ValueError(f"Unrecognized option `multitask_handling={multitask_handling}`")

        if (target_nan_mask == "ignore") and (multitask_handling is None):
            raise ValueError(
                "Cannot use the option `multitask_handling=None` when `target_nan_mask=ignore`. Use either 'flatten' or 'mean-per-label'"
            )

        return multitask_handling

    @staticmethod
    def _get_metric_class(metric):
        from graphium.utils.spaces import METRICS_DICT

        if isinstance(metric, str):
            metric_name = MetricWrapper._ipu_metrics_name_conversion(metric)
            metric = METRICS_DICT[metric_name]
        else:
            metric_name = None
            metric = metric
        return metric, metric_name
    
    @staticmethod
    def _ipu_metrics_name_conversion(metric, warning=True):
        r"""
        Convert the metric name from the removed ipu metrics to the regular torchmetrics metrics
        """
        metric_name = metric
        if metric_name.endswith("_ipu"): # For backward compatibility when loading models with metrics for ipu
            metric_name = metric_name[:-4]
            if metric_name == "average_precision": # A previous typo in the `spaces.py`
                metric_name = "averageprecision"
            if warning:
                logger.warning(f"Using the metric `{metric_name}` instead of `{metric}`")
        return metric_name

    def update(self, preds: Tensor, target: Tensor) -> Tensor:
        r"""
        Update the parameters of the metric, apply the thresholder if provided, and manage the NaNs.
        See `torchmetrics.Metric.update` for more details.
        """
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)

        if target.ndim == 1:
            target = target.unsqueeze(-1)

        # Threshold the prediction
        if self.thresholder is not None:
            preds, target = self.thresholder(preds, target)

        # for the classifigression task, cast predictions from
        # (batch_size, n_targets * n_brackets) to (batch_size, n_targets, n_brackets)
        # TODO: make this more flexible to the target shape in the future
        if preds.shape[1] != target.shape[1]:
            preds = preds.view(target.shape[0], target.shape[1], -1)
            classifigression = True
        else:
            classifigression = False

        if (self.multitask_handling is None):
            # In case of no multi-task handling, apply the nan filtering, then compute the metrics
            assert (
                self.target_nan_mask != "ignore"
            ), f"Cannot use the option `multitask_handling=None` when `target_nan_mask=ignore`. Use either 'flatten' or 'mean-per-label'"
            preds, target = self._filter_nans(preds, target)
            if self.squeeze_targets:
                target = target.squeeze()
            if self.target_to_int:
                target = target.to(int)
            self.metric.update(preds, target)


        elif self.multitask_handling == "flatten":
            # Flatten the tensors, apply the nan filtering, then compute the metrics
            if classifigression:
                preds = preds.view(-1, preds.shape[-1])
                target = target.flatten()
            else:
                preds, target = preds.flatten(), target.flatten()
            preds, target = self._filter_nans(preds, target)
            if self.squeeze_targets:
                target = target.squeeze()
            if self.target_to_int:
                target = target.to(int)
            self.metric.update(preds, target)

        elif isinstance(self.metric, MetricToConcatenatedTorchMetrics):
            # NaN's and multitask handling are handled by the MetricToConcatenatedTorchMetrics
            if self.squeeze_targets:
                target = target.squeeze()
            if self.target_to_int:
                target = target.to(int)
            self.metric.update(preds, target)
        
        elif self.multitask_handling == "mean-per-label":
            # Loop the columns (last dim) of the tensors, apply the nan filtering, compute the metrics per column, then average the metrics
            target_list = [target[..., ii] for ii in range(target.shape[-1])]
            if classifigression:
                preds_list = [preds[..., ii, :] for ii in range(preds.shape[1])]
            else:
                preds_list = [preds[..., ii] for ii in range(preds.shape[-1])]

            if not isinstance(self.metric, list):
                self.metric = [deepcopy(self.metric) for _ in range(len(target_list))]
            for ii in range(len(target_list)):
                try:
                    this_preds, this_target = self._filter_nans(preds_list[ii], target_list[ii])
                    if self.squeeze_targets:
                        this_target = this_target.squeeze()
                    if self.target_to_int:
                        this_target = this_target.to(int)
                    self.metric[ii].update(this_preds, this_target)
                except:
                    pass
        else:
            # Wrong option
            raise ValueError(f"Invalid option `self.multitask_handling={self.multitask_handling}`")

    def compute(self) -> Tensor:
        r"""
        Compute the metric with the method `self.compute`
        """
        if isinstance(self.metric, list):
            metrics = [metric.compute() for metric in self.metric]
            return nan_mean(torch.stack(metrics))

        return self.metric.compute()

    def update_compute(self, preds: Tensor, target: Tensor) -> Tensor:
        r"""
        Update the parameters of the metric, apply the thresholder if provided, and manage the NaNs.
        Then compute the metric with the method `self.compute`
        """

        self.update(preds, target)
        return self.compute()

    def reset(self) -> None:
        r"""
        Reset the metric with the method `self.metric.reset`
        """
        if isinstance(self.metric, list):
            for metric in self.metric:
                metric.reset()
        else:
            self.metric.reset()

    def to(self, device: Union[str, torch.device]) -> None:
        r"""
        Move the metric to the device with the method `self.metric.to`
        """
        if isinstance(self.metric, list):
            for metric in self.metric:
                metric.to(device)
        else:
            self.metric.to(device)

    @property
    def device(self) -> torch.device:
        r"""
        Return the device of the metric with the method `self.metric.device` or `self.metric[0].device`
        """
        if isinstance(self.metric, list):
            return self.metric[0].device
        return self.metric.device


    def _filter_nans(self, preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """Handle the NaNs according to the chosen options"""

        return _filter_nans(preds, target, self.target_nan_mask)

    def __call__(self, preds: Tensor, target: Tensor) -> Tensor:
        r"""
        Compute the metric with the method `self.compute`
        """
        return self.update_compute(preds, target)

    def __repr__(self):
        r"""
        Control how the class is printed
        """
        full_str = f"{self.metric.__repr__()}"
        if self.thresholder is not None:
            full_str += f"({self.thresholder})"

        return full_str

    def __eq__(self, obj) -> bool:
        is_eq = [
            self.metric == obj.metric,
            self.metric_name == obj.metric_name,
            self.thresholder == obj.thresholder,
            self.target_nan_mask == obj.target_nan_mask,
            self.multitask_handling == obj.multitask_handling,
            self.squeeze_targets == obj.squeeze_targets,
            self.target_to_int == obj.target_to_int,
            self.kwargs == obj.kwargs,
        ]
        return all(is_eq)

    def __getstate__(self):
        """Serialize the class for pickling."""
        state = {}
        if self.metric_name is None:
            state["metric"] = self.metric
        else:
            state["metric"] = self.metric_name
        state["target_nan_mask"] = self.target_nan_mask
        state["multitask_handling"] = self.multitask_handling
        state["squeeze_targets"] = self.squeeze_targets
        state["target_to_int"] = self.target_to_int
        state["kwargs"] = self.kwargs
        state["threshold_kwargs"] = None
        if self.thresholder is not None:
            state["threshold_kwargs"] = self.thresholder.__getstate__()
        return state

    def __setstate__(self, state: dict):
        """Reload the class from pickling."""
        state["metric"], state["metric_name"] = self._get_metric_class(state["metric"])
        thresholder = state.pop("threshold_kwargs", None)
        if thresholder is not None:
            thresholder = Thresholder(**thresholder)
        state["thresholder"] = thresholder
        state["metric"], state["at_compute_kwargs"] = self._initialize_metric(state["metric"], state["target_nan_mask"], state["multitask_handling"], **state["kwargs"])

        self.__dict__.update(state)

class LossWrapper():
    r"""
    A simple wrapper to convert any metric or loss to an equivalent of `torchmetrics.Metric`
    by adding the `update`, `compute`, and `reset` methods to make it compatible with `MetricWrapper`.
    However, it is simply limited to computing the average of the metric over all the updates.
    """

    def __init__(self, loss):
        self.loss = loss
        self.scores: List[Tensor] = []

    def update(self, preds: Tensor, target: Tensor):
        self.scores.append(self.loss(preds, target))

    def compute(self):
        if len(self.scores) == 0:
            raise ValueError("No scores to compute")
        elif len(self.scores) == 1:
            return self.scores[0]
        return nan_mean(torch.stack(self.scores))
    
    def to(self, device: Union[str, torch.device]):
        for ii in range(len(self.scores)):
            self.scores[ii] = self.scores[ii].to(device)

    @property
    def device(self) -> torch.device:
        self.loss.device

    def reset(self):
        self.scores = []


class MetricToMeanTorchMetrics(Metric):
    r"""
    A simple wrapper to convert any metric or loss to an equivalent of `torchmetrics.Metric`
    by adding the `update`, `compute`, and `reset` methods to make it compatible with `MetricWrapper`.

    However, it is limited in functionality. At each `.update()`, it computes the metric and stores in a list.
    Then at `.compute()` it returns the average of the computed metric, while ignoring NaNs.
    """
    scores: List[Tensor] = []

    def __init__(self, metric_fn):
        super().__init__(dist_sync_on_step=False)
        self.metric_fn = metric_fn
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor):
        self.scores.append(self.metric_fn(preds.detach(), target))

    def compute(self):
        if len(self.scores) == 0:
            raise ValueError("No scores to compute")
        elif len(self.scores) == 1:
            return self.scores[0]
        return nan_mean(torch.stack(self.scores))


class MetricToConcatenatedTorchMetrics(Metric):

    preds: List[Tensor] # Always on CPU
    target: List[Tensor] # Always on CPU

    def __init__(self, 
                    metric_fn: Callable,
                    target_nan_mask: Union[Literal[None, "none", "ignore"], int] = None,
                    multitask_handling: Literal[None, "none", "flatten", "mean-per-label"] = None,
                    **kwargs,
                 ):
        r"""
            A wrapper around the `torchmetrics.Metric` to handle the saving and syncing of `preds` and `target` tensors,
            and moving them to the CPU.
            This is useful for certain metrics that require to save all preds and targets, such as auroc and average_precision.
            Otherwise, if using `MetricWrapper` with the option `mean-per-label`, the `preds` and `target` would be
            duplicated for each label, causing major memory spikes. 
            On top of that, all preds and targets would be on the GPU, which would cause the memory to increase at every step, 
            and potentially lead to out-of-memory before the end of the epoch.

            Parameters
            ----------

            metric_fn:
                The metric function to use. This function should take `preds` and `target` as input, and return a scalar value.

            target_nan_mask:
                - None: Do not change behaviour if there are NaNs

                - int, float: Value used to replace NaNs. For example, if `target_nan_mask==0`, then
                  all NaNs will be replaced by zeros

                - 'ignore': The NaN values will be removed from the tensor before computing the metrics.
                  Must be coupled with the `multitask_handling='flatten'` or `multitask_handling='mean-per-label'`.

            multitask_handling:
                - None: Do not process the tensor before passing it to the metric.
                  Cannot use the option `multitask_handling=None` when `target_nan_mask=ignore`.
                  Use either 'flatten' or 'mean-per-label'.

                - 'flatten': Flatten the tensor to produce the equivalent of a single task

                - 'mean-per-label': Loop all the labels columns, process them as a single task,
                    and average the results over each task
                  *This option might slow down the computation if there are too many labels*

        """
                 
        super().__init__(compute_on_cpu=True, dist_sync_on_step=False, sync_on_compute=False)
        self.metric_fn = metric_fn
        self.target_nan_mask = target_nan_mask
        self.multitask_handling = multitask_handling
        self.kwargs = kwargs
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self._to_device_warned: bool = False
        super().to("cpu")

    def update(self, preds: Tensor, target: Tensor):

        # If distributed, gather the preds and target tensors
        if self.dist_sync_fn is not None:
            preds_list = self.dist_sync_fn(preds, self.process_group)
            target_list = self.dist_sync_fn(target, self.process_group)
            preds = dim_zero_cat(preds_list)
            target = dim_zero_cat(target_list)

        # Move the tensors to the CPU after gathering them
        self.preds.append(preds.detach().cpu())
        self.target.append(target.cpu())

    def compute(self):
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        if (self.multitask_handling is None) or (self.multitask_handling in ["none", "flatten"]):
            preds, target = _filter_nans(preds, target, self.target_nan_mask)
            value = self.metric_fn(preds, target,  **self.kwargs)

        elif self.multitask_handling == "mean-per-label":
            value = []
            # Loop the columns (last dim) of the tensors, apply the nan filtering, compute the metrics per column, then average the metrics
            target_list = [target[..., ii] for ii in range(target.shape[-1])]
            preds_list = [preds[..., ii] for ii in range(preds.shape[-1])]
            for ii in range(len(target_list)):
                try:
                    this_preds, this_target = _filter_nans(preds_list[ii], target_list[ii], self.target_nan_mask)
                    value.append(self.metric_fn(this_preds, this_target, **self.kwargs))
                except:
                    pass
            value = nan_mean(torch.stack(value))
        else:
            # Wrong option
            raise ValueError(f"Invalid option `self.multitask_handling={self.multitask_handling}`")
        return value
    
    def to(self, device: Union[str, torch.device]):
        """
        Disables the moving of the metric to another device. Stays on CPU to avoid overflow.
        """
        device = torch.device(device)
        if device == torch.device("cpu"):
            return
        if not self._to_device_warned:
            self._to_device_warned = True
            logger.warning(f"{self.get_obj_name(self)}({self.get_obj_name(self.metric_fn)}) stays on `{self.device}`, won't move to `{device}`")
        
    @staticmethod
    def get_obj_name(obj):
        """
        Returns the name of a function, class, or instance of a class.
        
        Parameters:
        - obj: The object to get the name of.
        
        Returns:
        - The name of the object as a string.
        """
        # If the object is a class or function, return its __name__
        if hasattr(obj, '__name__'):
            return obj.__name__
        # If the object is an instance of a class, return its class's __name__
        elif hasattr(obj, '__class__'):
            return obj.__class__.__name__
        else:
            return str(obj)  # Fallback to converting the object to string

