from typing import Union, Callable, Optional, Dict, Any

import sys

import torch
from torch import Tensor
import operator as op

from torchmetrics.utilities.distributed import reduce
import torchmetrics.functional.regression.mae

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
        operator: Union[str, Callable] = "greater",
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
            if op_name in ["greater", "gt"]:
                op_str = ">"
                operator = op.gt
            elif op_name in ["lower", "lt"]:
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


class MetricWrapper:
    r"""
    Allows to initialize a metric from a name or Callable, and initialize the
    `Thresholder` in case the metric requires a threshold.
    """

    def __init__(
        self,
        metric: Union[str, Callable],
        threshold_kwargs: Optional[Dict[str, Any]] = None,
        target_nan_mask: Optional[Union[str, int]] = None,
        multitask_handling: Optional[str] = None,
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

        self.metric, self.metric_name = self._get_metric(metric)
        self.thresholder = None
        if threshold_kwargs is not None:
            self.thresholder = Thresholder(**threshold_kwargs)

        self.target_nan_mask = self._parse_target_nan_mask(target_nan_mask)
        self.multitask_handling = self._parse_multitask_handling(multitask_handling, self.target_nan_mask)
        self.squeeze_targets = squeeze_targets
        self.target_to_int = target_to_int
        self.kwargs = kwargs

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
    def _get_metric(metric):
        from graphium.utils.spaces import METRICS_DICT

        if isinstance(metric, str):
            metric_name = metric
            metric = METRICS_DICT[metric]
        else:
            metric_name = None
            metric = metric
        return metric, metric_name

    def compute(self, preds: Tensor, target: Tensor) -> Tensor:
        r"""
        Compute the metric, apply the thresholder if provided, and manage the NaNs
        """
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)

        if target.ndim == 1:
            target = target.unsqueeze(-1)

        # Threshold the prediction
        if self.thresholder is not None:
            preds, target = self.thresholder(preds, target)

        target_nans = torch.isnan(target)

        # for the classifigression task, cast predictions from
        # (batch_size, n_targets * n_brackets) to (batch_size, n_targets, n_brackets)
        # TODO: make this more flexible to the target shape in the future
        if preds.shape[1] != target.shape[1]:
            preds = preds.view(target.shape[0], target.shape[1], -1)
            classifigression = True
        else:
            classifigression = False

        if self.multitask_handling is None:
            # In case of no multi-task handling, apply the nan filtering, then compute the metrics
            assert (
                self.target_nan_mask != "ignore"
            ), f"Cannot use the option `multitask_handling=None` when `target_nan_mask=ignore`. Use either 'flatten' or 'mean-per-label'"
            preds, target = self._filter_nans(preds, target)
            if self.squeeze_targets:
                target = target.squeeze()
            if self.target_to_int:
                target = target.to(int)
            metric_val = self.metric(preds, target, **self.kwargs)
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
            metric_val = self.metric(preds, target, **self.kwargs)
        elif self.multitask_handling == "mean-per-label":
            # Loop the columns (last dim) of the tensors, apply the nan filtering, compute the metrics per column, then average the metrics
            target_list = [target[..., ii][~target_nans[..., ii]] for ii in range(target.shape[-1])]
            # TODO: make this more flexible to the target shape in the future
            if classifigression:
                preds_list = [preds[..., i, :][~target_nans[..., i]] for i in range(preds.shape[1])]
            else:
                preds_list = [preds[..., ii][~target_nans[..., ii]] for ii in range(preds.shape[-1])]
            metric_val = []
            for ii in range(len(target_list)):
                try:
                    this_preds, this_target = self._filter_nans(preds_list[ii], target_list[ii])
                    if self.squeeze_targets:
                        this_target = this_target.squeeze()
                    if self.target_to_int:
                        this_target = this_target.to(int)
                    metric_val.append(self.metric(this_preds, this_target, **self.kwargs))
                except:
                    pass
            # Average the metric
            metric_val = nan_mean(torch.stack(metric_val))
        else:
            # Wrong option
            raise ValueError(f"Invalid option `self.multitask_handling={self.multitask_handling}`")

        return metric_val

    def _filter_nans(self, preds: Tensor, target: Tensor):
        """Handle the NaNs according to the chosen options"""
        target_nans = torch.isnan(target)

        if self.target_nan_mask is None:
            pass
        elif isinstance(self.target_nan_mask, (int, float)):
            target = target.clone()
            target[torch.isnan(target)] = self.target_nan_mask
        elif self.target_nan_mask == "ignore":
            target = target[~target_nans]
            preds = preds[~target_nans]
        else:
            raise ValueError(f"Invalid option `{self.target_nan_mask}`")
        return preds, target

    def __call__(self, preds: Tensor, target: Tensor) -> Tensor:
        r"""
        Compute the metric with the method `self.compute`
        """
        return self.compute(preds, target)

    def __repr__(self):
        r"""
        Control how the class is printed
        """
        full_str = f"{self.metric.__name__}"
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
        state["metric"], state["metric_name"] = self._get_metric(state["metric"])
        thresholder = state.pop("threshold_kwargs", None)
        if thresholder is not None:
            thresholder = Thresholder(**thresholder)
        state["thresholder"] = thresholder

        self.__dict__.update(state)
