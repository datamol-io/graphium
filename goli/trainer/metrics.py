from os import stat
from typing import Union, Callable, Optional, Dict, Any

import sys

import torch
import operator as op

from torchmetrics.utilities.distributed import reduce
import torchmetrics.functional.regression.mae

from goli.utils.tensor import nan_mean

# NOTE(hadim): the below is a fix to be able to import previously saved Goli model that are incompatible
# with the current version of torchmetrics.
# In the future, we should NOT save any torchmetrics objects during serialization.
# See https://github.com/valence-discovery/goli/issues/106
sys.modules["torchmetrics.functional.regression.mean_absolute_error"] = torchmetrics.functional.regression.mae

EPS = 1e-5


class Thresholder:
    def __init__(
        self,
        threshold: float,
        operator: Union[str, Callable] = "greater",
        th_on_preds: bool = True,
        th_on_target: bool = False,
        target_to_int: bool = False,
    ):

        # Basic params
        self.threshold = threshold
        self.th_on_target = th_on_target
        self.th_on_preds = th_on_preds
        self.target_to_int = target_to_int
        self.operator, self.op_str = self._get_operator(operator)

    def compute(self, preds: torch.Tensor, target: torch.Tensor):
        # Apply the threshold on the predictions
        if self.th_on_preds:
            preds = self.operator(preds, self.threshold)

        # Apply the threshold on the targets
        if self.th_on_target:
            target = self.operator(target, self.threshold)

        if self.target_to_int:
            target = target.to(int)

        return preds, target

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
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
        state["target_to_int"] = self.target_to_int

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
            self.target_to_int == obj.target_to_int,
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

                - 'ignore-flatten': The Tensor will be reduced to a vector without the NaN values.

                - 'ignore-mean-label': NaNs will be ignored when computing the loss. Note that each column
                  has a different number of NaNs, so the metric will be computed separately
                  on each column, and the metric result will be averaged over all columns.
                  *This option might slowdown the computation if there are too many labels*

            kwargs:
                Other arguments to call with the metric
        """

        self.metric, self.metric_name = self._get_metric(metric)
        self.thresholder = None
        if threshold_kwargs is not None:
            self.thresholder = Thresholder(**threshold_kwargs)
        self.target_nan_mask = target_nan_mask
        self.kwargs = kwargs

    @staticmethod
    def _get_metric(metric):
        from goli.utils.spaces import METRICS_DICT
        if isinstance(metric, str):
            metric_name = metric
            metric = METRICS_DICT[metric]
        else:
            metric_name = None
            metric = metric
        return metric, metric_name

    def compute(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the metric, apply the thresholder if provided, and manage the NaNs
        """

        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)

        if target.ndim == 1:
            target = target.unsqueeze(-1)

        target_nans = torch.isnan(target)

        # Threshold the prediction
        if self.thresholder is not None:
            preds, target = self.thresholder(preds, target)

        # Manage the NaNs
        if self.target_nan_mask is None:
            pass
        elif isinstance(self.target_nan_mask, (int, float)):
            target = target.clone()
            target[torch.isnan(target)] = self.target_nan_mask
        elif self.target_nan_mask == "ignore-flatten":
            target = target[~target_nans]
            preds = preds[~target_nans]
        elif self.target_nan_mask == "ignore-mean-label":
            target_list = [target[..., ii][~target_nans[..., ii]] for ii in range(target.shape[-1])]
            preds_list = [preds[..., ii][~target_nans[..., ii]] for ii in range(preds.shape[-1])]
            target = target_list
            preds = preds_list
        else:
            raise ValueError(f"Invalid option `{self.target_nan_mask}`")

        if self.target_nan_mask == "ignore-mean-label":

            # Compute the metric for each column, and output nan if there's an error on a given column
            metric_val = []
            for ii in range(len(target)):
                try:
                    metric_val.append(self.metric(preds[ii], target[ii], **self.kwargs))
                except:
                    pass

            # Average the metric
            metric_val = nan_mean(torch.stack(metric_val))

        else:
            metric_val = self.metric(preds, target, **self.kwargs)
        return metric_val

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
            self.kwargs == obj.kwargs,
        ]
        return is_eq

    def __getstate__(self):
        """Serialize the class for pickling."""
        state = {}
        if self.metric_name is None:
            state["metric"] = self.metric
        else:
            state["metric"] = self.metric_name
        state["target_nan_mask"] = self.target_nan_mask
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
