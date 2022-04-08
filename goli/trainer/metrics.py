from typing import Union, Callable, Optional, Dict, Any
import torch
import operator as op
from torchmetrics.utilities import reduce

from goli.utils.tensor import nan_mean

EPS = 1e-5


class Thresholder:
    def __init__(
        self,
        threshold: float,
        operator: str = "greater",
        th_on_preds: bool = True,
        th_on_target: bool = False,
        target_to_int: bool = False,
    ):

        # Basic params
        self.threshold = threshold
        self.th_on_target = th_on_target
        self.th_on_preds = th_on_preds
        self.target_to_int = target_to_int

        # Operator can either be a string, or a callable
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
        elif operator is None:
            pass
        else:
            raise TypeError(f"operator must be either `str` or `callable`, provided: `{type(operator)}`")

        self.operator = operator
        self.op_str = op_str

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
        from goli.utils.spaces import METRICS_DICT

        self.metric = METRICS_DICT[metric] if isinstance(metric, str) else metric

        self.thresholder = None
        if threshold_kwargs is not None:
            self.thresholder = Thresholder(**threshold_kwargs)

        self.target_nan_mask = target_nan_mask

        self.kwargs = kwargs

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
