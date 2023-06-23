from typing import Optional, Tuple, Sequence, Literal

import torch
from torch import BoolTensor, IntTensor, Tensor
from torchmetrics.functional import auroc, average_precision, pearson_corrcoef, r2_score
from torchmetrics.utilities.checks import _input_squeeze
from torchmetrics.functional.classification.accuracy import (
    _mode,
    _check_subset_validity,
    _accuracy_compute,
    _accuracy_update,
)
from torchmetrics.functional.classification.precision_recall import _precision_compute, _recall_compute
from torchmetrics.functional.classification.f_beta import _fbeta_compute
from torchmetrics.functional import mean_squared_error, mean_absolute_error
from torchmetrics.utilities.checks import _input_squeeze
from torchmetrics.utilities.enums import AverageMethod

from graphium.utils.tensor import nan_mean
from graphium.ipu.ipu_utils import import_poptorch


def auroc_ipu(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
    max_fpr: Optional[float] = None,
    sample_weights: Optional[Sequence] = None,
):
    """
    A modified version of the `torchmetrics.functional.auroc` that can ignore NaNs
    by giving them the same value for both `preds` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    target = target.clone()
    preds = preds.clone()

    # Replace the nan-targets in the preds/target tensors by 0
    nan_targets = target.isnan()
    preds[nan_targets] = 0.0
    target[nan_targets] = 0.0

    # Get the original weight matrix. If None, set all weights = 1
    if sample_weights is None:
        sample_weights = torch.ones(target.shape[0], dtype=preds.dtype, device=preds.device)
    sample_weights[nan_targets] = 0.0

    # Compute the loss, and rescale by the number of nan elements
    score = auroc(
        preds=preds,
        target=target.to(int),
        num_classes=num_classes,
        task=task,
        pos_label=pos_label,
        average=average,
        max_fpr=max_fpr,
        sample_weights=sample_weights,
    )

    return score


def average_precision_ipu(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
    ignore_index: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
    sample_weights: Optional[Sequence] = None,
):
    """
    A modified version of the `torchmetrics.functional.average_precision` that can ignore NaNs
    by giving them the same value for both `preds` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    target = target.clone()
    preds = preds.clone()

    # Replace the nan-targets in the preds/target tensors by 0
    # Average precision is not sensitive to true negatives
    nan_targets = target.isnan()
    preds[nan_targets] = 0.0
    target[nan_targets] = 0.0

    # No need to use sample weights (which is no longer supported in torchmetrics >=0.10)
    # # Get the original weight matrix. If None, set all weights = 1
    # if sample_weights is None:
    #     sample_weights = torch.ones(target.shape[0], dtype=preds.dtype, device=preds.device)
    # sample_weights[nan_targets] = 0.0

    # Compute the loss, and rescale by the number of nan elements
    score = average_precision(
        preds=preds,
        target=target,
        num_classes=num_classes,
        task=task,
        ignore_index=ignore_index,
        pos_label=pos_label,
        average=average,
        # sample_weights=sample_weights,
    )

    return score


def precision_ipu(
    preds: Tensor,
    target: Tensor,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
):
    """
    A modified version of the `torchmetrics.functional.precision` that can ignore NaNs
    by giving them the same value for both `preds` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    (tp, fp, tn, fn), mode = get_confusion_matrix(
        preds=preds,
        target=target,
        average=average,
        mdmc_average=mdmc_average,
        threshold=threshold,
        top_k=top_k,
        subset_accuracy=False,
        num_classes=num_classes,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )

    return _precision_compute(tp, fp, fn, average, mdmc_average)


def recall_ipu(
    preds: Tensor,
    target: Tensor,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
):
    """
    A modified version of the `torchmetrics.functional.recall` that can ignore NaNs
    by giving them the same value for both `preds` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    (tp, fp, tn, fn), mode = get_confusion_matrix(
        preds=preds,
        target=target,
        average=average,
        mdmc_average=mdmc_average,
        threshold=threshold,
        top_k=top_k,
        num_classes=num_classes,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )

    return _recall_compute(tp, fp, fn, average, mdmc_average)


def accuracy_ipu(
    preds: Tensor,
    target: Tensor,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = "global",
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    subset_accuracy: bool = False,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    """
    A modified version of the `torchmetrics.functional.accuracy` that can ignore NaNs
    by giving them the same value for both `preds` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth labels
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the ``preds`` or ``target``,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`pages/classification:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.
        subset_accuracy:
            Whether to compute subset accuracy for multi-label and multi-dimensional
            multi-class inputs (has no effect for other input types).

            - For multi-label inputs, if the parameter is set to ``True``, then all labels for
              each sample must be correctly predicted for the sample to count as correct. If it
              is set to ``False``, then all labels are counted separately - this is equivalent to
              flattening inputs beforehand (i.e. ``preds = preds.flatten()`` and same for ``target``).

            - For multi-dimensional multi-class inputs, if the parameter is set to ``True``, then all
              sub-sample (on the extra axis) must be correct for the sample to be counted as correct.
              If it is set to ``False``, then all sub-samples are counter separately - this is equivalent,
              in the case of label predictions, to flattening the inputs beforehand (i.e.
              ``preds = preds.flatten()`` and same for ``target``). Note that the ``top_k`` parameter
              still applies in both cases, if set.

    Raises:
        ValueError:
            If ``top_k`` parameter is set for ``multi-label`` inputs.
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"``, ``None``.
        ValueError:
            If ``mdmc_average`` is not one of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``average`` is set but ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set
            and ``ignore_index`` is not in the range ``[0, num_classes)``.
        ValueError:
            If ``top_k`` is not an ``integer`` larger than ``0``.
    """

    (tp, fp, tn, fn), mode = get_confusion_matrix(
        preds=preds,
        target=target,
        average=average,
        mdmc_average=mdmc_average,
        threshold=threshold,
        top_k=top_k,
        subset_accuracy=subset_accuracy,
        num_classes=num_classes,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )

    return _accuracy_compute(tp, fp, tn, fn, average, mdmc_average, mode)


def get_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = "global",
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    subset_accuracy: bool = False,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[Tuple[Tensor], Tensor]:
    """
    Calculates the confusion matrix according to the specified average method.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth labels
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the ``preds`` or ``target``,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`pages/classification:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
    """
    allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    if average in ["macro", "weighted", "none", None] and (not num_classes or num_classes < 1):
        raise ValueError(f"When you set `average` as {average}, you have to provide the number of classes.")

    allowed_mdmc_average = [None, "samplewise", "global"]
    if mdmc_average not in allowed_mdmc_average:
        raise ValueError(f"The `mdmc_average` has to be one of {allowed_mdmc_average}, got {mdmc_average}.")

    if num_classes and ignore_index is not None and (not ignore_index < num_classes or num_classes == 1):
        raise ValueError(
            f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes"
        )

    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        raise ValueError(f"The `top_k` should be an integer larger than 0, got {top_k}")

    #### ADDED ####
    # Put all the NaNs as the 0-class
    nans = torch.isnan(target)
    target[nans] = 0
    preds[nans] = 0
    if (preds.ndim > 1) and (preds.shape[1] > 1):
        preds[nans, 0] = 1
    target = target.to(int)
    #### END ADDED ####

    preds, target = _input_squeeze(preds, target)
    mode = _mode(preds, target, threshold, top_k, num_classes, multiclass, ignore_index)
    reduce = "macro" if average in ["weighted", "none", None] else average

    if subset_accuracy and _check_subset_validity(mode):
        # correct, total = _subset_accuracy_update(preds, target, threshold, top_k, ignore_index)
        # return _subset_accuracy_compute(correct, total)
        raise NotImplementedError("subset_accuracy not implemented")
    tp, fp, tn, fn = _accuracy_update(
        preds, target, reduce, mdmc_average, threshold, num_classes, top_k, multiclass, ignore_index, mode
    )

    #### ADDED ####
    num_nans = nans.sum(0)
    if tp.numel() > 1:
        tp[0] = tp[0] - num_nans
        tn[1:] = tn[1:] - num_nans
    else:
        tn = tn - num_nans
        if (preds.ndim > 1) and (preds.shape[1] > 1):
            tp = tp - num_nans
    #### END ADDED ####

    return (tp, fp, tn, fn), mode


class NaNTensor(Tensor):
    """
    Class to create and manage a NaN tensor along it's properties

    The goal of the class is to override the regular tensor such that the basic
    operations (sum, mean, max, etc) ignore the NaNs in the input.
    It also supports NaNs in integer tensors (as the lowest integer possible).
    """

    @property
    def get_nans(self) -> BoolTensor:
        """
        Gets the boolean Tensor containing the location of NaNs.
        In the case of an integer tensor, this returns where the tensor is equal to its minimal value
        In the case of a boolean tensor, this returns a Tensor filled with `False`
        """
        if self.is_floating_point():
            return self.isnan()
        elif self.is_signed():
            return self == torch.iinfo(self.dtype).min
        else:
            return torch.zeros(self.shape, device=self.device, dtype=bool)

    def sum(self, *args, **kwargs) -> Tensor:
        """
        Overloads the traditional sum to ignore the NaNs
        """
        tensor = self.to(float)
        tensor[self.get_nans] = float("nan")
        if self.is_floating_point():
            dtype = self.dtype
        else:
            dtype = torch.int64
        return tensor.nansum(*args, **kwargs).to(dtype)

    def mean(self, *args, **kwargs) -> Tensor:
        """
        Overloads the traditional mean to ignore the NaNs
        """
        tensor = self.to(float)
        tensor[self.get_nans] = float("nan")
        return nan_mean(tensor, *args, **kwargs).to(self.dtype)

    def numel(self) -> int:
        """
        Returns the number of non-NaN elements.
        """
        return super(NaNTensor, ~self.get_nans).sum()

    def min(self, *args, **kwargs) -> Tensor:
        """
        Returns the min vale of a tensor whitout NaNs
        """
        tensor = self
        tensor = tensor[~self.get_nans]
        return super(NaNTensor, tensor).min(*args, **kwargs)

    def max(self, *args, **kwargs) -> Tensor:
        """
        Returns the max vale of a tensor whitout NaNs
        """
        tensor = self
        tensor = tensor[~self.get_nans]
        return super(NaNTensor, tensor).max(*args, **kwargs)

    def argsort(self, dim=-1, descending=False) -> IntTensor:
        """
        Return the indices that sort the tensor, while putting all the NaNs to the end of the sorting.
        """
        tensor = self
        if descending:
            tensor[tensor.get_nans] = float("-inf")
        else:
            tensor[tensor.get_nans] = float("inf")
        return super(NaNTensor, tensor).argsort(dim=dim, descending=descending)

    def size(self, dim) -> Tensor:
        """
        Instead of returning the size, return the number of non-NaN elements in
        a specific dimension. Useful for the `r2_score` metric.
        """
        return (~self.get_nans).sum(dim=dim)

    def __lt__(self, other) -> Tensor:
        """
        Stupid fix that allows the code to work with `r2_score`,
        since it requires the size to be > 2. But since `self.size` now returns
        a Tensor instead of a value, we check that all elements are > 2.
        """
        if (not isinstance(other, Tensor)) and (other == 2):
            return super().__lt__(other).all()
        else:
            return super().__lt__(other)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This __torch_function__ implementation wraps subclasses such that
        methods called on subclasses return a subclass instance instead of
        a ``torch.Tensor`` instance.

        One corollary to this is that you need coverage for torch.Tensor
        methods if implementing __torch_function__ for subclasses.

        Affects the call torch.sum() as to behave the same way as NaNTensor.sum()

        We recommend always calling ``super().__torch_function__`` as the base
        case when doing the above.

        While not mandatory, we recommend making `__torch_function__` a classmethod.
        """
        if func.__name__ == "sum":
            kwargs = {} if kwargs is None else kwargs
            return args[0].sum(*args[1:], **kwargs)
        else:
            return super().__torch_function__(func, types, args=args, kwargs=kwargs)


def pearson_ipu(preds, target):
    """Computes pearson correlation coefficient.

    Handles NaNs in the target without reshaping tensors in order to work on IPU.

    Args:
        preds: estimated scores
        target: ground truth scores
    """
    preds = NaNTensor(preds)
    target = NaNTensor(target)
    preds[target.get_nans] = float("nan")
    pearson = pearson_corrcoef(preds, target.to(preds.dtype))
    return Tensor(pearson)


def spearman_ipu(preds, target):
    """Computes spearman rank correlation coefficient.

    Handles NaNs in the target without reshaping tensors in order to work on IPU.

    Args:
        preds: estimated scores
        target: ground truth scores
    """
    nans = target.isnan()
    dtype = preds.dtype
    preds[nans] = float("inf")
    target[nans] = float("inf")
    preds_sort = _rank_data(preds).to(dtype=dtype)
    target_sort = _rank_data(target).to(dtype=dtype)
    target_sort[nans] = float("nan")
    spearman = pearson_ipu(preds_sort, target_sort)
    return Tensor(spearman)


def _rank_data(data: Tensor) -> Tensor:
    """Calculate the rank for each element of a tensor.

    The rank refers to the indices of an element in the corresponding sorted tensor (starting from 1).
    Duplicates of the same value will be assigned the mean of their rank.

    Adopted from `Rank of element tensor`_
    """
    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx] = torch.arange(1, n + 1, dtype=data.dtype, device=data.device)

    # TODO: Repeats not yet supported
    # repeats = _find_repeats(data)
    # for r in repeats:
    #     condition = data == r
    #     rank[condition] = rank[condition].mean()
    return rank


def r2_score_ipu(preds, target, *args, **kwargs) -> Tensor:
    """
    Computes r2 score also known as `R2 Score_Coefficient Determination`_:

    .. math:: R^2 = 1 - \frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}=\sum_i (y_i - f(x_i))^2` is the sum of residual squares, and
    :math:`SS_{tot}=\sum_i (y_i - \bar{y})^2` is total sum of squares. Can also calculate
    adjusted r2 score given by

    .. math:: R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}

    where the parameter :math:`k` (the number of independent regressors) should
    be provided as the ``adjusted`` argument.
    Handles NaNs without reshaping tensors in order to work on IPU.

    Args:
        preds: estimated labels
        target: ground truth labels
        adjusted: number of independent regressors for calculating adjusted r2 score.
        multioutput: Defines aggregation in the case of multiple output scores. Can be one of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances
    """
    preds = NaNTensor(preds)
    target = NaNTensor(target)
    preds[target.get_nans] = float("nan")
    score = r2_score(preds, target, *args, **kwargs)
    return Tensor(score)


def fbeta_score_ipu(
    preds: Tensor,
    target: Tensor,
    beta: float = 1.0,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
):
    """
    A modified version of the `torchmetrics.functional.classification.f_beta._fbeta_compute`
    that can ignore NaNs by giving them the same value for both `preds` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth labels
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the ``preds`` or ``target``,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`pages/classification:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.
        subset_accuracy:
            Whether to compute subset accuracy for multi-label and multi-dimensional
            multi-class inputs (has no effect for other input types).

            - For multi-label inputs, if the parameter is set to ``True``, then all labels for
              each sample must be correctly predicted for the sample to count as correct. If it
              is set to ``False``, then all labels are counted separately - this is equivalent to
              flattening inputs beforehand (i.e. ``preds = preds.flatten()`` and same for ``target``).

            - For multi-dimensional multi-class inputs, if the parameter is set to ``True``, then all
              sub-sample (on the extra axis) must be correct for the sample to be counted as correct.
              If it is set to ``False``, then all sub-samples are counter separately - this is equivalent,
              in the case of label predictions, to flattening the inputs beforehand (i.e.
              ``preds = preds.flatten()`` and same for ``target``). Note that the ``top_k`` parameter
              still applies in both cases, if set.

    Raises:
        ValueError:
            If ``top_k`` parameter is set for ``multi-label`` inputs.
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"``, ``None``.
        ValueError:
            If ``mdmc_average`` is not one of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``average`` is set but ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set
            and ``ignore_index`` is not in the range ``[0, num_classes)``.
        ValueError:
            If ``top_k`` is not an ``integer`` larger than ``0``.
    """

    (tp, fp, tn, fn), mode = get_confusion_matrix(
        preds=preds,
        target=target,
        average=average,
        mdmc_average=mdmc_average,
        ignore_index=ignore_index,
        num_classes=num_classes,
        threshold=threshold,
        top_k=top_k,
        multiclass=multiclass,
    )

    b2 = beta**2
    fbeta = ((1 + b2) * tp) / ((1 + b2) * tp + b2 * fn + fp)

    if average in (None, "none", AverageMethod.NONE):
        pass
    elif average == AverageMethod.MICRO:
        pass
    elif average == AverageMethod.MACRO:
        fbeta = fbeta.mean()
    elif average == AverageMethod.WEIGHTED:
        weights = tp + fn
        fbeta = (weights * fbeta).sum() / weights.sum()
    else:
        raise ValueError(
            f"`average={average}` not yet supported. Chose between None, Micro, Macro, or Weighted"
        )

    return fbeta


def f1_score_ipu(
    preds: Tensor,
    target: Tensor,
    beta: float = 1.0,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
):
    """
    A modified version of the `torchmetrics.functional.classification.f_beta._fbeta_compute`
    that can ignore NaNs by giving them the same value for both `preds` and `target`.
    Used to calculate the f1_score on IPU with beta parameter equal to 1.0
    This allows it to work with compilation and IPUs since it doesn't modify the tensor's shape.

    Computes f_beta metric from stat scores: true positives, false positives, true negatives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives
        beta: The parameter `beta` (which determines the weight of recall in the combined score)
        ignore_index: Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method
        average: Defines the reduction that is applied
        mdmc_average: Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter)
    """

    return fbeta_score_ipu(
        preds,
        target,
        beta=beta,
        average=average,
        mdmc_average=mdmc_average,
        ignore_index=ignore_index,
        num_classes=num_classes,
        threshold=threshold,
        top_k=top_k,
        multiclass=multiclass,
    )


def mean_squared_error_ipu(preds: Tensor, target: Tensor, squared: bool) -> Tensor:
    """Computes mean squared error.

    Handles NaNs without reshaping tensors in order to work on IPU.

    Args:
        preds: estimated labels
        target: ground truth labels
        squared: returns RMSE value if set to False

    Return:
        Tensor with MSE
    """
    target = target.clone()
    preds = preds.clone()

    # Replace the nan-targets in the preds/target tensors by 0
    nan_targets = target.isnan()
    preds[nan_targets] = 0.0
    target[nan_targets] = 0.0

    # Compute the loss, and rescale by the number of nan elements
    loss = mean_squared_error(preds, target, squared)

    if squared:
        factor = nan_targets.numel() / ((~nan_targets).sum())
    else:
        factor = (nan_targets.numel() / ((~nan_targets).sum())).sqrt()

    loss = loss * factor

    return loss


def mean_absolute_error_ipu(preds: Tensor, target: Tensor) -> Tensor:
    """Computes mean absolute error.

    Handles NaNs without reshaping tensors in order to work on IPU.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with MAE
    """
    target = target.clone()
    preds = preds.clone()

    # Replace the nan-targets in the preds/target tensors by 0
    nan_targets = target.isnan()
    preds[nan_targets] = 0.0
    target[nan_targets] = 0.0

    # Compute the loss, and rescale by the number of nan elements
    loss = mean_absolute_error(preds, target)
    loss = loss * nan_targets.numel() / ((~nan_targets).sum())

    return loss
