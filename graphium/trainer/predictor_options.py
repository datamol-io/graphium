r"""Data classes to group together related arguments for the creation of a Predictor Module."""


"""
Replace the usage of **kwargs by adding checks to make sure that everything is type-safe.
    Stricter typing is important because:
        - It makes finding bugs easier since incorrect types will cause an obvious error.
        - Static analysis becomes easier, and the IDE can give hints about errors in the code before runtime.
Add the post-init function to do the checks immediately.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from inspect import signature, isclass

from torch import nn

from graphium.utils.spaces import LOSS_DICT
from graphium.utils.spaces import SCHEDULER_DICT


@dataclass
class ModelOptions:
    r"""
    This data class stores the arguments necessary to instantiate a model for the Predictor.

    Parameters:
        model_class:
            pytorch module used to create a model

        model_kwargs:
            Key-word arguments used to initialize the model from `model_class`.
    """
    model_class: Type[nn.Module]
    model_kwargs: Dict[str, Any]


@dataclass
class OptimOptions:
    r"""
    This data class stores the arguments necessary to configure the optimizer for the Predictor.

    Parameters:
        optim_kwargs:
            Dictionnary used to initialize the optimizer, with possible keys below.

            - lr `float`: Learning rate (Default=`1e-3`)
            - weight_decay `float`: Weight decay used to regularize the optimizer (Default=`0.`)

        torch_scheduler_kwargs:
            Dictionnary for the scheduling of learning rate, with possible keys below.

            - type `str`: Type of the learning rate to use from pytorch. Examples are
                `'ReduceLROnPlateau'` (default), `'CosineAnnealingWarmRestarts'`, `'StepLR'`, etc.
            - **kwargs: Any other argument for the learning rate scheduler

        scheduler_kwargs:
            Dictionnary for the scheduling of the learning rate modification used by pytorch-lightning

            - monitor `str`: metric to track (Default=`"loss/val"`)
            - interval `str`: Whether to look at iterations or epochs (Default=`"epoch"`)
            - strict `bool`: if set to True will enforce that value specified in monitor is available
                while trying to call scheduler.step(), and stop training if not found. If False will
                only give a warning and continue training (without calling the scheduler). (Default=`True`)
            - frequency `int`: **TODO: NOT REALLY SURE HOW IT WORKS!** (Default=`1`)

        scheduler_class: The class to use for the scheduler, or the str representing the scheduler.

    """
    optim_kwargs: Optional[Dict[str, Any]] = None
    torch_scheduler_kwargs: Optional[Dict[str, Any]] = None
    scheduler_kwargs: Optional[Dict[str, Any]] = None
    scheduler_class: Optional[Union[str, Type]] = None

    # Instead of passing a dictionary to be processed by the predictor,
    # this class will process the dictionary in advance and return the optimizer
    def set_kwargs(self):
        # Set the parameters and default value for the optimizer, and check values
        if self.optim_kwargs is None:
            self.optim_kwargs = {}
        self.optim_kwargs.setdefault("lr", 1e-3)
        self.optim_kwargs.setdefault("weight_decay", 0.0)
        assert self.optim_kwargs["lr"] > 0
        assert self.optim_kwargs["weight_decay"] >= 0

        # Set the lightning scheduler
        if self.scheduler_kwargs is None:
            self.scheduler_kwargs = {}
        self.scheduler_kwargs.setdefault("interval", "epoch")
        self.scheduler_kwargs.setdefault("monitor", "loss/val")
        self.scheduler_kwargs.setdefault("mode", "min")
        self.scheduler_kwargs.setdefault("frequency", 1)
        self.scheduler_kwargs.setdefault("strict", True)

        # Set the pytorch scheduler arguments
        if self.torch_scheduler_kwargs is None:
            self.torch_scheduler_kwargs = {}
        self.torch_scheduler_kwargs.setdefault("module_type", "ReduceLROnPlateau")

        # Get the class for the scheduler
        scheduler_class = self.torch_scheduler_kwargs.get("module_type", None)
        if self.scheduler_class is None:
            if isinstance(scheduler_class, str):
                self.scheduler_class = SCHEDULER_DICT[scheduler_class]
            elif isclass(scheduler_class):
                self.scheduler_class = scheduler_class
            else:
                raise TypeError("`scheduler_class` should be a str or a class")

        # Add the `monitor` and `mode` variables
        sig = signature(self.scheduler_class.__init__)
        key_args = [p.name for p in sig.parameters.values()]
        if "monitor" in key_args:
            self.torch_scheduler_kwargs.setdefault("monitor", self.scheduler_kwargs["monitor"])
        if "mode" in key_args:
            self.torch_scheduler_kwargs.setdefault("mode", self.scheduler_kwargs["mode"])


@dataclass
class EvalOptions:
    r"""
    This data class stores the arguments necessary to instantiate a model for the Predictor.

    Parameters:
        loss_fun:
            Loss function used during training.
            Acceptable strings are graphium.utils.spaces.LOSS_DICT.keys().
            If a dict, must contain a 'name' key with one of the acceptable loss function strings
            as a value. The rest of the dict will be used as the arguments passed to the loss object.
            Otherwise, a callable object must be provided, with a method `loss_fun._get_name()`.

        metrics:
            A dictionnary of metrics to compute on the prediction, other than the loss function.
            These metrics will be logged into WandB or other.

        metrics_on_progress_bar:
            The metrics names from `metrics` to display also on the progress bar of the training

        metrics_on_training_set:
            The metrics names from `metrics` to be computed on the training set for each iteration.
            If `None`, all the metrics are computed. Using less metrics can significantly improve
            performance, depending on the number of readouts.
    """
    loss_fun: Union[str, Dict, Callable]
    metrics: Dict[str, Callable] = None
    metrics_on_progress_bar: List[str] = field(default_factory=List[str])
    metrics_on_training_set: Optional[List[str]] = None

    def check_metrics_validity(self):
        """
        Check that the metrics for the progress_par and training_set are valid
        """
        if self.metrics_on_progress_bar is not None:
            selected = set(self.metrics_on_progress_bar)
            assert selected.issubset(
                set(self.metrics.keys())
            ), f"Metrics {selected - set(self.metrics.keys())} not in `metrics` with choices {set(self.metrics.keys())}"

        if self.metrics_on_training_set is not None:
            selected = set(self.metrics_on_training_set)
            assert selected.issubset(
                set(self.metrics.keys())
            ), f"Metrics {selected - set(self.metrics.keys())} not in `metrics` with choices {set(self.metrics.keys())}"

    # Parse before or after?
    @staticmethod
    def parse_loss_fun(loss_fun: Union[str, Dict, Callable]) -> Callable:
        r"""
        Parse the loss function from a string or a dict

        Parameters:
            loss_fun:
                A callable corresponding to the loss function, a string specifying the loss
                function from `LOSS_DICT`, or a dict containing a key 'name' specifying the
                loss function, and the rest of the dict used as the arguments for the loss.
                Accepted strings are: graphium.utils.spaces.LOSS_DICT.keys().

        Returns:
            Callable:
                Function or callable to compute the loss, takes `preds` and `targets` as inputs.
        """

        if isinstance(loss_fun, str):
            if loss_fun not in LOSS_DICT.keys():
                raise ValueError(
                    f"`loss_fun` expected to be one of the strings in {LOSS_DICT.keys()}. "
                    f"Provided: {loss_fun}."
                )
            loss_fun = LOSS_DICT[loss_fun]()
        elif isinstance(loss_fun, dict):
            if loss_fun.get("name") is None:
                raise ValueError(f"`loss_fun` expected to have a key 'name'.")
            if loss_fun["name"] not in LOSS_DICT.keys():
                raise ValueError(
                    f"`loss_fun['name']` expected to be one of the strings in {LOSS_DICT.keys()}. "
                    f"Provided: {loss_fun}."
                )
            loss_fun = deepcopy(loss_fun)
            loss_name = loss_fun.pop("name")
            loss_fun = LOSS_DICT[loss_name](**loss_fun)
        elif not callable(loss_fun):
            raise ValueError(f"`loss_fun` must be `str`, `dict` or `callable`. Provided: {type(loss_fun)}")

        return loss_fun


@dataclass
class FlagOptions:
    r"""
    This data class stores the arguments necessary to instantiate a model for the Predictor.

    Parameters:
        flag_kwargs:
            Keyword arguments used for FLAG, and adversarial data augmentation for graph networks.
            See: https://arxiv.org/abs/2010.09891

            - n_steps: An integer that specifies the number of ascent steps when running FLAG during training.
                Default value of 0 trains GNNs without FLAG, and any value greater than 0 will use FLAG with that
                many iterations.

            - alpha: A float that specifies the ascent step size when running FLAG. Default=0.01
    """
    flag_kwargs: Dict[str, Any] = None

    # Set the parameters and default values for the FLAG adversarial augmentation, and check values
    def set_kwargs(self):
        if self.flag_kwargs is None:
            self.flag_kwargs = {}
        self.flag_kwargs.setdefault("alpha", 0.01)
        self.flag_kwargs.setdefault("n_steps", 0)
        assert isinstance(self.flag_kwargs["n_steps"], int) and (self.flag_kwargs["n_steps"] >= 0)
        assert self.flag_kwargs["alpha"] >= 0
