r"""Data classes to group together related arguments for the creation of a Predictor Module."""


"""
Replace the usage of **kwargs by adding checks to make sure that everything is type-safe.
    Stricter typing is important because:
        - It makes finding bugs easier since incorrect types will cause an obvious error.
        - Static analysis becomes easier, and the IDE can give hints about errors in the code before runtime.
Add the post-init function to do the checks immediately.
"""

from dataclasses import dataclass, field
from loguru import logger
from typing import Any, Callable, Dict, List, Optional, Type, Union

from torch import nn

from goli.utils.spaces import LOSS_DICT


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
    """
    optim_kwargs: Optional[Dict[str, Any]] = None
    # lr_reduce_on_plateau_kwargs: Optional[Dict[str, Any]] = None
    torch_scheduler_kwargs: Optional[Dict[str, Any]] = None
    scheduler_kwargs: Optional[Dict[str, Any]] = None

    # Instead of passing a dictionary to be processed by the predictor,
    # this class will process the dictionary in advance and return the optimizer
    def set_kwargs(self):
        # Set the parameters and default value for the optimizer, and check values
        optim_kwargs = (
            self.optim_kwargs
        )  # Save kwargs that were initially given. But does this mess with the fact that the class attribute is also called optim_kwargs?
        self.optim_kwargs = {"lr": 1e-3, "weight_decay": 0.0}
        if optim_kwargs is not None:
            self.optim_kwargs.update(optim_kwargs)
        assert self.optim_kwargs["lr"] > 0
        assert self.optim_kwargs["weight_decay"] >= 0

        # Set the lightning scheduler
        scheduler_kwargs = self.scheduler_kwargs
        self.scheduler_kwargs = {
            "interval": "epoch",
            "monitor": "loss/val",
            "mode": "min",
            "frequency": 1,
            "strict": True,
        }
        if scheduler_kwargs is not None:
            self.scheduler_kwargs.update(scheduler_kwargs)

        # Set the pytorch scheduler arguments
        torch_scheduler_kwargs = self.torch_scheduler_kwargs
        if torch_scheduler_kwargs is None:
            self.torch_scheduler_kwargs = {}
        else:
            self.torch_scheduler_kwargs = torch_scheduler_kwargs
        self.torch_scheduler_kwargs.setdefault("module_type", "ReduceLROnPlateau")


@dataclass
class EvalOptions:
    r"""
    This data class stores the arguments necessary to instantiate a model for the Predictor.

        Parameters:
            loss_fun:
                Loss function used during training.
                Acceptable strings are 'mse', 'bce', 'mae', 'cosine'.
                Otherwise, a callable object must be provided, with a method `loss_fun._get_name()`.

            metrics:
                A dictionnary of metrics to compute on the prediction, other than the loss function.
                These metrics will be logged into TensorBoard.

            metrics_on_progress_bar:
                The metrics names from `metrics` to display also on the progress bar of the training

            metrics_on_training_set:
                The metrics names from `metrics` to be computed on the training set for each iteration.
                If `None`, all the metrics are computed. Using less metrics can significantly improve
                performance, depending on the number of readouts.
    """
    loss_fun: Union[str, Callable]
    metrics: Dict[str, Callable] = None
    metrics_on_progress_bar: List[str] = field(default_factory=List[str])
    metrics_on_training_set: Optional[List[str]] = None

    # Parse before or after?
    @staticmethod
    def parse_loss_fun(loss_fun: Union[str, Callable]) -> Callable:
        r"""
        Parse the loss function from a string

        Parameters:
            loss_fun:
                A callable corresponding to the loss function or a string
                specifying the loss function from `LOSS_DICT`. Accepted strings are:
                "mse", "bce", "l1", "mae", "cosine".

        Returns:
            Callable:
                Function or callable to compute the loss, takes `preds` and `targets` as inputs.
        """

        if isinstance(loss_fun, str):
            loss_fun = LOSS_DICT[loss_fun]
        elif not callable(loss_fun):
            raise ValueError(f"`loss_fun` must be `str` or `callable`. Provided: {type(loss_fun)}")

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
    #    flag_kwargs_set: Dict[str, Any] = field(init=False, repr=True)

    #    def __post_init__(self):
    #        self.flag_kwargs_set = {"alpha": 0.01, "n_steps": 0}
    #        if self.flag_kwargs is not None:
    #            self.flag_kwargs_set.update(self.flag_kwargs)
    #        assert isinstance(self.flag_kwargs_set["n_steps"], int) and (self.flag_kwargs_set["n_steps"] >= 0)
    #        assert self.flag_kwargs_set["alpha"] > 0

    # Set the parameters and default values for the FLAG adversarial augmentation, and check values
    def set_kwargs(self):
        flag_kwargs = self.flag_kwargs
        self.flag_kwargs = {"alpha": 0.01, "n_steps": 0}
        if flag_kwargs is not None:
            self.flag_kwargs.update(flag_kwargs)
        assert isinstance(self.flag_kwargs["n_steps"], int) and (self.flag_kwargs["n_steps"] >= 0)
        assert self.flag_kwargs["alpha"] >= 0
