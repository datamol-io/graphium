import abc
import inspect
from numbers import Real
from typing import Optional


class MupMixin(abc.ABC):
    @abc.abstractmethod
    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: Optional[bool] = None):
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        This is done using the `scale_kwargs()` method with `scale_factor = 1 / divide_factor`.

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension for the nodes. If None, the default for scale_kwargs is used
        Returns:
            kwargs: Dictionary of parameters to be used to instanciate the base model divided by the factor
        """
        ...

    def scale_kwargs(self, scale_factor: Real, scale_in_dim: bool = False):
        """
        Create a "scaled" version of the module where the hidden dims are scaled as in muTransfer.

        This can be used with `scale_factor` < 1 to create a "base" model to extract shape
        information as in the `mup` package or with `scale_factor` > 1 to create a scaled model
        to which optimal hyperparameters can be "muTransferred" from the original model

        Parameters:
            scale_factor: Factor by which to scale the width.
            scale_in_dim: Whether to factor the input dimension for the nodes

        Returns:
            kwargs: Dictionary of parameters to be used to instantiate the scaled model
        """

        divide_factor = 1 / scale_factor

        if not scale_in_dim:
            return self.make_mup_base_kwargs(divide_factor=divide_factor)

        # If scale_in_dim passed, need to check it can be forwarded
        try:
            return self.make_mup_base_kwargs(divide_factor=divide_factor, factor_in_dim=scale_in_dim)
        except TypeError as e:
            raise RuntimeError(
                "This error may have been caused by passing scale_in_dim to scale_kwargs "
                "for a class that does not support passing factor_in_dim to make_mup_base_kwargs, "
                "which cannot be done"
            ) from e
