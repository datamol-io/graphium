from typing import Union, Callable, Optional, Type, Tuple, Iterable
from copy import deepcopy
from loguru import logger


import torch
import torch.nn as nn
import mup.init as mupi
from mup import set_base_shapes

from graphium.nn.base_layers import FCLayer, MLP


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_ensemble: int,
        bias: bool = True,
        init_fn: Optional[Callable] = None,
    ):
        r"""
        Multiple linear layers that are applied in parallel with batched matrix multiplication with `torch.matmul`.

        Parameters:
            in_dim:
                Input dimension of the linear layers
            out_dim:
                Output dimension of the linear layers.
            num_ensemble:
                Number of linear layers in the ensemble.


        """
        super(EnsembleLinear, self).__init__()

        # Initialize weight and bias as learnable parameters
        self.weight = nn.Parameter(torch.Tensor(num_ensemble, out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_ensemble, 1, out_dim))
        else:
            self.register_parameter("bias", None)

        # Initialize parameters
        self.init_fn = init_fn if init_fn is not None else mupi.xavier_uniform_
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the linear layer using the `init_fn`.
        """
        set_base_shapes(self, None, rescale_params=False)  # Set the shapes of the tensors, useful for mup
        # Initialize weight using the provided initialization function
        self.init_fn(self.weight)

        # Initialize bias if present
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the batched linear transformation on the input features.

        Parameters:
                h: `torch.Tensor[B, Din]` or `torch.Tensor[..., 1, B, Din]` or `torch.Tensor[..., L, B, Din]`:
                    Input feature tensor, before the batched linear transformation.
                    `Din` is the number of input features, `B` is the batch size, and `L` is the number of linear layers.

        Returns:
                `torch.Tensor[..., L, B, Dout]`:
                    Output feature tensor, after the batched linear transformation.
                    `Dout` is the number of output features, , `B` is the batch size, and `L` is the number of linear layers.
        """

        # Perform the linear transformation using torch.matmul
        h = torch.matmul(self.weight, h.transpose(-1, -2)).transpose(-1, -2)

        # Add bias if present
        if self.bias is not None:
            h += self.bias

        return h


class EnsembleFCLayer(FCLayer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_ensemble: int,
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        bias: bool = True,
        init_fn: Optional[Callable] = None,
        is_readout_layer: bool = False,
        droppath_rate: float = 0.0,
    ):
        r"""
        Multiple fully connected layers running in parallel.
        This layer is centered around a `torch.nn.Linear` module.
        The order in which transformations are applied is:

        - Dense Layer
        - Activation
        - Dropout (if applicable)
        - Batch Normalization (if applicable)

        Parameters:
            in_dim:
                Input dimension of the layer (the `torch.nn.Linear`)
            out_dim:
                Output dimension of the layer.
            num_ensemble:
                Number of linear layers in the ensemble.
            dropout:
                The ratio of units to dropout. No dropout by default.
            activation:
                Activation function to use.
            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function
            bias:
                Whether to enable bias in for the linear layer.
            init_fn:
                Initialization function to use for the weight of the layer. Default is
                $$\mathcal{U}(-\sqrt{k}, \sqrt{k})$$ with $$k=\frac{1}{ \text{in_dim}}$$
            is_readout_layer: Whether the layer should be treated as a readout layer by replacing of `torch.nn.Linear`
                by `mup.MuReadout` from the muTransfer method https://github.com/microsoft/mup

            droppath_rate:
                stochastic depth drop rate, between 0 and 1, see https://arxiv.org/abs/1603.09382
        Attributes:
            dropout (int):
                The ratio of units to dropout.
            normalization (None or Callable):
                Normalization layer
            linear (`torch.nn.Linear`):
                The linear layer
            activation (`torch.nn.Module`):
                The activation layer
            init_fn (Callable):
                Initialization function used for the weight of the layer
            in_dim (int):
                Input dimension of the linear layer
            out_dim (int):
                Output dimension of the linear layer
        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            bias=bias,
            init_fn=init_fn,
            is_readout_layer=is_readout_layer,
            droppath_rate=droppath_rate,
        )

        # Linear layer, or MuReadout layer
        if not is_readout_layer:
            self.linear = EnsembleLinear(
                in_dim, out_dim, num_ensemble=num_ensemble, bias=bias, init_fn=init_fn
            )
        else:
            self.linear = EnsembleMuReadoutGraphium(in_dim, out_dim, num_ensemble=num_ensemble, bias=bias)

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        """
        Reset the parameters of the linear layer using the `init_fn`.
        """
        set_base_shapes(self, None, rescale_params=False)  # Set the shapes of the tensors, useful for mup
        self.linear.reset_parameters()

    def __repr__(self):
        rep = super().__repr__()
        rep = rep[:-1] + f", num_ensemble={self.linear.weight.shape[0]})"
        return rep


class EnsembleMuReadoutGraphium(EnsembleLinear):
    """
    This layer implements an ensemble version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_ensemble: int,
        bias: bool = True,
        init_fn: Optional[Callable] = None,
        readout_zero_init=False,
        output_mult=1.0,
    ):
        self.in_dim = in_dim
        self.output_mult = output_mult
        self.readout_zero_init = readout_zero_init
        self._base_width = in_dim
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            num_ensemble=num_ensemble,
            bias=bias,
            init_fn=init_fn,
        )

    def reset_parameters(self) -> None:
        if self.readout_zero_init:
            self.weight.data[:] = 0
            if self.bias is not None:
                self.bias.data[:] = 0
        else:
            super().reset_parameters()

    def width_mult(self):
        assert hasattr(self.weight, "infshape"), (
            "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
            "switch to distributed training with "
            "torch.nn.parallel.DistributedDataParallel instead"
        )
        return self.weight.infshape.width_mult()

    def _rescale_parameters(self):
        """Rescale parameters to convert SP initialization to μP initialization.

        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult() ** 0.5
        self.weight.data *= self.width_mult() ** 0.5
        self._has_rescaled_params = True

    def forward(self, x):
        return super().forward(self.output_mult * x / self.width_mult())

    @property
    def absolute_width(self):
        return float(self.in_dim)

    @property
    def base_width(self):
        return self._base_width

    @base_width.setter
    def base_width(self, val):
        if val is None:
            return
        assert isinstance(
            val, (int, torch.int, torch.long)
        ), f"`base_width` must be None, int or long, provided {val} of type {type(val)}"
        self._base_width = val

    def width_mult(self):
        return self.absolute_width / self.base_width


class EnsembleMLP(MLP):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Union[Iterable[int], int],
        out_dim: int,
        num_ensemble: int,
        depth: Optional[int] = None,
        reduction: Optional[Union[str, Callable]] = "none",
        activation: Union[str, Callable] = "relu",
        last_activation: Union[str, Callable] = "none",
        dropout: float = 0.0,
        last_dropout: float = 0.0,
        normalization: Union[Type[None], str, Callable] = "none",
        last_normalization: Union[Type[None], str, Callable] = "none",
        first_normalization: Union[Type[None], str, Callable] = "none",
        last_layer_is_readout: bool = False,
        droppath_rate: float = 0.0,
        constant_droppath_rate: bool = True,
    ):
        r"""
        Simple multi-layer perceptron, built of a series of FCLayers

        Parameters:
            in_dim:
                Input dimension of the MLP
            hidden_dims:
                Either an integer specifying all the hidden dimensions,
                or a list of dimensions in the hidden layers.
            out_dim:
                Output dimension of the MLP.
            num_ensemble:
                Number of MLPs that run in parallel.
            depth:
                If `hidden_dims` is an integer, `depth` is 1 + the number of
                hidden layers to use.
                If `hidden_dims` is a list, then
                `depth` must be `None` or equal to `len(hidden_dims) + 1`
            reduction:
                Reduction to use at the end of the MLP. Choices:

                - "none" or `None`: No reduction
                - "mean": Mean reduction
                - "sum": Sum reduction
                - "max": Max reduction
                - "min": Min reduction
                - "median": Median reduction
                - `Callable`: Any callable function. Must take `dim` as a keyword argument.
           activation:
                Activation function to use in all the layers except the last.
                if `layers==1`, this parameter is ignored
            last_activation:
                Activation function to use in the last layer.
            dropout:
                The ratio of units to dropout. Must be between 0 and 1
            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization in the hidden layers.
                - `Callable`: Any callable function

                if `layers==1`, this parameter is ignored
            last_normalization:
                Norrmalization to use **after the last layer**. Same options as `normalization`.
            first_normalization:
                Norrmalization to use in **before the first layer**. Same options as `normalization`.
            last_dropout:
                The ratio of units to dropout at the last layer.
            last_layer_is_readout: Whether the last layer should be treated as a readout layer.
                Allows to use the `mup.MuReadout` from the muTransfer method https://github.com/microsoft/mup
            droppath_rate:
                stochastic depth drop rate, between 0 and 1.
                See https://arxiv.org/abs/1603.09382
            constant_droppath_rate:
                If `True`, drop rates will remain constant accross layers.
                Otherwise, drop rates will vary stochastically.
                See `DropPath.get_stochastic_drop_rate`
        """

        super().__init__(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            depth=depth,
            activation=activation,
            last_activation=last_activation,
            dropout=dropout,
            last_dropout=last_dropout,
            normalization=normalization,
            last_normalization=last_normalization,
            first_normalization=first_normalization,
            last_layer_is_readout=last_layer_is_readout,
            droppath_rate=droppath_rate,
            constant_droppath_rate=constant_droppath_rate,
            fc_layer=EnsembleFCLayer,
            fc_layer_kwargs={"num_ensemble": num_ensemble},
        )

        self.reduction_fn = self._parse_reduction(reduction)

    def _parse_reduction(self, reduction: Optional[Union[str, Callable]]) -> Optional[Callable]:
        r"""
        Parse the reduction argument.
        """

        if isinstance(reduction, str):
            reduction = reduction.lower()
        if reduction is None or reduction == "none":
            return None
        elif reduction == "mean":
            return torch.mean
        elif reduction == "sum":
            return torch.sum
        elif reduction == "max":

            def max_vals(x, dim):
                return torch.max(x, dim=dim).values

            return max_vals
        elif reduction == "min":

            def min_vals(x, dim):
                return torch.min(x, dim=dim).values

            return min_vals
        elif reduction == "median":

            def median_vals(x, dim):
                return torch.median(x, dim=dim).values

            return median_vals
        elif callable(reduction):
            return reduction
        else:
            raise ValueError(f"Unknown reduction {reduction}")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the ensemble MLP on the input features, then reduce the output if specified.

        Parameters:

            h: `torch.Tensor[B, Din]` or `torch.Tensor[..., 1, B, Din]` or `torch.Tensor[..., L, B, Din]`:

                Input feature tensor, before the MLP.
                `Din` is the number of input features, `B` is the batch size, and `L` is the number of ensembles.

        Returns:

            `torch.Tensor[..., L, B, Dout]` or `torch.Tensor[..., B, Dout]`:

                Output feature tensor, after the MLP.
                `Dout` is the number of output features, `B` is the batch size, and `L` is the number of ensembles.
                `L` is removed if a reduction is specified.
        """
        h = super().forward(h)
        if self.reduction_fn is not None:
            h = self.reduction_fn(h, dim=-3)
        return h

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        rep = super().__repr__()
        rep = rep[:-1] + f", num_ensemble={self.layers[0].linear.weight.shape[0]})"
