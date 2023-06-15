"""
Different types of residual connections, including None, Simple (ResNet-like),
Concat and DenseNet
"""

import abc
import torch
import torch.nn as nn
from typing import List, Union, Callable

from graphium.nn.base_layers import FCLayer
from graphium.utils.decorators import classproperty


class ResidualConnectionBase(nn.Module):
    def __init__(self, skip_steps: int = 1):
        r"""
        Abstract class for the residual connections. Using this class,
        we implement different types of residual connections, such as
        the ResNet, weighted-ResNet, skip-concat and DensNet.

        The following methods must be implemented in a children class

        - ``h_dim_increase_type()``
        - ``has_weights()``

        Parameters:

            skip_steps: int
                The number of steps to skip between the residual connections.
                If `1`, all the layers are connected. If `2`, half of the
                layers are connected.
        """

        super().__init__()
        self.skip_steps = skip_steps

    def _bool_apply_skip_step(self, step_idx: int):
        r"""
        Whether to apply the skip connection, depending on the
        ``step_idx`` and ``self.skip_steps``.

        Parameters:

            step_idx: int
                The current layer step index.

        """
        return (self.skip_steps != 0) and ((step_idx % self.skip_steps) == 0)

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        return f"{self.__class__.__name__}(skip_steps={self.skip_steps})"

    @classproperty
    @abc.abstractmethod
    def h_dim_increase_type(cls):
        r"""
        How does the dimension of the output features increases after each layer?

        Returns:

            h_dim_increase_type: None or str
                - ``None``: The dimension of the output features do not change at each layer.
                E.g. ResNet.

                - "previous": The dimension of the output features is the concatenation of
                the previous layer with the new layer.

                - "cumulative": The dimension of the output features is the concatenation
                of all previous layers.

        """
        ...

    def get_true_out_dims(self, out_dims: List) -> List:
        r"""
        find the true output dimensions
        Parameters:
            out_dims: List
        Returns:
            true_out_dims: List
        """
        true_out_dims = [out_dims[0]]
        out_dims_at_skip = [out_dims[0]]
        for ii in range(1, len(out_dims) - 1):
            # For the `None` type, don't change the output dims
            if self.h_dim_increase_type is None:
                true_out_dims.append(out_dims[ii])

            # For the "previous" type, add the previous layers when the skip connection applies
            elif self.h_dim_increase_type == "previous":
                if self._bool_apply_skip_step(step_idx=ii):
                    true_out_dims.append(out_dims[ii] + out_dims_at_skip[-1])
                    out_dims_at_skip.append(out_dims[ii])
                else:
                    true_out_dims.append(out_dims[ii])

            # For the "cumulative" type, add all previous layers when the skip connection applies
            elif self.h_dim_increase_type == "cumulative":
                if self._bool_apply_skip_step(step_idx=ii):
                    true_out_dims.append(out_dims[ii] + out_dims_at_skip[-1])
                    out_dims_at_skip.append(true_out_dims[ii])
                else:
                    true_out_dims.append(out_dims[ii])
            else:
                raise ValueError(f"undefined value: {self.h_dim_increase_type}")

        return true_out_dims

    @classproperty
    @abc.abstractmethod
    def has_weights(cls):
        r"""
        Returns:

            has_weights: bool
                Whether the residual connection uses weights

        """
        ...


class ResidualConnectionNone(ResidualConnectionBase):
    r"""
    No residual connection.
    This class is only used for simpler code compatibility
    """

    def __init__(self, skip_steps: int = 1):
        super().__init__(skip_steps=skip_steps)

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        return f"{self.__class__.__name__}"

    @classproperty
    def h_dim_increase_type(cls):
        r"""
        Returns:

            None:
                The dimension of the output features do not change at each layer.
        """

        return None

    @classproperty
    def has_weights(cls):
        r"""
        Returns:

            False
                The current class does not use weights

        """
        return False

    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Ignore the skip connection.

        Returns:

            h: torch.Tensor(..., m)
                Return same as input.

            h_prev: torch.Tensor(..., m)
                Return same as input.

        """
        return h, h_prev


class ResidualConnectionSimple(ResidualConnectionBase):
    def __init__(self, skip_steps: int = 1):
        r"""
        Class for the simple residual connections proposed by ResNet,
        where the current layer output is summed to a
        previous layer output.

        Parameters:

            skip_steps: int
                The number of steps to skip between the residual connections.
                If `1`, all the layers are connected. If `2`, half of the
                layers are connected.
        """
        super().__init__(skip_steps=skip_steps)

    @classproperty
    def h_dim_increase_type(cls):
        r"""
        Returns:

            None:
                The dimension of the output features do not change at each layer.
        """

        return None

    @classproperty
    def has_weights(cls):
        r"""
        Returns:

            False
                The current class does not use weights

        """
        return False

    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Add ``h`` with the previous layers with skip connection ``h_prev``,
        similar to ResNet.

        Parameters:

            h: torch.Tensor(..., m)
                The current layer features

            h_prev: torch.Tensor(..., m), None
                The features from the previous layer with a skip connection.
                At ``step_idx==0``, ``h_prev`` can be set to ``None``.

            step_idx: int
                Current layer index or step index in the forward loop of the architecture.

        Returns:

            h: torch.Tensor(..., m)
                Either return ``h`` unchanged, or the sum with
                on ``h_prev``, depending on the ``step_idx`` and ``self.skip_steps``.

            h_prev: torch.Tensor(..., m)
                Either return ``h_prev`` unchanged, or the same value as ``h``,
                depending on the ``step_idx`` and ``self.skip_steps``.

        """
        if self._bool_apply_skip_step(step_idx):
            if step_idx > 0:
                h = h + h_prev
            h_prev = h

        return h, h_prev


class ResidualConnectionWeighted(ResidualConnectionBase):
    def __init__(
        self,
        out_dims,
        skip_steps: int = 1,
        dropout=0.0,
        activation: Union[str, Callable] = "none",
        normalization="none",
        bias=False,
    ):
        r"""
        Class for the simple residual connections proposed by ResNet,
        with an added layer in the residual connection itself.
        The layer output is summed to a a non-linear transformation
        of a previous layer output.

        Parameters:

            skip_steps: int
                The number of steps to skip between the residual connections.
                If `1`, all the layers are connected. If `2`, half of the
                layers are connected.

            out_dims: list(int)
                list of all output dimensions for the network
                that will use this residual connection.
                E.g. ``out_dims = [4, 8, 8, 8, 2]``.

            dropout: float
                value between 0 and 1.0 representing the percentage of dropout
                to use in the weights

            activation: str, Callable
                The activation function to use after the skip weights

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization in the hidden layers.
                - `Callable`: Any callable function

            bias: bool
                Whether to apply add a bias after the weights

        """

        super().__init__(skip_steps=skip_steps)

        self.residual_list = nn.ModuleList()
        self.skip_count = 0
        self.out_dims = out_dims

        for ii in range(0, len(self.out_dims) - 1, self.skip_steps):
            this_dim = self.out_dims[ii]
            self.residual_list.append(
                FCLayer(
                    this_dim,
                    this_dim,
                    activation=activation,
                    dropout=dropout,
                    normalization=normalization,
                    bias=False,
                )
            )

    @classproperty
    def h_dim_increase_type(cls):
        r"""
        Returns:

            None:
                The dimension of the output features do not change at each layer.
        """
        return None

    @classproperty
    def has_weights(cls):
        r"""
        Returns:

            True
                The current class uses weights

        """
        return True

    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Add ``h`` with the previous layers with skip connection ``h_prev``, after
        a feed-forward layer.

        Parameters:

            h: torch.Tensor(..., m)
                The current layer features

            h_prev: torch.Tensor(..., m), None
                The features from the previous layer with a skip connection.
                At ``step_idx==0``, ``h_prev`` can be set to ``None``.

            step_idx: int
                Current layer index or step index in the forward loop of the architecture.

        Returns:

            h: torch.Tensor(..., m)
                Either return ``h`` unchanged, or the sum with the output of a NN layer
                on ``h_prev``, depending on the ``step_idx`` and ``self.skip_steps``.

            h_prev: torch.Tensor(..., m)
                Either return ``h_prev`` unchanged, or the same value as ``h``,
                depending on the ``step_idx`` and ``self.skip_steps``.

        """

        if self._bool_apply_skip_step(step_idx):
            if step_idx > 0:
                h = h + self.residual_list[self.skip_count].forward(h_prev)
                self.skip_count += 1
            h_prev = h

        return h, h_prev

    def _bool_apply_skip_step(self, step_idx: int):
        return super()._bool_apply_skip_step(step_idx) and self.skip_count < len(self.residual_list)


class ResidualConnectionConcat(ResidualConnectionBase):
    def __init__(self, skip_steps: int = 1):
        r"""
        Class for the simple residual connections proposed but where
        the skip connection features are concatenated to the current
        layer features.

        Parameters:

            skip_steps: int
                The number of steps to skip between the residual connections.
                If `1`, all the layers are connected. If `2`, half of the
                layers are connected.
        """

        super().__init__(skip_steps=skip_steps)

    @classproperty
    def h_dim_increase_type(cls):
        r"""
        Returns:

            "previous":
                The dimension of the output layer is the concatenation with the previous layer.
        """

        return "previous"

    @classproperty
    def has_weights(cls):
        r"""
        Returns:

            False
                The current class does not use weights

        """
        return False

    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Concatenate ``h`` with the previous layers with skip connection ``h_prev``.

        Parameters:

            h: torch.Tensor(..., m)
                The current layer features

            h_prev: torch.Tensor(..., n), None
                The features from the previous layer with a skip connection.
                Usually, we have ``n`` equal to ``m``.
                At ``step_idx==0``, ``h_prev`` can be set to ``None``.

            step_idx: int
                Current layer index or step index in the forward loop of the architecture.

        Returns:

            h: torch.Tensor(..., m) or torch.Tensor(..., m + n)
                Either return ``h`` unchanged, or the concatenation
                with ``h_prev``, depending on the ``step_idx`` and ``self.skip_steps``.

            h_prev: torch.Tensor(..., m) or torch.Tensor(..., m + n)
                Either return ``h_prev`` unchanged, or the same value as ``h``,
                depending on the ``step_idx`` and ``self.skip_steps``.

        """

        if self._bool_apply_skip_step(step_idx):
            h_in = h
            if step_idx > 0:
                h = torch.cat([h, h_prev], dim=-1)
            h_prev = h_in

        return h, h_prev


class ResidualConnectionDenseNet(ResidualConnectionBase):
    def __init__(self, skip_steps: int = 1):
        r"""
        Class for the residual connections proposed by DenseNet, where
        all previous skip connection features are concatenated to the current
        layer features.

        Parameters:

            skip_steps: int
                The number of steps to skip between the residual connections.
                If `1`, all the layers are connected. If `2`, half of the
                layers are connected.
        """

        super().__init__(skip_steps=skip_steps)

    @classproperty
    def h_dim_increase_type(cls):
        r"""
        Returns:

            "cumulative":
                The dimension of the output layer is the concatenation of all the previous layer.
        """

        return "cumulative"

    @classproperty
    def has_weights(cls):
        r"""
        Returns:

            False
                The current class does not use weights

        """
        return False

    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Concatenate ``h`` with all the previous layers with skip connection ``h_prev``.

        Parameters:

            h: torch.Tensor(..., m)
                The current layer features

            h_prev: torch.Tensor(..., n), None
                The features from the previous layers.
                n = ((step_idx // self.skip_steps) + 1) * m

                At ``step_idx==0``, ``h_prev`` can be set to ``None``.

            step_idx: int
                Current layer index or step index in the forward loop of the architecture.

        Returns:

            h: torch.Tensor(..., m) or torch.Tensor(..., m + n)
                Either return ``h`` unchanged, or the concatenation
                with ``h_prev``, depending on the ``step_idx`` and ``self.skip_steps``.

            h_prev: torch.Tensor(..., m) or torch.Tensor(..., m + n)
                Either return ``h_prev`` unchanged, or the same value as ``h``,
                depending on the ``step_idx`` and ``self.skip_steps``.

        """

        if self._bool_apply_skip_step(step_idx):
            if step_idx > 0:
                h = torch.cat([h, h_prev], dim=-1)
            h_prev = h

        return h, h_prev


class ResidualConnectionRandom(ResidualConnectionBase):
    def __init__(self, skip_steps=1, out_dims: List[int] = None, num_layers: int = None):
        r"""
        Class for the random residual connection, where each layer is connected
        to each following layer with a random weight between 0 and 1.
        Parameters:
            skip_steps:
                Parameter only there for compatibility with other classes of the same parent.
            out_dims:
                The list of output dimensions. Only required to get the number
                of layers. Must be provided if `num_layers` is None.
            num_layers:
                The number of layers. Must be provided if `out_dims` is None.
        """
        if skip_steps != 1:
            raise ValueError("Only `skip_step=1` is implemented")
        super().__init__(skip_steps=skip_steps)

        if out_dims is not None:
            if num_layers is not None:
                assert num_layers == len(out_dims)
            num_layers = len(out_dims)
        if num_layers is None:
            raise ValueError("Either `out_dims` or `num_layers` must be provided")
        self.num_layers = num_layers

        self.random_dict_weights = {}
        for ii in range(1, self.num_layers):
            random_weights = torch.rand(ii)
            self.random_dict_weights[ii] = random_weights

    @classproperty
    def h_dim_increase_type(cls):
        r"""
        Returns:
            None:
                The dimension of the output features do not change at each layer.
        """

        return None

    @classproperty
    def has_weights(cls):
        r"""
        Returns:
            False
                The current class does not use weights
        """
        return False

    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Add ``h`` with the previous layers with skip connection ``h_prev``,
        similar to ResNet.
        Parameters:
            h: torch.Tensor(..., m)
                The current layer features
            h_prev: torch.Tensor(..., m), None
                The features from the previous layer with a skip connection.
                At ``step_idx==0``, ``h_prev`` can be set to ``None``.
            step_idx: int
                Current layer index or step index in the forward loop of the architecture.
        Returns:
            h: torch.Tensor(..., m)
                Either return ``h`` unchanged, or the sum with
                on ``h_prev``, depending on the ``step_idx`` and ``self.skip_steps``.
            h_prev: torch.Tensor(..., m)
                Either return ``h_prev`` unchanged, or the same value as ``h``,
                depending on the ``step_idx`` and ``self.skip_steps``.
        """

        if self._bool_apply_skip_step(step_idx):
            for i in range(0, step_idx):
                h += (
                    self.random_dict_weights[step_idx][i].to(dtype=h_prev[i].dtype, device=h_prev[i].device)
                    * h_prev[i]
                )
            if h_prev is None:
                h_prev = [h]
            else:
                h_prev.append(h)

        return h, h_prev
