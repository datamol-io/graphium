import torch
import torch.nn as nn
import abc
from typing import Union, Callable

from goli.nn.base_layers import get_activation
from goli.utils.decorators import classproperty


class BaseGraphStructure():
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        **kwargs,
    ):
        r"""
        Abstract class used to standardize the implementation of DGL layers
        in the current library. It will allow a network to seemlesly swap between
        different GNN layers by better understanding the expected inputs
        and outputs.

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            activation:
                activation function to use in the layer

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function
        """

        super().__init__()

        # Basic attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalization = normalization
        self.dropout = dropout
        self.activation = activation

    def _initialize_activation_dropout_norm(self):

        if not isinstance(self, nn.Module):
            raise TypeError("This function requires the current object to be an `nn.Module`. Use multi-inheritance or the class `BaseGraphModule` instead")

        # Build the layers
        self.activation_layer = get_activation(self.activation)

        self.dropout_layer = None
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        self.norm_layer = self._parse_norm(self.normalization)


    def _parse_norm(self, normalization):

        if normalization is None or normalization == "none":
            parsed_norm = None
        elif callable(normalization):
            parsed_norm = normalization
        elif normalization == "batch_norm":
            parsed_norm = nn.BatchNorm1d(self.out_dim * self.out_dim_factor)
        elif normalization == "layer_norm":
            parsed_norm = nn.LayerNorm(self.out_dim * self.out_dim_factor)
        else:
            raise ValueError(
                f"Undefined normalization `{normalization}`, must be `None`, `Callable`, 'batch_norm', 'layer_norm', 'none'"
            )
        return parsed_norm

    def apply_norm_activation_dropout(
        self,
        h: torch.Tensor,
        normalization: bool = True,
        activation: bool = True,
        dropout: bool = True,
    ):
        r"""
        Apply the different normalization and the dropout to the
        output layer.

        Parameters:

            h:
                Feature tensor, to be normalized

            normalization:
                Whether to apply the normalization

            activation:
                Whether to apply the activation layer

            dropout:
                Whether to apply the dropout layer

        Returns:

            h:
                Normalized and dropped-out features

        """

        if normalization and (self.norm_layer is not None):
            h = self.norm_layer(h)

        if activation and (self.activation_layer is not None):
            h = self.activation_layer(h)

        if dropout and (self.dropout_layer is not None):
            h = self.dropout_layer(h)

        return h

    @classproperty
    def layer_supports_edges(cls) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        supports output edges edges or not.

        Returns:

            bool:
                Whether the layer supports the use of edges
        """
        ...

    @property
    @abc.abstractmethod
    def layer_inputs_edges(self) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_input_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Whether the layer uses input edges in the forward pass
        """
        ...

    @property
    @abc.abstractmethod
    def layer_outputs_edges(self) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_output_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Whether the layer outputs edges in the forward pass
        """
        ...

    @property
    @abc.abstractmethod
    def out_dim_factor(self) -> int:
        r"""
        Abstract method.
        Get the factor by which the output dimension is multiplied for
        the next layer.

        For standard layers, this will return ``1``.

        But for others, such as ``GatLayer``, the output is the concatenation
        of the outputs from each head, so the out_dim gets multiplied by
        the number of heads, and this function should return the number
        of heads.

        Returns:

            int:
                The factor that multiplies the output dimensions
        """
        ...

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        f = self.out_dim_factor
        out_dim_f_print = "" if f == 1 else f" * {f}"
        return f"{self.__class__.__name__}({self.in_dim} -> {self.out_dim}{out_dim_f_print}, activation={self.activation})"



class BaseGraphModule(BaseGraphStructure, nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
    ):
        r"""
        Abstract class used to standardize the implementation of DGL layers
        in the current library. It will allow a network to seemlesly swap between
        different GNN layers by better understanding the expected inputs
        and outputs.

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            activation:
                activation function to use in the layer

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function
        """

        super(BaseGraphModule, self).__init__(
            in_dim = in_dim,
            out_dim = out_dim,
            normalization = normalization,
            dropout = dropout,
            activation = activation,
            )

        self._initialize_activation_dropout_norm()
