import torch.nn as nn
import abc
from typing import List, Dict, Tuple


class BaseDGLLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation, dropout: float, batch_norm: bool):
        super().__init__()

        # Initializing attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

    @staticmethod
    @abc.abstractmethod
    def layer_supports_edges() -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        supports edges or not.

        Returns
        ---------

        supports_edges: bool
            Whether the layer supports the use of edges
        """
        ...

    @abc.abstractmethod
    def layer_uses_edges(self) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        uses edges or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns
        ---------

        uses_edges: bool
            Whether the layer uses edges
        """
        ...

    @abc.abstractmethod
    def get_out_dim_factor(self) -> int:
        r"""
        Get the factor by which the output dimension is multiplied for
        the next layer.

        For standard layers, this will return ``1``.

        But for others, such as ``GatLayer``, the output is the concatenation
        of the outputs from each head, so the out_dim gets multiplied by
        the number of heads, and this function should return the number
        of heads.

        Returns
        ---------

        dim_factor: int
            The factor that multiplies the dimensions
        """
        ...

    @abc.abstractmethod
    def get_true_out_dims(self, out_dims: List[int]) -> List[int]:
        r"""
        Take a list of output dimensions, and return the same list, but
        multiplied by the value returned by ``self.get_out_dim_factor()``

        Parameters
        -------------

        out_dims: list(int)
            The output dimensions desired for the model

        Returns
        ------------

        true_out_dims: list(int)
            The true output dimensions returned by the model

        """
        ...

    @abc.abstractmethod
    def get_layer_wise_kwargs(self, num_layers, **kwargs) -> Tuple(Dict[List], List[str]):
        r"""
        Abstract method that transforms some set of arguments into a list of
        arguments, such that they can have different values for each layer.

        Parameters
        -------------

        num_layers: int
            The number of layers in the global model

        kwargs:
            The set of key-word arguments, containing at least the key that
            we want to convert to a layer-wise argument.

        Returns
        ------------

        layer_wise_kwargs: Dict(List)
            The set of key-word arguments, with each key associated to a list
            of the same size as ``num_layers``.

        kwargs_keys_to_remove: List(str)
            Key-word arguments to remove from the initializatio of the layer

        """
        ...

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, activation={self.activation})"
