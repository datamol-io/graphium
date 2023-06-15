from typing import List, Dict, Any, Union, Callable
import abc
import torch
from torch_geometric.data import Batch


from graphium.nn.base_layers import get_norm
from graphium.nn.utils import MupMixin


class BaseEncoder(torch.nn.Module, MupMixin):
    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        in_dim: int,
        out_dim: int,
        num_layers: int,
        activation: Union[str, Callable] = "relu",
        first_normalization=None,
        use_input_keys_prefix: bool = True,  # TODO: might be redundant along with parse_input_keys_with_prefix function
    ):
        r"""
        Base class for all positional and structural encoders.
        Initialize the encoder with the following arguments:
        Parameters:
            input_keys: The keys from the graph to use as input
            output_keys: The keys to return as output encodings
            in_dim: The input dimension for the encoder
            out_dim: The output dimension of the encodings
            num_layers: The number of layers of the encoder
            activation: The activation function to use
            first_normalization: The normalization to use before the first layer
            use_input_keys_prefix: Whether to use the `key_prefix` argument in the `forward` method.
            This is useful when the encodings are categorized by the function `get_all_positional_encoding`
        """
        super().__init__()

        if type(in_dim) is list:
            in_dim = sum(in_dim)

        self.input_keys = self.parse_input_keys(input_keys)
        self.output_keys = self.parse_output_keys(output_keys)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation = activation
        self.use_input_keys_prefix = use_input_keys_prefix
        self.first_normalization = get_norm(first_normalization, dim=in_dim)

    # TODO: the function below seems redundant; could be removed/replaced moving forward
    def parse_input_keys_with_prefix(self, key_prefix):
        """
        Parse the `input_keys` argument, given a certain prefix.
        If the prefix is `None`, it is ignored
        """
        ### TODO: redundant
        input_keys = self.input_keys
        if (key_prefix is not None) and (self.use_input_keys_prefix):
            input_keys = [f"{k}" for k in input_keys]
            # input_keys = [f"{key_prefix}/{k}" for k in input_keys]
        ###
        return input_keys

    @abc.abstractmethod
    def forward(self, graph: Batch, key_prefix=None) -> Dict[str, torch.Tensor]:
        r"""
        Forward pass of the encoder on a graph.
        This is a method to be implemented by the child class.
        Parameters:
            graph: The input pyg Batch
        """
        raise ValueError("This method must be implemented by the child class")

    @abc.abstractmethod
    def parse_input_keys(self, input_keys: List[str]) -> List[str]:
        r"""
        Parse the `input_keys` argument. This is a method to be implemented by the child class.
        Parameters:
            input_keys: The input keys to parse
        """
        raise ValueError("This method must be implemented by the child class")

    @abc.abstractmethod
    def parse_output_keys(self, output_keys: List[str]) -> List[str]:
        """
        Parse the `output_keys` argument.  This is a method to be implemented by the child class.
        Parameters:
            output_keys: The output keys to parse
        """
        raise ValueError("This method must be implemented by the child class")

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameters:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension
        Returns:
            A dictionary with the base model arguments
        """

        base_kwargs = {
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "in_dim": round(self.in_dim / divide_factor) if factor_in_dim else self.in_dim,
            "out_dim": round(self.out_dim / divide_factor),
            "num_layers": self.num_layers,
            "activation": self.activation,
            "first_normalization": type(self.first_normalization).__name__,
            "use_input_keys_prefix": self.use_input_keys_prefix,
        }

        return base_kwargs
