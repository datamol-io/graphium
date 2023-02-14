import torch
import torch.nn as nn
from typing import Union, Callable, List, Dict, Any

from goli.nn.base_layers import MLP


class GaussianKernelPosEncoder(torch.nn.Module):
    """Configurable gaussian kernel-based Positional Encoding node and edge encoder.

    Useful for encoding 3D conformation positions.

    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(
        self,
        on_keys: List[str], # The keys from the pyg graph (I think?) #! Andy change for pyg_keys? For all encoders
        out_level: List[str], # Whether to return on the nodes, edges, or both
        in_dim: int,
        num_kernel: int, # replaces hidden_dim, Number of gaussian kernel used.
        out_dim: int,
        num_layers: int,
        num_heads: int,
        activation: Union[str, Callable] = "gelu",
        dropout=0.0,
        normalization="none",
        first_normalization="none",
    ):

        super().__init__()

        # Check the out_level
        self.out_level = out_level.lower()
        accepted_out_levels = ["node", "edge"]
        if not (self.out_level in accepted_out_levels): #! Change to accept both nodes and edges
            raise ValueError(f"`out_level` must be in {accepted_out_levels}, provided {out_level}")

        self.on_keys = self.parse_on_keys(on_keys)
        self.out_level = out_level
        self.in_dim = in_dim
        self.num_kernel = num_kernel
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.first_normalization = first_normalization

        #! Andy, please don't forget the `Preprocess3DPositions`

        self.gaussian_proj = MLP(
            in_dim=num_kernel,
            hidden_dim=num_kernel,
            out_dim=num_heads,
            layers=num_layers,
            activation=activation,
        )
        self.node_proj = nn.Linear(num_kernel, out_dim)

        self.num_kernel = num_kernel
        self.means = nn.Embedding(1, num_kernel)
        self.stds = nn.Embedding(1, num_kernel)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, **encoding) -> Dict[torch.Tensor]:

        encoding = encoding[self.on_keys[0]] #! Andy change how to use the on_keys
        self.gaussian_proj #! Andy use this
        self.node_proj #! Andy use this

        # [batch, nodes, nodes, 1]
        input = input.unsqueeze(-1)
        # [batch, nodes, nodes, num_kernels]
        expanded_input = input.expand(-1, -1, -1, self.num_kernels)
        # [num_kernels]
        mean = self.means.weight.float().view(-1)
        # [num_kernels]
        std = self.stds.weight.float().view(-1).abs() + 0.01  # epsilon is 0.01 that matches gps++ value
        pi = 3.141592653
        pre_exp_factor = (2 * pi) ** 0.5
        # [batch, nodes, nodes, num_kernels]
        tensor_with_kernel = torch.exp(-0.5 * (((expanded_input - mean) / std) ** 2)) / (pre_exp_factor * std)

        # Set the level (node vs edge)
        output = {self.out_level: tensor_with_kernel} #! Andy, change the out_level to use both node and edge

        return output

    def parse_on_keys(self, on_keys):
        # Parse the `on_keys`.
        if len(on_keys) != 1:
            raise ValueError(f"`{self.__class__}` only supports one key")
        return on_keys


    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension
        """

        #! Andy, we need to check that all hidden dimensions are scaled by the needed factor.
        #! Do we need to do it for the number of kernels.
        return dict(
            on_keys=self.on_keys,
            out_level=self.out_level,
            in_dim=round(self.in_dim / divide_factor) if factor_in_dim else self.in_dim,
            # hidden_dim=round(self.hidden_dim / divide_factor),
            out_dim=round(self.out_dim / divide_factor),
            num_layers=self.num_layers,
            activation=self.activation,
            dropout=self.dropout,
            normalization=self.normalization,
            first_normalization=self.first_normalization,
        )
