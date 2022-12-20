import torch
import torch.nn as nn
from typing import Union, Callable, List, Dict, Any

from goli.nn.base_layers import MLP


class MLPEncoder(torch.nn.Module):
    """Configurable kernel-based Positional Encoding node encoder.

    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.

    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(
        self,
        on_keys: List[str],
        out_level: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: Union[str, Callable] = "relu",
        dropout=0.0,
        normalization="none",
        first_normalization="none",
    ):
        super().__init__()

        # Check the out_level
        self.out_level = out_level.lower()
        accepted_out_levels = ["node", "edge"]
        if not (self.out_level in accepted_out_levels):
            raise ValueError(f"`out_level` must be in {accepted_out_levels}, provided {out_level}")

        self.on_keys = self.parse_on_keys(on_keys)
        self.out_level = out_level
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.first_normalization = first_normalization

        # Initialize the MLP
        self.pe_encoder = MLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            layers=num_layers,
            dropout=dropout,
            activation=activation,
            last_activation=activation,
            first_normalization=first_normalization,
            normalization=normalization,
            last_normalization=normalization,
            last_dropout=dropout,
        )

    def parse_on_keys(self, on_keys):
        # Parse the `on_keys`.
        if len(on_keys) != 1:
            raise ValueError(f"`{self.__class__}` only supports one key")
        # if list(on_keys.keys())[0] != "encoding":
        #     raise ValueError(f"`on_keys` must contain the key 'encoding'")
        return on_keys

    def forward(self, **encoding):
        encoding = encoding[self.on_keys[0]]
        # Run the MLP
        encoding = self.pe_encoder(encoding)  # (Num nodes) x dim_pe

        # Set the level (node vs edge)
        output = {self.out_level: encoding}

        return output

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension
        """
        return dict(
            on_keys=self.on_keys,
            out_level=self.out_level,
            in_dim=round(self.in_dim / divide_factor) if factor_in_dim else self.in_dim,
            hidden_dim=round(self.hidden_dim / divide_factor),
            out_dim=round(self.out_dim / divide_factor),
            num_layers=self.num_layers,
            activation=self.activation,
            dropout=self.dropout,
            normalization=self.normalization,
            first_normalization=self.first_normalization,
        )
