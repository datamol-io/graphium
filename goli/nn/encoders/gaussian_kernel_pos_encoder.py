import torch
import torch.nn as nn
from typing import Union, Callable, List, Dict, Any
from torch_geometric.data import Batch

from goli.nn.base_layers import MLP
from goli.nn.pyg_layers.utils import GaussianLayer, Preprocess3DPositions
from goli.ipu.to_dense_batch import to_dense_batch, to_sparse_batch
from goli.ipu.ipu_utils import import_poptorch


class GaussianKernelPosEncoder(torch.nn.Module):
    """Configurable gaussian kernel-based Positional Encoding node and edge encoder.

    Useful for encoding 3D conformation positions.

    """

    def __init__(
        self,
        on_keys: List[
            str
        ],  # The keys from the pyg graph (I think?) #! Andy change for pyg_keys? For all encoders
        out_level: List[str],  # Whether to return on the nodes, edges, or both
        in_dim: int,
        out_dim: int,
        num_kernel: int,  # replaces hidden_dim, Number of gaussian kernel used.
        num_layers: int,
        num_heads: int,
        max_num_nodes_per_graph: int,
        activation: Union[str, Callable] = "gelu",
        dropout=0.0,
        normalization="none",
        first_normalization="none",
    ):
        super().__init__()

        # Check the out_level
        self.out_level = out_level.lower()
        accepted_out_levels = ["node", "edge"]
        if not (self.out_level in accepted_out_levels):  #! Change to accept both nodes and edges
            raise ValueError(f"`out_level` must be in {accepted_out_levels}, provided {out_level}")

        self.on_keys = self.parse_on_keys(on_keys)
        self.out_level = out_level
        self.in_dim = in_dim
        self.num_kernel = num_kernel
        self.out_dim = num_kernel
        self.num_heads = num_heads
        self.max_num_nodes_per_graph = max_num_nodes_per_graph
        self.embed_dim = num_kernel  # set to be number of kernels
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.first_normalization = first_normalization

        # * parameters for preprocessing 3d positions
        self.preprocess_3d_positions = Preprocess3DPositions(
            self.num_heads,
            self.embed_dim,
            self.num_kernel,
        )

    def parse_on_keys(self, on_keys):
        # Parse the `on_keys`.
        if len(on_keys) != 1:
            raise ValueError(f"`{self.__class__}` only supports one key")
        return on_keys

    def forward(self, batch: Batch) -> Dict[str, Any]:
        poptorch = import_poptorch(raise_error=False)
        on_ipu = (poptorch is not None) and (poptorch.isRunningOnIpu())
        max_num_nodes_per_graph = None
        if on_ipu:
            max_num_nodes_per_graph = self.max_num_nodes_per_graph

        attn_bias_3d, node_feature_3d = self.preprocess_3d_positions(batch, max_num_nodes_per_graph, on_ipu)

        output = {self.out_level: node_feature_3d}  #! Andy, change the out_level to use both node and edge
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

        #! Andy, we need to check that all hidden dimensions are scaled by the needed factor.
        #! Do we need to do it for the number of kernels.
        return dict(
            on_keys=self.on_keys,
            out_level=self.out_level,
            in_dim=round(self.in_dim / divide_factor) if factor_in_dim else self.in_dim,
            num_kernel=self.num_kernel,
            out_dim=round(self.out_dim / divide_factor),
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_num_nodes_per_graph=self.max_num_nodes_per_graph,
            activation=self.activation,
            dropout=self.dropout,
            normalization=self.normalization,
            first_normalization=self.first_normalization,
        )
