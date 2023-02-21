import torch
from typing import Union, Callable, List, Dict, Any, Optional
from torch_geometric.data import Batch

from goli.nn.pyg_layers.utils import PreprocessPositions
from goli.ipu.to_dense_batch import to_dense_batch, to_sparse_batch
from goli.ipu.ipu_utils import import_poptorch
from goli.nn.encoders.base_encoder import BaseEncoder


class GaussianKernelPosEncoder(BaseEncoder):
    """Configurable gaussian kernel-based Positional Encoding node and edge encoder.

    Useful for encoding 3D conformation positions.

    """

    def __init__(
        self,
        input_keys: List[str],  # The keys from the pyg graph
        output_keys: List[str],  # The keys to return
        in_dim: int,
        out_dim: int,
        num_layers: int,
        max_num_nodes_per_graph: int,
        activation: Union[str, Callable] = "gelu",
        first_normalization="none",
        use_input_keys_prefix: bool = True,
        num_heads: int = 1,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            in_dim=in_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            activation=activation,
            first_normalization=first_normalization,
            use_input_keys_prefix=use_input_keys_prefix,
        )

        self.num_heads = num_heads
        self.max_num_nodes_per_graph = max_num_nodes_per_graph

        # parameters for preprocessing 3d positions
        self.preprocess_3d_positions = PreprocessPositions(
            num_heads=self.num_heads,
            embed_dim=self.out_dim,
            num_kernel=self.out_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            first_normalization=self.first_normalization,
        )

    def parse_input_keys(self, input_keys):
        # Parse the `input_keys`.
        if len(input_keys) != 1:
            raise ValueError(f"`{self.__class__}` only supports one key")
        for key in input_keys:
            assert not key.startswith("edge_"), f"Input keys must be node features, not edge features, for encoder {self.__class__}"
            assert not key.startswith("graph_"), f"Input keys must be node features, not graph features, for encoder {self.__class__}"
        return input_keys

    def parse_output_keys(self, output_keys):
        for key in output_keys:
            assert not key.startswith("edge_"), "Edge encodings are not supported for this encoder"
        return output_keys

    def forward(self, batch: Batch, key_prefix: Optional[str] = None) -> Dict[str, Any]:
        input_keys = self.parse_input_keys_with_prefix(key_prefix)

        poptorch = import_poptorch(raise_error=False) # TODO: Change to `is_running_on_ipu` after merge
        on_ipu = (poptorch is not None) and (poptorch.isRunningOnIpu())
        max_num_nodes_per_graph = None
        if on_ipu:
            max_num_nodes_per_graph = self.max_num_nodes_per_graph

        attn_bias_3d, node_feature_3d = self.preprocess_3d_positions(batch, max_num_nodes_per_graph, on_ipu, positions_3d_key=input_keys[0])

        # Return the output features for both the nodes and the edges
        output = {}
        for key in self.output_keys:
            if isinstance(key, str) and key.startswith("graph_"):
                output[key] = attn_bias_3d
            else:
                output[key] = node_feature_3d
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
        base_kwargs = super().make_mup_base_kwargs(divide_factor=divide_factor, factor_in_dim=factor_in_dim)
        base_kwargs.update(dict(
            num_heads=self.num_heads,
            max_num_nodes_per_graph=self.max_num_nodes_per_graph,
        ))
        return base_kwargs
