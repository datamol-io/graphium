from typing import Union, Callable, List, Dict, Any, Optional
from torch_geometric.data import Batch

from graphium.nn.pyg_layers.utils import PreprocessPositions
from graphium.ipu.ipu_utils import is_running_on_ipu
from graphium.nn.encoders.base_encoder import BaseEncoder


class GaussianKernelPosEncoder(BaseEncoder):
    def __init__(
        self,
        input_keys: List[str],  # The keys from the pyg graph
        output_keys: List[str],  # The keys to return
        in_dim: int,
        out_dim: int,
        embed_dim: int,
        num_layers: int,
        max_num_nodes_per_graph: Optional[int] = None,
        activation: Union[str, Callable] = "gelu",
        first_normalization="none",
        use_input_keys_prefix: bool = True,
        num_heads: int = 1,
    ):
        r"""
        Configurable gaussian kernel-based Positional Encoding node and edge encoder.
        Useful for encoding 3D conformation positions.

        Parameters:
            input_keys: The keys from the pyg graph to use as input
            output_keys: The keys to return corresponding to the output encodings
            in_dim: The input dimension for the encoder
            out_dim: The output dimension of the encodings
            embed_dim: The dimension of the embedding
            num_layers: The number of layers of the encoder
            max_num_nodes_per_graph: The maximum number of nodes per graph
            activation: The activation function to use
            first_normalization: The normalization to use before the first layer
            use_input_keys_prefix: Whether to use the `key_prefix` argument in the `forward` method.
            num_heads: The number of heads to use for the multi-head attention
        """
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

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_num_nodes_per_graph = max_num_nodes_per_graph

        # parameters for preprocessing 3d positions
        self.preprocess_3d_positions = PreprocessPositions(
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            num_kernel=self.out_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            first_normalization=self.first_normalization,
        )

    def parse_input_keys(
        self,
        input_keys: List[str],
    ) -> List[str]:
        r"""
        Parse the `input_keys`.
        Parameters:
            input_keys: The input keys to parse
        Returns:
            The parsed input keys
        """

        if len(input_keys) != 1:
            raise ValueError(f"`{self.__class__}` only supports one key")
        for key in input_keys:
            assert not key.startswith(
                "edge_"
            ), f"Input keys must be node features, not edge features, for encoder {self.__class__}"
            assert not key.startswith(
                "nodepair_"
            ), f"Input keys must be node features, not nodepair features, for encoder {self.__class__}"
            assert not key.startswith(
                "graph_"
            ), f"Input keys must be node features, not graph features, for encoder {self.__class__}"
        return input_keys

    def parse_output_keys(
        self,
        output_keys: List[str],
    ) -> List[str]:
        r"""
        Parse the `output_keys`.
        Parameters:
            output_keys: The output keys to parse
        Returns:
            The parsed output keys
        """
        for key in output_keys:
            assert not key.startswith("edge_"), "Edge encodings are not supported for this encoder"
            assert not key.startswith("graph_"), "Graph encodings are not supported for this encoder"
        return output_keys

    def forward(self, batch: Batch, key_prefix: Optional[str] = None) -> Dict[str, Any]:
        r"""
        forward function of the GaussianKernelPosEncoder class
        Parameters:
            batch: The batch of pyg graphs
            key_prefix: The prefix to use for the input keys
        Returns:
            A dictionary of the output encodings with keys specified by `output_keys`
        """
        input_keys = self.parse_input_keys_with_prefix(key_prefix)

        on_ipu = is_running_on_ipu()
        max_num_nodes_per_graph = None
        if on_ipu:
            max_num_nodes_per_graph = self.max_num_nodes_per_graph

        attn_bias_3d, node_feature_3d = self.preprocess_3d_positions(
            batch, max_num_nodes_per_graph, on_ipu, positions_3d_key=input_keys[0]
        )

        # Return `attn_bias_3d` if the key starts with 'nodepair_'
        # Crash if the key starts with 'edge_' or 'graph_'
        # Return `node_feature_3d` otherwise
        output = {}
        for key in self.output_keys:
            if isinstance(key, str) and key.startswith("nodepair_"):
                output[key] = attn_bias_3d
            elif isinstance(key, str) and key.startswith("edge_"):
                raise ValueError("Edge encodings are not supported for this encoder")
            else:
                output[key] = node_feature_3d
        return output

    def make_mup_base_kwargs(
        self,
        divide_factor: float = 2.0,
        factor_in_dim: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension
        Returns:
            A dictionary of the base model kwargs
        """
        base_kwargs = super().make_mup_base_kwargs(divide_factor=divide_factor, factor_in_dim=factor_in_dim)
        base_kwargs.update(
            dict(
                num_heads=self.num_heads,
                embed_dim=round(self.embed_dim / divide_factor),
                max_num_nodes_per_graph=self.max_num_nodes_per_graph,
            )
        )
        return base_kwargs
