import torch
import torch.nn as nn
from torch import Tensor

from typing import Union, Callable, List, Dict, Any, Optional, Tuple
from torch_geometric.data import Batch
from torch_geometric.nn.models.dimenet import BesselBasisLayer, SphericalBasisLayer
from torch_geometric.nn import radius_graph

from graphium.nn.encoders.base_encoder import BaseEncoder
from graphium.nn.pyg_layers.utils import triplets
from graphium.nn.pyg_layers.dimenet_pyg import OutputBlock
from graphium.nn.base_layers import FCLayer


class BesselSphericalPosEncoder(BaseEncoder):
    def __init__(
        self,
        input_keys: List[str],  # The keys from the pyg graph
        output_keys: List[str],  # The keys to return
        in_dim: int,
        out_dim: int,
        num_layers: int,
        out_dim_edges: int,
        num_output_layers: int,
        num_spherical: int,
        num_radial: int,
        max_num_neighbors: int = 32,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        activation: Union[str, Callable] = "gelu",
        first_normalization="none",
        use_input_keys_prefix: bool = True,
    ):
        r"""
        Configurable DimeNet's embedding encoder from the
        `"Directional Message Passing for Molecular Graphs" <https://arxiv.org/abs/2003.03123> paper.

        [!] code uses the pytorch-geometric implementation of BesselBasisLayer & SphericalBasisLayer

        Parameters:
            input_keys: The keys from the graph to use as input
            output_keys: The keys to return as output encodings
                (**should at least contain: `edge_rbf`, `triplet_sbf`, `radius_edge_index`; Optional: `node_*`, `edge_*`)
            in_dim: The input dimension for the encoder (**not used)
            out_dim: The output dimension of the node encodings
            num_layers: The number of layers of the encoder (**not used)
            out_dim_edges: The output dimension of the edge encodings
            num_output_layers: The number of layers of the OutBlock
            num_spherical (int): Number of spherical harmonics.
            num_radial (int): Number of radial basis functions.
            max_num_neighbors (int): The maximum number of neighbors to consider in radius graph. (default: 32)
            cutoff (float): Cutoff distance for interatomic interactions. (default: 5.0)
            envelope_exponent (int): Shape of the smooth cutoff. (default: 5)
            activation: The activation function to use
            first_normalization: The normalization to use before the first layer
            use_input_keys_prefix: Whether to use the `key_prefix` argument in the `forward` method.

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
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope_exponent = envelope_exponent
        self.num_spherical = num_spherical
        self.max_num_neighbors = max_num_neighbors

        # static Bessel embeddings
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)  # for edges
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)  # for triplets

        # edge embedding (num_radial -> out_dim)
        # ***simplified version*** by removing atom type input
        self.rbf_proj = FCLayer(num_radial, out_dim_edges, activation=activation)
        # node embedding from edge embedding (out_dim_edges -> out_dim)
        self.output_block_0 = OutputBlock(
            num_radial, out_dim_edges, out_dim, num_output_layers, activation
        )  # 1 outblock right after embedding in original dimenet

    def forward(self, batch: Batch, key_prefix: Optional[str] = None) -> Dict[str, Any]:
        r"""
        forward function of the DimeNetEncoder class
        Parameters:
            batch: The batch of pyg graphs
            key_prefix: The prefix to use for the input keys
        Returns:
            A dictionary of the outpaut encodings with keys specified by `output_keys`
        """
        ### Get the input keys ###
        positions_3d_key = self.parse_input_keys_with_prefix(key_prefix)[0]
        # be in shape [num_nodes, 3]
        pos = batch[positions_3d_key]
        # Create radius graph in encoder (not use chemical topology of molecules)
        radius_edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch.batch, max_num_neighbors=self.max_num_neighbors
        )

        # Process edges and triplets.
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(radius_edge_index, num_nodes=pos.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        # [num_edges, num_radial]
        rbf = self.rbf(dist)
        # [num_triplets, num_spherical * num_radial]
        sbf = self.sbf(dist, angle, idx_kj)
        # initial 3D edge embedding [num_edges, out_dim] (may merge with other edge encoder's output)
        edge_feature_3d = self.rbf_proj(rbf)
        # initial 3D node embedding [num_nodes, out_dim] (align with original DimeNet implementation)
        P = self.output_block_0(edge_feature_3d, rbf, i, num_nodes=pos.size(0))

        # Crash if the key starts with 'graph_'
        # Return `rbf` and `sbf` for necessary message passing in DimeNet
        # Return `radius_edge_index` for necessary message passing in DimeNet
        # Return `P` as node embedding (if the key starts with 'node_')
        # Return `edge_feature_3d` otherwise
        output = {}
        for key in self.output_keys:
            if isinstance(key, str) and key.startswith("graph_"):
                raise ValueError("Graph encodings are not supported for this encoder")
            elif key.startswith("node_"):
                output[key] = P
            elif key == "edge_rbf":
                output[key] = rbf
            elif key == "triplet_sbf":
                output[key] = sbf
            elif key == "radius_edge_index":
                output[key] = radius_edge_index
            else:
                output[key] = edge_feature_3d
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
                num_spherical=self.num_spherical,
                num_radial=round(self.num_radial / divide_factor),
                cutoff=self.cutoff,
                envelope_exponent=self.envelope_exponent,
            )
        )
        return base_kwargs

    # these two below aligned with `gaussian_kernel_pos_encoder``
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
        # all of three are required to do DimeNet-style message passing
        assert "edge_rbf" in output_keys, "Edge radial basis feature should present for this encoder"
        assert (
            "triplet_sbf" in output_keys
        ), "Triplet(angle) spherical radial basis feature should present for this encoder"
        assert (
            "radius_edge_index" in output_keys
        ), "Radius edge index (graph) should be built in forward of this encoder"

        for key in output_keys:
            assert not key.startswith("graph_"), "Graph encodings are not supported for this encoder"
        return output_keys
