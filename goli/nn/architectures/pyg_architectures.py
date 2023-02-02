from torch import Tensor
from torch.nn import Module
from typing import Tuple, Union, List, Optional

from torch_geometric.data import Data, Batch

from goli.nn.base_graph_layer import BaseGraphModule
from goli.nn.pyg_layers import VirtualNodePyg, parse_pooling_layer_pyg
from goli.nn.architectures.global_architectures import FeedForwardGraphBase


class FeedForwardPyg(FeedForwardGraphBase):
    r"""
    A flexible neural network architecture, with variable hidden dimensions,
    support for multiple layer types, and support for different residual
    connections.

    This class is meant to work with different PyG-based graph neural networks
    layers. Any layer must inherit from `goli.nn.base_graph_layer.BaseGraphStructure`
    or `goli.nn.base_graph_layer.BaseGraphLayer`.
    """

    def _graph_layer_forward(
        self,
        layer: BaseGraphModule,
        g: Batch,
        h: Tensor,
        e: Optional[Tensor],
        h_prev: Optional[Tensor],
        e_prev: Optional[Tensor],
        step_idx: int,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        r"""
        Apply the *i-th* PyG graph layer, where *i* is the index given by `step_idx`.
        The layer is applied differently depending if there are edge features or not.

        Then, the residual is also applied on both the features and the edges (if applicable)

        Parameters:

            layer:
                The PyG layer used for the convolution

            g:
                graph on which the convolution is done

            h (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the input features

            e (torch.Tensor[..., N, Ein]):
                Edge feature tensor, before convolution.
                `N` is the number of nodes, `Ein` is the input edge features

            h_prev:
                Node feature of the previous residual connection, or `None`

            e_prev:
                Edge feature of the previous residual connection, or `None`

            step_idx:
                The current step idx in the forward loop

        Returns:

            h (torch.Tensor[..., N, Dout]):
                Node feature tensor, after convolution and residual.
                `N` is the number of nodes, `Dout` is the output features of the layer and residual

            e:
                Edge feature tensor, after convolution and residual.
                `N` is the number of nodes, `Ein` is the input edge features

            h_prev:
                Node feature tensor to be used at the next residual connection, or `None`

            e_prev:
                Edge feature tensor to be used at the next residual connection, or `None`

        """

        # Set node / edge features into the graph
        g = self._set_node_feats(g, h, key="h")
        g = self._set_edge_feats(g, e, key="edge_attr")

        # Apply the GNN layer
        g = layer(g)

        # Get the node / edge features from the graph
        h = self._get_node_feats(g, key="h")
        e = self._get_edge_feats(g, key="edge_attr")

        # Apply the residual layers on the features and edges (if applicable)
        if step_idx < len(self.layers) - 1:
            h, h_prev = self.residual_layer.forward(h, h_prev, step_idx=step_idx)
            if (self.residual_edges_layer is not None) and (layer.layer_outputs_edges):
                e, e_prev = self.residual_edges_layer.forward(e, e_prev, step_idx=step_idx)

        return h, e, h_prev, e_prev

    def _parse_virtual_node_class(self) -> type:
        return VirtualNodePyg

    def _parse_pooling_layer(
        self, in_dim: int, pooling: Union[str, List[str]], **kwargs
    ) -> Tuple[Module, int]:
        return parse_pooling_layer_pyg(in_dim, pooling, **kwargs)

    def _get_node_feats(self, g: Union[Data, Batch], key: str = "h") -> Tensor:
        """
        Get the node features of a PyG graph `g`.

        Parameters:
            g: graph
            key: key associated to the node features
        """
        return g.get(key, None)

    def _get_edge_feats(self, g: Union[Data, Batch], key: str = "edge_attr") -> Tensor:
        """
        Get the edge features of a PyG graph `g`.

        Parameters:
            g: graph
            key: key associated to the edge features
        """
        return g.get(key, None) if (self.in_dim_edges > 0) else None

    def _set_node_feats(
        self, g: Union[Data, Batch], node_feats: Tensor, key: str = "h"
    ) -> Union[Data, Batch]:
        """
        Set the node features of a PyG graph `g`, and return the graph.

        Parameters:
            g: graph
            key: key associated to the node features
        """
        assert node_feats.shape[0] == g.num_nodes
        g[key] = node_feats
        return g

    def _set_edge_feats(
        self, g: Union[Data, Batch], edge_feats: Tensor, key: str = "edge_attr"
    ) -> Union[Data, Batch]:
        """
        Set the edge features of a PyG graph `g`, and return the graph.

        Parameters:
            g: graph
            key: key associated to the node features
        """
        if (self.in_dim_edges > 0) and (edge_feats is not None):
            assert edge_feats.shape[0] == g.num_edges
            g[key] = edge_feats
        return g
