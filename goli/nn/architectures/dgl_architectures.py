from torch import Tensor
from torch.nn import Module
import dgl
from typing import Tuple, Union, List, Optional

from goli.nn.base_graph_layer import BaseGraphModule

from goli.nn.dgl_layers import VirtualNodeDgl, parse_pooling_layer_dgl
from goli.nn.architectures.global_architectures import FeedForwardGraphBase


class FeedForwardDGL(FeedForwardGraphBase):
    r"""
    A flexible neural network architecture, with variable hidden dimensions,
    support for multiple layer types, and support for different residual
    connections.

    This class is meant to work with different DGL-based graph neural networks
    layers. Any layer must inherit from `goli.nn.base_graph_layer.BaseGraphStructure`
    or `goli.nn.base_graph_layer.BaseGraphLayer`.
    """

    def _graph_layer_forward(
        self,
        layer: BaseGraphModule,
        g: dgl.DGLGraph,
        h: Tensor,
        e: Optional[Tensor],
        h_prev: Optional[Tensor],
        e_prev: Optional[Tensor],
        step_idx: int,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        r"""
        Apply the *i-th* DGL graph layer, where *i* is the index given by `step_idx`.
        The layer is applied differently depending if there are edge features or not.

        Then, the residual is also applied on both the features and the edges (if applicable)

        Parameters:

            layer:
                The DGL layer used for the convolution

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
        # Apply the GNN layer with the right inputs/outputs
        if layer.layer_inputs_edges and layer.layer_outputs_edges:
            h, e = layer(g=g, h=h, e=e)
        elif layer.layer_inputs_edges:
            h = layer(g=g, h=h, e=e)
        elif layer.layer_outputs_edges:
            h, e = layer(g=g, h=h)
        else:
            h = layer(g=g, h=h)

        # Apply the residual layers on the features and edges (if applicable)
        if step_idx < len(self.layers) - 1:
            h, h_prev = self.residual_layer.forward(h, h_prev, step_idx=step_idx)
            if (self.residual_edges_layer is not None) and (layer.layer_outputs_edges):
                e, e_prev = self.residual_edges_layer.forward(e, e_prev, step_idx=step_idx)

        return h, e, h_prev, e_prev

    def _parse_virtual_node_class(self) -> type:
        return VirtualNodeDgl

    def _parse_pooling_layer(
        self, in_dim: int, pooling: Union[str, List[str]], **kwargs
    ) -> Tuple[Module, int]:
        return parse_pooling_layer_dgl(in_dim, pooling, **kwargs)

    def _get_node_feats(self, g: dgl.DGLGraph, key: str = "h") -> Tensor:
        """
        Get the node features of a DGL graph `g`.

        Parameters:
            g: graph
            key: key associated to the node features
        """
        return g.ndata.get(key, None)

    def _get_edge_feats(self, g: dgl.DGLGraph, key: str = "edge_attr") -> Tensor:
        """
        Get the edge features of a DGL graph `g`.

        Parameters:
            g: graph
            key: key associated to the edge features
        """
        return g.edata.get(key, None) if (self.in_dim_edges > 0) else None

    def _set_node_feats(self, g: dgl.DGLGraph, node_feats: Tensor, key: str = "h") -> dgl.DGLGraph:
        """
        Set the node features of a DGL graph `g`, and return the graph.

        Parameters:
            g: graph
            key: key associated to the node features
        """
        assert node_feats.shape[0] == g.num_nodes()
        g.ndata[key] = node_feats
        return g

    def _set_edge_feats(self, g: dgl.DGLGraph, edge_feats: Tensor, key: str = "edge_attr") -> dgl.DGLGraph:
        """
        Set the edge features of a DGL graph `g`, and return the graph.

        Parameters:
            g: graph
            key: key associated to the node features
        """
        if (self.in_dim_edges > 0) and (edge_feats is not None):
            assert edge_feats.shape[0] == g.num_edges()
            g.edata[key] = edge_feats
        return g
