from torch import Tensor
from torch.nn import Module
from typing import Tuple, Union, List, Optional

from torch_geometric.data import Data, Batch

from goli.nn.base_graph_layer import BaseGraphModule
from goli.nn.pyg_layers import VirtualNodePyg, parse_pooling_layer_pyg
from goli.nn.architectures.global_architectures import FeedForwardGraph


class FeedForwardPyg(FeedForwardGraph):
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
        A flexible neural network architecture, with variable hidden dimensions,
        support for multiple layer types, and support for different residual
        connections.

        This class is meant to work with different PyG-based graph neural networks
        layers. Any layer must inherit from `goli.nn.base_graph_layer.BaseGraphStructure`
        or `goli.nn.base_graph_layer.BaseGraphLayer`.

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
        g["h"] = h
        g["edge_attr"] = e

        # Apply the GNN layer
        g = layer(g)

        # Get the node / edge features from the graph
        h = g["h"]
        e = g["edge_attr"]

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
