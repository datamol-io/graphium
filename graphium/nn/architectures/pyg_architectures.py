from torch import Tensor
from torch.nn import Module
from typing import Tuple, Union, List, Optional

from torch_geometric.data import Data, Batch

from graphium.nn.base_graph_layer import BaseGraphModule
from graphium.nn.pyg_layers import VirtualNodePyg, parse_pooling_layer_pyg
from graphium.nn.architectures.global_architectures import FeedForwardGraph


class FeedForwardPyg(FeedForwardGraph):
    def _graph_layer_forward(
        self,
        layer: BaseGraphModule,
        g: Batch,
        feat: Tensor,
        edge_feat: Optional[Tensor],
        feat_prev: Optional[Tensor],
        edge_feat_prev: Optional[Tensor],
        step_idx: int,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        r"""
        A flexible neural network architecture, with variable hidden dimensions,
        support for multiple layer types, and support for different residual
        connections.

        This class is meant to work with different PyG-based graph neural networks
        layers. Any layer must inherit from `graphium.nn.base_graph_layer.BaseGraphStructure`
        or `graphium.nn.base_graph_layer.BaseGraphLayer`.

        Apply the *i-th* PyG graph layer, where *i* is the index given by `step_idx`.
        The layer is applied differently depending if there are edge features or not.

        Then, the residual is also applied on both the features and the edges (if applicable)

        Parameters:

            layer:
                The PyG layer used for the convolution

            g:
                graph on which the convolution is done

            feat (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the input features

            edge_feat (torch.Tensor[..., N, Ein]):
                Edge feature tensor, before convolution.
                `N` is the number of nodes, `Ein` is the input edge features

            feat_prev:
                Node feature of the previous residual connection, or `None`

            edge_feat_prev:
                Edge feature of the previous residual connection, or `None`

            step_idx:
                The current step idx in the forward loop

        Returns:

            feat (torch.Tensor[..., N, Dout]):
                Node feature tensor, after convolution and residual.
                `N` is the number of nodes, `Dout` is the output features of the layer and residual

            edge_feat (torch.Tensor[..., N, Eout]):
                Edge feature tensor, after convolution and residual.
                `N` is the number of nodes, `Ein` is the input edge features

            feat_prev:
                Node feature tensor to be used at the next residual connection, or `None`

            edge_feat_prev:
                Edge feature tensor to be used at the next residual connection, or `None`

        """

        # Set node / edge features into the graph
        g["feat"] = feat
        g["edge_feat"] = edge_feat

        # Apply the GNN layer
        g = layer(g)

        # Get the node / edge features from the graph
        feat = g["feat"]
        edge_feat = g["edge_feat"]

        # Apply the residual layers on the features and edges (if applicable)
        if step_idx < len(self.layers) - 1:
            feat, feat_prev = self.residual_layer.forward(feat, feat_prev, step_idx=step_idx)
            if (self.residual_edges_layer is not None) and (layer.layer_outputs_edges):
                edge_feat, edge_feat_prev = self.residual_edges_layer.forward(
                    edge_feat, edge_feat_prev, step_idx=step_idx
                )

        return feat, edge_feat, feat_prev, edge_feat_prev

    def _parse_virtual_node_class(self) -> type:
        return VirtualNodePyg

    def _parse_pooling_layer(
        self, in_dim: int, pooling: Union[str, List[str]], **kwargs
    ) -> Tuple[Module, int]:
        return parse_pooling_layer_pyg(in_dim, pooling, **kwargs)
