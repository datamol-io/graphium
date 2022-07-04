import torch
import dgl
from typing import Tuple, Union, List

from goli.nn.base_graph_layer import BaseGraphModule

from goli.nn.pyg_layers import VirtualNodePyg, parse_pooling_layer_pyg
from goli.nn.architectures.global_architectures import FeedForwardGraphBase

class FeedForwardPyg(FeedForwardGraphBase):
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
        h: torch.Tensor,
        e: Union[torch.Tensor, None],
        h_prev: Union[torch.Tensor, None],
        e_prev: Union[torch.Tensor, None],
        step_idx: int
        ) -> Tuple[
            torch.Tensor,
            Union[torch.Tensor, None],
            Union[torch.Tensor, None],
            Union[torch.Tensor, None]]:
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

        g = layer(g)
        h = g.x
        e = g.edge_attr

        # Apply the residual layers on the features and edges (if applicable)
        if step_idx < len(self.layers) - 1:
            h, h_prev = self.residual_layer.forward(h, h_prev, step_idx=step_idx)
            if (self.residual_edges_layer is not None) and (layer.layer_outputs_edges):
                e, e_prev = self.residual_edges_layer.forward(e, e_prev, step_idx=step_idx)
        g.x = h
        g.edge_attr = e

        return h, e, h_prev, e_prev

    def _parse_virtual_node_class(self) -> type:
        return VirtualNodePyg

    def _parse_pooling_layer(self, in_dim: int, pooling: Union[str, List[str]], **kwargs) -> Tuple[torch.nn.Module, int]:
        return parse_pooling_layer_pyg(in_dim, pooling, **kwargs)


    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        r"""
        Apply the full graph neural network on the input graph and node features.

        Parameters:

            g:
                graph on which the convolution is done.
                Must contain the following elements:

                - `g.ndata["h"]`: `torch.Tensor[..., N, Din]`.
                  Input node feature tensor, before the network.
                  `N` is the number of nodes, `Din` is the input features

                - `g.edata["e"]`: `torch.Tensor[..., N, Ein]` **Optional**.
                  The edge features to use. It will be ignored if the
                  model doesn't supporte edge features or if
                  `self.in_dim_edges==0`.

        Returns:

            `torch.Tensor[..., M, Dout]` or `torch.Tensor[..., N, Dout]`:
                Node or graph feature tensor, after the network.
                `N` is the number of nodes, `M` is the number of graphs,
                `Dout` is the output dimension ``self.out_dim``
                If the `self.pooling` is [`None`], then it returns node features and the output dimension is `N`,
                otherwise it returns graph features and the output dimension is `M`

        """

        # Get node and edge features
        h = g.x
        e = g.edge_attr if (self.in_dim_edges > 0) else None

        pooled_h = super().forward(g, h, e)

        return pooled_h
