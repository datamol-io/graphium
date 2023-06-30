import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from typing import List, Union, Callable, Tuple, Optional, Dict
from copy import deepcopy

from torch_scatter import scatter
from torch_geometric.data import Data, Batch

from graphium.nn.base_layers import MLP, FCLayer
from graphium.utils.tensor import ModuleListConcat, ModuleWrap
from graphium.nn.base_layers import MuReadoutGraphium

EPS = 1e-6


def scatter_logsum_pool(x: Tensor, batch: LongTensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
    r"""
    Apply pooling over the nodes in the graph using a mean aggregation,
    but scaled by the log of the number of nodes. This gives the same
    expressive power as the sum, but helps deal with graphs that are
    significantly larger than others by using a logarithmic scale.

    $$r^{(i)} = \frac{\log N_i}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k$$

    Parameters:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    Returns:
        the pooled features tensor
    """
    dim_size = int(batch.max().detach() + 1) if dim_size is None else dim_size
    mean_pool = scatter(x, batch, dim=dim, dim_size=dim_size, reduce="mean")
    num_nodes = scatter(
        torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device),
        batch,
        dim=dim,
        dim_size=dim_size,
        reduce="sum",
    )
    lognum = torch.log(num_nodes)
    return mean_pool * lognum.unsqueeze(-1)


def scatter_std_pool(x: Tensor, batch: LongTensor, dim: int = 0, dim_size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    minimum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

    Parameters:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    Returns:
        the pooled features tensor
    """
    dim_size = int(batch.max().detach() + 1) if dim_size is None else dim_size
    mean = scatter(x, batch, dim=dim, out=None, dim_size=dim_size, reduce="mean")
    mean_squares = scatter(x * x, batch, dim=dim, out=None, dim_size=dim_size, reduce="mean")
    out = mean_squares - mean * mean
    return torch.sqrt(torch.relu(out) + 1e-5)


class PoolingWrapperPyg(ModuleWrap):
    def __init__(self, func, feat_type, *args, **kwargs) -> None:
        super().__init__(func, *args, **kwargs)
        self.feat_type = feat_type

    def forward(self, g: Batch, feature: Tensor, *args, **kwargs):
        """
        forward function
        Parameters:
            g: the pyg batch graph
            feature: the node features
        Returns:
            the pooled features
        """
        dim_size = g.num_graphs
        if self.feat_type == "node":
            index = g.batch
        elif self.feat_type == "edge":
            index = g.batch[g.edge_index][0]
        else:
            index = g.batch
        return self.func(feature, index, dim_size=dim_size, *args, **kwargs, **self.kwargs)


def parse_pooling_layer_pyg(in_dim: int, pooling: Union[str, List[str]], feat_type: str = "node", **kwargs):
    r"""
    Select the pooling layers from a list of strings, and put them
    in a Module that concatenates their outputs.

    Parameters:

        in_dim:
            The dimension at the input layer of the pooling

        pooling:
            The list of pooling layers to use. The accepted strings are:

            - "none": No pooling
            - "sum": Sum all the nodes for each graph
            - "mean": Mean all the nodes for each graph
            - "logsum": Mean all the nodes then multiply by log(num_nodes) for each graph
            - "max": Max all the nodes for each graph
            - "min": Min all the nodes for each graph
            - "std": Standard deviation of all the nodes for each graph

    """

    # Create the pooling layer
    pool_layer = ModuleListConcat()
    out_pool_dim = 0
    if isinstance(pooling, str):
        pooling = [pooling]
    assert feat_type in ["node", "edge", "global"]
    for this_pool in pooling:
        this_pool = None if this_pool is None else this_pool.lower()
        out_pool_dim += in_dim
        if this_pool == "sum":
            pool_layer.append(PoolingWrapperPyg(scatter, dim=0, reduce="add", feat_type=feat_type, **kwargs))
        elif this_pool == "mean":
            pool_layer.append(PoolingWrapperPyg(scatter, dim=0, reduce="mean", feat_type=feat_type, **kwargs))
        elif this_pool == "logsum":
            pool_layer.append(PoolingWrapperPyg(scatter_logsum_pool, dim=0, feat_type=feat_type, **kwargs))
        elif this_pool == "max":
            pool_layer.append(PoolingWrapperPyg(scatter, dim=0, reduce="max", feat_type=feat_type, **kwargs))
        elif this_pool == "min":
            pool_layer.append(PoolingWrapperPyg(scatter, dim=0, reduce="min", feat_type=feat_type, **kwargs))
        elif this_pool == "std":
            pool_layer.append(PoolingWrapperPyg(scatter_std_pool, dim=0, feat_type=feat_type, **kwargs))
        elif (this_pool == "none") or (this_pool is None):
            pass
        else:
            raise NotImplementedError(f"Undefined pooling `{this_pool}`")

    return pool_layer, out_pool_dim


class VirtualNodePyg(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: Optional[int],
        out_dim_edges: Optional[int],
        vn_type: Union[type(None), str] = "sum",
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        bias: bool = True,
        residual: bool = True,
        use_edges: bool = False,
        **kwargs,
    ):
        r"""
        The VirtualNode is a layer that pool the features of the graph,
        applies a neural network layer on the pooled features,
        then add the result back to the node features of every node.

        Parameters:

            in_dim:
                Input feature dimensions of the virtual node layer.

            out_dim:
                Output feature dimensions of the virtual node layer.

            in_dim_edges:
                Input feature dimensions of the virtual node layer for the edges.

            out_dim_edges:
                Output feature dimensions of the virtual node layer for the edges.

            vn_type:
                The type of the virtual node. Choices are:

                - "none": No pooling
                - "sum": Sum all the nodes for each graph
                - "mean": Mean all the nodes for each graph
                - "logsum": Mean all the nodes then multiply by log(num_nodes) for each graph
                - "max": Max all the nodes for each graph
                - "min": Min all the nodes for each graph
                - "std": Standard deviation of all the nodes for each graph

            activation:
                activation function to use in the neural network layer.

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            bias:
                Whether to add a bias to the neural network

            residual:
                Whether all virtual nodes should be connected together
                via a residual connection

            use_edges:
                Boolean flag to select if edges are used in the global node
                aggregation and update of features

        """
        super().__init__()
        if (vn_type is None) or (vn_type.lower() == "none"):
            self.vn_type = None
            self.fc_layer = None
            self.residual = None
            return
        # Layer sizes stored here
        self.in_dim_edges = in_dim_edges
        self.out_dim_edges = out_dim_edges
        self.in_dim_nodes = in_dim
        self.out_dim_nodes = out_dim

        # Use edge features in pooling
        self.use_edges = use_edges
        has_edges = (
            (in_dim_edges is not None)
            and (out_dim_edges is not None)
            and (in_dim_edges > 0)
            and (out_dim_edges > 0)
        )
        if self.use_edges and not has_edges:
            raise ValueError("The edge features are not defined but `use_edges` is True")
        self.vn_type = vn_type.lower()
        self.layer, out_pool_dim = parse_pooling_layer_pyg(
            in_dim=self.in_dim_nodes, pooling=self.vn_type, feat_type="node"
        )
        self.residual = residual

        if self.use_edges:
            self.edge_layer, out_edge_pool_dim = parse_pooling_layer_pyg(
                in_dim=self.in_dim_edges, pooling=self.vn_type, feat_type="edge"
            )

            out_pool_dim = out_pool_dim + out_edge_pool_dim

        self.fc_layer = FCLayer(
            in_dim=out_pool_dim,
            out_dim=out_pool_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            bias=bias,
        )

        # Projection layers from the pooling layer to node and edge feature sizes
        self.node_projection = MuReadoutGraphium(out_pool_dim, self.out_dim_nodes)
        self.edge_projection = None
        if self.use_edges:
            self.edge_projection = MuReadoutGraphium(out_pool_dim, self.out_dim_edges)

    def forward(
        self, g: Union[Data, Batch], feat: Tensor, vn_feat: LongTensor, edge_feat: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Apply the virtual node layer.

        Parameters:

            g:
                PyG Graphs or Batched graphs.

            feat (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the input features

            vn_feat (torch.Tensor[..., M, Din]):
                Graph feature of the previous virtual node, or `None`
                `M` is the number of graphs, `Din` is the input features.
                It is added to the result after the MLP, as a residual connection

            edge_feat (torch.Tensor[..., E, Din]):
                Edge feature tensor, before convolution.
                `E` is the number of edges, `Din` is the input features

        Returns:

            `feat = torch.Tensor[..., N, Dout]`:
                Node feature tensor, after convolution and residual.
                `N` is the number of nodes, `Dout` is the output features of the layer and residual

            `vn_feat = torch.Tensor[..., M, Dout]`:
                Graph feature tensor to be used at the next virtual node, or `None`
                `M` is the number of graphs, `Dout` is the output features

            `edge_feat = torch.Tensor[..., N, Dout]`:
                Edge feature tensor, after convolution and residual - if edges are used, otherwise returned unchanged.
                `N` is the number of edges, `Dout` is the output features of the layer and residual

        """

        # Pool the features
        if self.vn_type is None:
            return feat, vn_feat, edge_feat
        pool = self.layer(g, feat)
        if self.use_edges:
            edge_pool = self.edge_layer(g, edge_feat)
            pool = torch.cat((pool, edge_pool), -1)

        vn_h_temp = self.fc_layer.forward(vn_feat + pool)

        if self.residual:
            vn_feat = vn_feat + vn_h_temp
        else:
            vn_feat = vn_h_temp

        # Add the virtual node projections to the node features
        feat = feat + self.node_projection(vn_feat[g.batch])

        if self.use_edges:
            # Add the virtual node projections to the edge features
            edge_feat = edge_feat + self.edge_projection(vn_feat[g.batch[g.edge_index][0]])
        return feat, vn_feat, edge_feat
