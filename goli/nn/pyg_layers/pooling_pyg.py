import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from typing import List, Union, Callable, Tuple, Optional

from torch_scatter import scatter
from torch_geometric.data import Data, Batch

from goli.nn.base_layers import MLP, FCLayer
from goli.utils.tensor import ModuleListConcat, ModuleWrap


EPS = 1e-6


def scatter_logsum_pool(x: Tensor, batch: LongTensor, dim: int = 0, dim_size: Optional[int] = None):
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

    :rtype: :class:`Tensor`
    """
    dim_size = int(batch.max().item() + 1) if dim_size is None else dim_size
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

    :rtype: :class:`Tensor`
    """
    dim_size = int(batch.max().item() + 1) if dim_size is None else dim_size
    mean = scatter(x, batch, dim=dim, out=None, dim_size=dim_size, reduce="mean")
    mean_squares = scatter(x * x, batch, dim=dim, out=None, dim_size=dim_size, reduce="mean")
    out = mean_squares - mean * mean
    return torch.sqrt(torch.relu(out) + 1e-5)


class PoolingWrapperPyg(ModuleWrap):
    def __init__(self, func, feat_type, *args, **kwargs) -> None:
        super().__init__(func, *args, **kwargs)
        self.feat_type = feat_type

    def forward(self, g, feature, *args, **kwargs):
        dim_size = g.num_graphs
        if self.feat_type is "node":
            index = g.batch
        elif self.feat_type is "edge":
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
        dim: int,
        dim_edges: int,
        global_latent: int,
        vn_type: Union[type(None), str] = "sum",
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        bias: bool = True,
        residual: bool = True,
        use_edges: bool = False,
        
    ):
        r"""
        The VirtualNode is a layer that pool the features of the graph,
        applies a neural network layer on the pooled features,
        then add the result back to the node features of every node.

        Parameters:

            dim:
                Input and output feature dimensions of the virtual node layer

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

            dim_edges:
                The dimmension of the edges, if edges are used, for the edge
                pooling layer

            global_latent:
                The global node latent size - the output dimmension for the FF layer

        """
        super().__init__()
        if (vn_type is None) or (vn_type.lower() == "none"):
            self.vn_type = None
            self.fc_layer = None
            self.residual = None
            return
        self.vn_type = vn_type.lower()
        self.layer, out_pool_dim = parse_pooling_layer_pyg(in_dim=dim, pooling=self.vn_type, feat_type="node")
        self.residual = residual

        self.use_edges = use_edges

        # If global latent not provided, default to output size of dim
        self.global_latent = global_latent
        # if global_latent is None:
        #     global_latent = dim

        # if self.use_edges:
        self.edge_layer, out_edge_pool_dim = parse_pooling_layer_pyg(
            in_dim=dim_edges, pooling=self.vn_type, feat_type="edge"
        )

        out_pool_dim = out_pool_dim + out_edge_pool_dim
            # TODO: Is it this? Are the edges being called even when they aren't
        #     dim += dim_edges
        # import ipdb; ipdb.set_trace()
        # if self.global_latent is not None:
        #     out_pool_dim += self.global_latent

        # import ipdb; ipdb.set_trace()
        self.fc_layer = FCLayer(
            in_dim=out_pool_dim,
            out_dim=dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            bias=bias,
        )
        # node_proj_in_dim = global_latent + dim
        self.node_projection = FCLayer(
            in_dim=dim,
            out_dim=dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            bias=bias,
        )

        # import ipdb; ipdb.set_trace()
        # if self.use_edges and self.global_latent:
        #     # edge_proj_in_dim = dim_edges #self.fc_layer.out_dim #+ dim_edges

        #     # if self.global_latent is not None:
        #     #     edge_proj_in_dim = global_latent + dim_edges
        #     #     dim_out_edges = 16
        #     # else:
        #     # edge_proj_in_dim = 32 
        #     #     dim_out_edges = 16
        #     # import ipdb; ipdb.set_trace()
        #     # dim_edges = 32
        #     # NOTE: This is unhappy because the mup can't tell what to do when there are options
        #     # This is true even if the exact same format as above. 
        #     edge_proj_in_dim = global_latent + dim_edges
        # import ipdb; ipdb.set_trace()
        
        """
        What is known about the issue:
            - The issue is only to do with this edges projection layer
            - If the shapes and sizes are set by hand it works perfectly
            - Despite the passed values being equal to the hardcoded values desired
            - The infinite projection thing means that the output is suggesting that it thinks the output is an OUTPUT 
                where the muP scaling should keep it fixed, however it is infinite
            - 'Change fan out to infinite dimmension?'
                - 
        """
        tmp_dim_edges = dim_edges
        tmp_in_pool_dim = out_pool_dim
        self.edge_projection = FCLayer(
            in_dim=dim,
            out_dim=dim_edges,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            bias=bias,
        )

    def forward(self, g: Union[Data, Batch], h: Tensor, vn_h: LongTensor, e: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Apply the virtual node layer.

        Parameters:

            g:
                PyG Graphs or Batched graphs.

            h (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the input features

            e (torch.Tensor[..., E, Din]):
                Edge feature tensor, before convolution.
                `E` is the number of edges, `Din` is the input features

            vn_h (torch.Tensor[..., M, Din]):
                Graph feature of the previous virtual node, or `None`
                `M` is the number of graphs, `Din` is the input features.
                It is added to the result after the MLP, as a residual connection

            batch

        Returns:

            `h = torch.Tensor[..., N, Dout]`:
                Node feature tensor, after convolution and residual.
                `N` is the number of nodes, `Dout` is the output features of the layer and residual

            `vn_h = torch.Tensor[..., M, Dout]`:
                Graph feature tensor to be used at the next virtual node, or `None`
                `M` is the number of graphs, `Dout` is the output features

        """

        # Pool the features
        if self.vn_type is None:
            return h, vn_h, e
        else:
            pool = self.layer(g, h)
        if self.use_edges:
            # import ipdb; ipdb.set_trace()
            edge_pool = self.edge_layer(g, e)
            pool = torch.cat((pool, edge_pool), -1)

        # Compute the new virtual node features
        # import ipdb; ipdb.set_trace()
        # TODO: Need to pick the pooling size so that the pool output is the same as the global latent size
        # import ipdb; ipdb.set_trace()
        # if self.global_latent is None:
        # import ipdb; ipdb.set_trace()
        if type(vn_h) is not float:
            print(f"V NODE: {vn_h.size(), pool.size()}")
        vn_h_temp = self.fc_layer.forward(vn_h + pool)
        # else:
            # vn_h_temp = torch.cat((vn_h, pool), -1)
            # vn_h_temp = self.fc_layer.forward(vn_h_temp)

        if self.residual:
            vn_h = vn_h + vn_h_temp
        else:
            vn_h = vn_h_temp

        # Add the virtual node value to the graph features
        # import ipdb; ipdb.set_trace()
        # h = h + vn_h[g.batch]
        # NOTE: Do we actually need the
        # if self.global_latent is not None:
            # h = torch.cat([h, vn_h[g.batch]], -1)
            # h = self.node_projection.forward(h)
        # else:
        print("nodes", (vn_h[g.batch]).size(), h.size(), self._node_projection(self.node_projection(vn_h[g.batch])).size())
        h = h + self.node_projection(vn_h[g.batch])

        print("edges", (vn_h[g.batch[g.edge_index][0]]).size(), e.size(), self.edge_projection(vn_h[g.batch[g.edge_index][0]]).size())
        e = e + self.edge_projection(vn_h[g.batch[g.edge_index][0]])
        # Add the virtual node to the edge features
        # if self.use_edges:
        #     if self.global_latent:
        #         # import ipdb; ipdb.set_trace()
        #         e = torch.cat([e, vn_h[g.batch[g.edge_index][0]]], -1)
        #         # import ipdb; ipdb.set_trace()
        # e = self.edge_projection.forward(e)
        #     else:
        # else:
        #     e = e

        return h, vn_h, e
