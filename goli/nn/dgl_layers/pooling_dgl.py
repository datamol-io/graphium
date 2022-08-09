from copy import deepcopy
import torch
import torch.nn as nn
from typing import List, Union, Callable, Tuple

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, Set2Set, GlobalAttentionPooling
from dgl import mean_nodes, sum_nodes, max_nodes

from goli.nn.base_layers import MLP, FCLayer
from goli.utils.tensor import ModuleListConcat


EPS = 1e-6


class S2SReadoutDgl(nn.Module):
    r"""
    Performs a Set2Set aggregation of all the graph nodes' features followed by a series of fully connected layers
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        fc_layers=3,
        device="cpu",
        final_activation: Union[str, Callable] = "relu",
    ):
        super().__init__()

        # set2set aggregation
        self.set2set = Set2Set(in_dim, device=device)

        # fully connected layers
        self.mlp = MLP(
            in_dim=2 * in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            layers=fc_layers,
            activation="relu",
            last_activation=final_activation,
            normalization="batch_norm",
            last_normalization="none",
            device=device,
        )

    def forward(self, x):
        x = self.set2set(x)
        return self.mlp(x)


class StdPoolingDgl(nn.Module):
    r"""Apply standard deviation pooling over the nodes in the graph.

    $$r^{(i)} = \sigma_{k=1}^{N_i}\left( x^{(i)}_k \right)$$
    """

    def __init__(self):
        super().__init__()
        self.sum_pooler = SumPooling()
        self.relu = nn.ReLU()

    def forward(self, graph, feat):
        r"""Compute standard deviation pooling.

        Parameters:
            graph : DGLGraph
                The graph.
            feat : torch.Tensor
                The input feature with shape :math:`(N, *)` where
                :math:`N` is the number of nodes in the graph.

        Returns:
            torch.Tensor
                The output feature with shape :math:`(B, *)`, where
                :math:`B` refers to the batch size.
        """

        readout = torch.sqrt(
            self.relu((self.sum_pooler(graph, feat**2)) - (self.sum_pooler(graph, feat) ** 2)) + EPS
        )
        return readout


class MinPoolingDgl(MaxPooling):
    r"""Apply min pooling over the nodes in the graph.

    $$r^{(i)} = \min_{k=1}^{N_i}\left( x^{(i)}_k \right)$$
    """

    def forward(self, graph, feat):
        r"""Compute max pooling.

        Parameters:
            graph : DGLGraph
                The graph.
            feat : torch.Tensor
                The input feature with shape :math:`(N, *)` where
                :math:`N` is the number of nodes in the graph.

        Returns:
            readout: torch.Tensor
                The output feature with shape :math:`(B, *)`, where
                :math:`B` refers to the batch size.
        """

        return -super().forward(graph, -feat)


class DirPoolingDgl(nn.Module):
    r"""
    Apply pooling over the nodes in the graph using a directional potential
    with an inner product.

    In most cases, this is a pooling using the Fiedler vector.
    This is basically equivalent to computing a Fourier transform for the
    Fiedler vector. Then, we use the absolute value due to the sign ambiguity

    """

    def __init__(self, dir_idx):
        super().__init__()
        self.sum_pooler = SumPooling()
        self.dir_idx = dir_idx

    def forward(self, graph, feat):
        r"""Compute directional inner-product pooling, and return absolute value.

        Parameters:
            graph : DGLGraph
                The graph. Must have the key `graph.ndata["pos_dir"]`
            feat : torch.Tensor
                The input feature with shape :math:`(N, *)` where
                :math:`N` is the number of nodes in the graph.

        Returns:
            readout: torch.Tensor
                The output feature with shape :math:`(B, *)`, where
                :math:`B` refers to the batch size.
        """

        if not ("pos_dir" in graph.ndata.keys()):
            raise KeyError("`pos_dir` key missing. Positional encodings are required for the `DirPoolingDgl` layer, make sure you add them under the `dglGraph.ndata['pos_dir']` key.")

        dir = graph.ndata["pos_dir"][:, self.dir_idx].unsqueeze(-1)
        pooled = torch.abs(self.sum_pooler(graph, feat * dir))

        return pooled


class LogSumPoolingDgl(AvgPooling):
    r"""
    Apply pooling over the nodes in the graph using a mean aggregation,
    but scaled by the log of the number of nodes. This gives the same
    expressive power as the sum, but helps deal with graphs that are
    significantly larger than others by using a logarithmic scale.

    $$r^{(i)} = \frac{\log N_i}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k$$
    """

    def forward(self, graph, feat):
        r"""Compute log-sum pooling.

        Parameters:
            graph : DGLGraph
                The graph.
            feat : torch.Tensor
                The input feature with shape :math:`(N, *)` where
                :math:`N` is the number of nodes in the graph.

        Returns:
            readout: torch.Tensor
                The output feature with shape :math:`(B, *)`, where
                :math:`B` refers to the batch size.
        """
        mean_pool = super().forward(graph=graph, feat=feat)
        lognum = torch.log(torch.as_tensor(graph.batch_num_nodes(), dtype=feat.dtype, device=feat.device))
        pool = mean_pool * lognum.unsqueeze(-1)
        return pool


def parse_pooling_layer_dgl(in_dim: int, pooling: Union[str, List[str]], **kwargs):
    r"""
    Select the pooling layers from a list of strings, and put them
    in a Module that concatenates their outputs.

    Parameters:

        in_dim:
            The dimension at the input layer of the pooling

        pooling:
            The list of pooling layers to use. The accepted strings are:

            - "sum": `SumPooling`
            - "mean": `MeanPooling`
            - "max": `MaxPooling`
            - "min": `MinPooling`
            - "std": `StdPooling`
            - "s2s": `Set2Set`
            - "dir{int}": `DirPooling`

    """

    # Create the pooling layer
    pool_layer = ModuleListConcat()
    out_pool_dim = 0
    if isinstance(pooling, str):
        pooling = [pooling]

    for this_pool in pooling:
        this_pool = None if this_pool is None else this_pool.lower()
        out_pool_dim += in_dim
        if this_pool == "sum":
            pool_layer.append(SumPooling())
        elif this_pool == "mean":
            pool_layer.append(AvgPooling(**kwargs))
        elif this_pool == "logsum":
            pool_layer.append(LogSumPoolingDgl())
        elif this_pool == "max":
            pool_layer.append(MaxPooling())
        elif this_pool == "min":
            pool_layer.append(MinPoolingDgl())
        elif this_pool == "std":
            pool_layer.append(StdPoolingDgl())
        elif this_pool == "s2s":
            this_kwargs = deepcopy(kwargs)
            n_iters = this_kwargs.pop("n_iters", 2)
            n_layers = this_kwargs.pop("n_layers", 2)
            pool_layer.append(Set2Set(input_dim=in_dim, n_iters=n_iters, n_layers=n_layers, **this_kwargs))
            out_pool_dim += in_dim
        elif isinstance(this_pool, str) and (this_pool[:3] == "dir"):
            dir_idx = int(this_pool[3:])
            pool_layer.append(DirPoolingDgl(dir_idx=dir_idx))
        elif (this_pool == "none") or (this_pool is None):
            pass
        else:
            raise NotImplementedError(f"Undefined pooling `{this_pool}`")

    return pool_layer, out_pool_dim


class VirtualNodeDgl(nn.Module):
    def __init__(
        self,
        dim: int,
        vn_type: Union[type(None), str] = "sum",
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        bias: bool = True,
        residual: bool = True,
    ):
        r"""
        The VirtualNode is a layer that pool the features of the graph,
        applies a neural network layer on the pooled features,
        then add the result back to the node features of every node.

        Parameters:

            dim:
                Input and output feature dimensions of the virtual node layer

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

        """
        super().__init__()
        if (vn_type is None) or (vn_type.lower() == "none"):
            self.vn_type = None
            self.fc_layer = None
            self.residual = None
            return

        self.vn_type = vn_type.lower()
        self.residual = residual
        self.fc_layer = FCLayer(
            in_dim=dim,
            out_dim=dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            bias=bias,
        )

    def forward(
        self, g: dgl.DGLGraph, h: torch.Tensor, vn_h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply the virtual node layer.

        Parameters:

            g:
                graph on which the convolution is done

            h (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the input features

            vn_h (torch.Tensor[..., M, Din]):
                Graph feature of the previous virtual node, or `None`
                `M` is the number of graphs, `Din` is the input features.
                It is added to the result after the MLP, as a residual connection

        Returns:

            `h = torch.Tensor[..., N, Dout]`:
                Node feature tensor, after convolution and residual.
                `N` is the number of nodes, `Dout` is the output features of the layer and residual

            `vn_h = torch.Tensor[..., M, Dout]`:
                Graph feature tensor to be used at the next virtual node, or `None`
                `M` is the number of graphs, `Dout` is the output features

        """

        g.ndata["h"] = h

        # Pool the features
        if self.vn_type is None:
            return h, vn_h
        elif self.vn_type == "mean":
            pool = mean_nodes(g, "h")
        elif self.vn_type == "max":
            pool = max_nodes(g, "h")
        elif self.vn_type == "sum":
            pool = sum_nodes(g, "h")
        elif self.vn_type == "logsum":
            pool = mean_nodes(g, "h")
            lognum = torch.log(torch.tensor(g.batch_num_nodes(), dtype=h.dtype, device=h.device))
            pool = pool * lognum.unsqueeze(-1)
        else:
            raise ValueError(
                f'Undefined input "{self.pooling}". Accepted values are "none", "sum", "mean", "logsum"'
            )

        # Compute the new virtual node features
        vn_h_temp = self.fc_layer.forward(vn_h + pool)
        if self.residual:
            vn_h = vn_h + vn_h_temp
        else:
            vn_h = vn_h_temp

        # Add the virtual node value to the graph features
        temp_h = torch.cat(
            [vn_h[ii : ii + 1].repeat(num_nodes, 1) for ii, num_nodes in enumerate(g.batch_num_nodes())],
            dim=0,
        )
        h = h + temp_h

        return h, vn_h
