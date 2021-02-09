import torch
import torch.nn as nn
from typing import List

from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, Set2Set, GlobalAttentionPooling
from dgl import mean_nodes, sum_nodes, max_nodes

from goli.dgl.base_layers import MLP, FCLayer
from goli.commons.utils import ModuleListConcat


class S2SReadout(nn.Module):
    r"""
    Performs a Set2Set aggregation of all the graph nodes' features followed by a series of fully connected layers
    """

    def __init__(self, in_dim, hidden_dim, out_dim, fc_layers=3, device="cpu", final_activation="relu"):
        super().__init__()

        # set2set aggregation
        self.set2set = Set2Set(in_dim, device=device)

        # fully connected layers
        self.mlp = MLP(
            in_dim=2 * in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            layers=fc_layers,
            mid_activation="relu",
            last_activation=final_activation,
            mid_batch_norm=True,
            last_batch_norm=False,
            device=device,
        )

    def forward(self, x):
        x = self.set2set(x)
        return self.mlp(x)


class StdPooling(nn.Module):
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
            self.relu((self.sum_pooler(graph, feat ** 2)) - (self.sum_pooler(graph, feat) ** 2)) + EPS
        )
        return readout


class MinPooling(MaxPooling):
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


def parse_pooling_layer(in_dim: int, pooling: List[str], n_iters: int = 2, n_layers: int = 2):
    r"""
    Select the pooling layers from a list of strings, and put them
    in a Module that concatenates their outputs.

    Parameters:

        in_dim: int
            The dimension at the input layer of the pooling

        pooling: list(str)
            The list of pooling layers to use. The accepted strings are:

            - "sum": SumPooling
            - "mean": MeanPooling
            - "max": MaxPooling
            - "min": MinPooling
            - "std": StdPooling
            - "s2s": Set2Set

        n_iters: int
            IGNORED FOR ALL POOLING LAYERS, EXCEPT "s2s".
            The number of iterations.

        n_layers : int
            IGNORED FOR ALL POOLING LAYERS, EXCEPT "s2s".
            The number of recurrent layers.
    """

    # TODO: Add configuration for the pooling layer kwargs

    # Create the pooling layer
    pool_layer = ModuleListConcat()
    out_pool_dim = 0

    for this_pool in pooling:
        this_pool = None if this_pool is None else this_pool.lower()
        out_pool_dim += in_dim
        if this_pool == "sum":
            pool_layer.append(SumPooling())
        elif this_pool == "mean":
            pool_layer.append(AvgPooling())
        elif this_pool == "max":
            pool_layer.append(MaxPooling())
        elif this_pool == "min":
            pool_layer.append(MinPooling())
        elif this_pool == "std":
            pool_layer.append(StdPooling())
            pool_layer.append(Set2Set(input_dim=in_dim, n_iters=n_iters, n_layers=n_layers))
            out_pool_dim += in_dim
        elif (this_pool == "none") or (this_pool is None):
            pass
        else:
            raise NotImplementedError(f"Undefined pooling `{this_pool}`")

    return pool_layer, out_pool_dim


class VirtualNode(nn.Module):
    def __init__(
        self, dim, dropout, batch_norm=False, bias=True, activation="relu", residual=True, vn_type="sum"
    ):
        super().__init__()
        if (vn_type is None) or (vn_type.lower() == "none"):
            self.vn_type = None
            self.fc_layer = None
            self.residual = None
            return

        self.vn_type = vn_type.lower()
        self.residual = residual
        self.fc_layer = FCLayer(
            in_size=dim,
            out_size=dim,
            activation=activation,
            dropout=dropout,
            b_norm=batch_norm,
            bias=bias,
        )

    def forward(self, g, h, vn_h):

        g.ndata["h"] = h

        # Pool the features
        if self.vn_type is None:
            return vn_h, h
        elif self.vn_type == "mean":
            pool = mean_nodes(g, "h")
        elif self.vn_type == "max":
            pool = max_nodes(g, "h")
        elif self.vn_type == "sum":
            pool = sum_nodes(g, "h")
        elif self.vn_type == "logsum":
            pool = mean_nodes(g, "h")
            lognum = torch.log(torch.tensor(g.batch_num_nodes, dtype=h.dtype, device=h.device))
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
            [vn_h[ii : ii + 1].repeat(num_nodes, 1) for ii, num_nodes in enumerate(g.batch_num_nodes)],
            dim=0,
        )
        h = h + temp_h

        return vn_h, h
