from typing import Callable, Union, Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

from graphium.nn.base_graph_layer import BaseGraphModule
from graphium.utils.decorators import classproperty
from graphium.nn.pyg_layers.utils import triplets
from graphium.nn.base_layers import MLP, FCLayer


class ResidualLayer(torch.nn.Module):
    r"""Modified from
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
    (align style with our codebase)
    """

    def __init__(self, hidden_channels: int, activation: Union[Callable, str]):
        super().__init__()
        self.lin1 = FCLayer(hidden_channels, hidden_channels, activation=activation)
        self.lin2 = FCLayer(hidden_channels, hidden_channels, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.lin2(self.lin1(x))


class OutputBlock(torch.nn.Module):
    r"""Modified from
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
    (align style)
    """

    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        activation: Union[Callable, str],
    ):
        super().__init__()

        self.lin_rbf = FCLayer(num_radial, hidden_channels, bias=False, activation=None)
        self.lins = nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(FCLayer(hidden_channels, hidden_channels, activation=activation))
        self.lin = FCLayer(hidden_channels, out_channels, bias=False, activation=None)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce="sum")
        for lin in self.lins:
            x = lin(x)
        return self.lin(x)


class InteractionBlock(nn.Module):
    r"""Modified from
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
    (align style; add output linear layer to allow change of dimension)
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        activation: Union[Callable, str],
    ):
        super().__init__()

        self.lin_rbf = FCLayer(num_radial, hidden_dim, activation=None, bias=False)
        self.lin_sbf = FCLayer(num_spherical * num_radial, num_bilinear, activation=None, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = FCLayer(hidden_dim, hidden_dim, activation=activation)
        self.lin_ji = FCLayer(hidden_dim, hidden_dim, activation=activation)

        self.W = nn.Parameter(torch.Tensor(hidden_dim, num_bilinear, hidden_dim))
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))

        self.layers_before_skip = nn.ModuleList(
            [ResidualLayer(hidden_dim, activation) for _ in range(num_before_skip)]
        )
        self.lin = FCLayer(hidden_dim, hidden_dim, activation=activation)
        self.layers_after_skip = nn.ModuleList(
            [ResidualLayer(hidden_dim, activation) for _ in range(num_after_skip)]
        )
        self.lin_out = FCLayer(hidden_dim, output_dim, activation=activation)

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor, idx_ji: Tensor) -> Tensor:
        """
        Parameters:
            x: edge features after encodings [num_edges, hidden_dim]
            rbf: bessel rbf of edges [num_edges, num_radial]
            sbf: spherical bessel rbf of triplets [num_triplet, num_spherical * num_radial]
            idx_kj: indices in edge of triplets [num_triplet] (value range from 0 to num_edges)
            idx_ji: indices in edge of triplets [num_triplet] (value range from 0 to num_edges)
        """
        rbf = self.lin_rbf(rbf)  # [num_edges, hidden_dim]
        sbf = self.lin_sbf(sbf)  # [num_triplet, hidden_dim]

        x_ji = self.lin_ji(x)
        x_kj = self.lin_kj(x)
        x_kj = x_kj * rbf

        x_kj = torch.einsum("wj,wl,ijl->wi", sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce="sum")

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.lin(h) + x
        for layer in self.layers_after_skip:
            h = layer(h)
        return self.lin_out(h)  # [num_edges, output_dim]


class DimeNetPyg(BaseGraphModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: int,
        out_dim_edges: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        **kwargs,
    ):
        r"""
        `"Directional Message Passing for Molecular Graphs" <https://arxiv.org/abs/2003.03123> paper.

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            in_dim_edges:
                Input feature dimensions of the edges

            out_dim_edges:
                Output feature dimensions of the edges

            num_bilinear:
                The dimension of bilinear layer in the interaction block

            num_spherical:
                The number of spherical harmonics

            num_radial:
                The number of radial basis functions

            num_before_skip:
                The number of residual layers before skip connection (default: 1)

            num_after_skip:
                The number of residual layers after skip connection (default: 2)

            num_output_layers:
                The number of output layers for a single output block (default: 3)

            activation:
                activation function to use in the layer

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            init_eps :
                Initial :math:`\epsilon` value, default: ``0``.

            learn_eps :
                If True, :math:`\epsilon` will be a learnable parameter.

        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            **kwargs,
        )

        # transform old node feature
        self.node_model = MLP(
            in_dim=in_dim,
            hidden_dims=in_dim,
            out_dim=out_dim,
            depth=2,
            activation=activation,
            normalization=self.normalization,
        )

        # update edge feature
        self.interaction_block = InteractionBlock(
            in_dim_edges,
            out_dim_edges,
            num_bilinear,
            num_spherical,
            num_radial,
            num_before_skip,
            num_after_skip,
            activation,
        )
        # updated edge feature -> new node feature
        self.output_block = OutputBlock(
            num_radial=num_radial,
            hidden_channels=out_dim_edges,
            out_channels=out_dim,
            num_layers=num_output_layers,
            activation=activation,
        )

    def forward(self, batch: Union[Data, Batch]) -> Union[Data, Batch]:
        r"""
        forward function of the layer
        Parameters:
            batch: pyg Batch graphs to pass through the layer
        Returns:
            batch: pyg Batch graphs
        """
        assert (
            "radius_edge_index" in batch
        ), "radius_edge_index not in batch, make sure to use 3D encoder firstly"
        # (j, i) = edge_index
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            batch.radius_edge_index, num_nodes=batch.feat.size(0)
        )
        x, P = batch.edge_feat, batch.feat
        rbf, sbf = batch.edge_rbf, batch.triplet_sbf

        # apply MLP to node embeddings
        P = self.node_model(P)  # [num_nodes, out_dim]

        # rbf and sbf should be computed during pos encoder
        x = self.interaction_block(x, rbf, sbf, idx_kj, idx_ji)
        P = P + self.output_block(x, rbf, i, num_nodes=batch.feat.size(0))  # [num_nodes, out_dim]

        batch.edge_feat = x  # updated edge features
        batch.feat = P  # updated node features

        return batch

    ############################################################################################################
    @classproperty
    def layer_supports_edges(cls) -> bool:
        r"""
        Return a boolean specifying if the layer type supports edges or not.

        Returns:

            supports_edges: bool
                Always ``True`` for the current class
        """
        return True

    @property
    def layer_inputs_edges(self) -> bool:
        r"""
        Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Always ``True`` for the current class
        """
        return True

    @property
    def layer_outputs_edges(self) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Always ``True`` for the current class
        """
        return True

    @property
    def out_dim_factor(self) -> int:
        r"""
        Get the factor by which the output dimension is multiplied for
        the next layer.

        For standard layers, this will return ``1``.

        But for others, such as ``GatLayer``, the output is the concatenation
        of the outputs from each head, so the out_dim gets multiplied by
        the number of heads, and this function should return the number
        of heads.

        Returns:

            int:
                Always ``1`` for the current class
        """
        return 1
