import math
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Tuple
from torch import Tensor

from goli.nn.base_layers import MLP, get_norm
from goli.ipu.to_dense_batch import to_dense_batch, to_sparse_batch


class PreprocessPositions(nn.Module):
    """
    Compute 3D attention bias and 3D node features according to the 3D position information.
    """

    def __init__(
        self,
        num_heads,
        embed_dim,
        num_kernel,
        in_dim=3,
        num_layers=2,
        activation="gelu",
        first_normalization="none",
    ):
        r"""
        Parameters:
            num_heads:
                Number of attention heads used in self-attention.
            embed_dim:
                Hidden dimension of node features.
            num_kernel:
                Number of gaussian kernels.
            num_layers: The number of layers in the MLP.
            activation: The activation function used in the MLP.
            first_normalization: The normalization function used before the gaussian kernel.

        """
        super().__init__()
        self.num_heads = num_heads
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        self.first_normalization = get_norm(first_normalization, dim=in_dim)

        self.gaussian = GaussianLayer(self.num_kernel, in_dim=in_dim)
        self.gaussian_proj = MLP(
            in_dim=self.num_kernel,
            hidden_dims=self.num_kernel,
            out_dim=self.num_heads,
            depth=num_layers,
            activation=activation,
            last_layer_is_readout=True,  # Since the output is not proportional to the hidden dim, but to the number of heads
        )

        # make sure the 3D node feature has the same dimension as the embedding size
        # so that it can be added to the original node features
        self.node_proj = nn.Linear(self.num_kernel, self.embed_dim)

    def forward(
        self, batch: Batch, max_num_nodes_per_graph: int, on_ipu: bool, positions_3d_key: str
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Inputs:
            batch:
                Batch object.
            max_num_nodes_per_graph:
                Maximum number of nodes per graph.
            on_ipu:
                If model rus on IPU.
            positions_3d_key:
                The key of the pyg graph object that contains the 3D positions.

        """

        pos = batch[positions_3d_key]
        if self.first_normalization is not None:
            pos = self.first_normalization(pos)
        batch_size = None if pos.device.type != "ipu" else batch.graph_is_true.shape[0]
        # batch_size = None if batch.feat.device.type != "ipu" else batch.graph_is_true.shape[0] #[Andy] batch.feat is only available after passing through layers, not a good attribute to check
        # pos: [batch, nodes, 3]
        # padding_mask: [batch, nodes]
        # idx: [totoal_nodes]
        pos, mask, idx = to_dense_batch(
            pos,
            batch=batch.batch,
            batch_size=batch_size,
            max_num_nodes_per_graph=max_num_nodes_per_graph,
            drop_nodes_last_graph=on_ipu,
        )
        # check nan with the pos from to_dense_batch,
        # and generate mask. 1 for nan, 0 for other values.
        # [batch, nodes]
        nan_mask = torch.isnan(pos)
        # we need the opposite of mask output
        padding_mask = ~mask
        # [batch, nodes]
        batch, n_node, _ = pos.shape
        # [batch, nodes, nodes, 3]
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        # [batch, nodes, nodes]
        distance = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        # [batch, nodes, nodes, num_kernel]
        distance_feature = self.gaussian(distance)
        # [batch, nodes, nodes, num_heads]
        attn_bias = self.gaussian_proj(distance_feature)
        # [batch, num_heads, nodes, nodes]
        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()
        # apply padding_mask on attn_bias
        # unsqueezed mask size: [batch, 1, 1, nodes] apply on tensor [batch, num_heads, nodes, nodes]
        attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2),
            float("-1000"),
        )
        # apply nan_mask on attn_bias
        # unsqueezed mask size: [batch, 1, 1, nodes] apply on tensor [batch, num_heads, nodes, nodes]
        attn_bias.masked_fill_(
            nan_mask.unsqueeze(1).unsqueeze(2),
            0.0,
        )
        # apply padding_mask on distance_feature
        # unsqueezed mask size: [batch, 1, nodes, 1] apply on tensor [batch, nodes, nodes, num_kernel]
        distance_feature.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0)
        # [batch, nodes, num_kernel]
        distance_feature_sum = distance_feature.sum(dim=-2)
        # Output of GaussianLayer is FP32, cast to dtype of self.node_proj here
        distance_feature_sum = distance_feature_sum.to(self.node_proj.weight.dtype)
        # [batch, nodes, embed_dim]
        node_feature = self.node_proj(distance_feature_sum)
        # apply nan_mask on node_feature
        # unsqueezed mask size: [batch, nodes, 1] apply on tensor [batch, nodes, embed_dim]
        node_feature.masked_fill(nan_mask.unsqueeze(-1).to(torch.bool), 0.0)
        # [total_nodes, embed_dim]
        node_feature = to_sparse_batch(node_feature, idx)

        return attn_bias, node_feature


class GaussianLayer(nn.Module):
    def __init__(self, num_kernels=128, in_dim=3):
        r"""
            Gaussian kernel function that applied on the all-to-all 3D distances.
        Parameters:
            num_kernels:
                Number of gaussian kernel used.
        """
        super().__init__()
        self.num_kernels = num_kernels
        self.means = nn.Embedding(1, num_kernels)
        self.stds = nn.Embedding(1, num_kernels)
        nn.init.uniform_(self.means.weight, 0, in_dim)
        nn.init.uniform_(self.stds.weight, 0, in_dim)

    def forward(self, input: Tensor) -> Tensor:
        # [batch, nodes, nodes, 1]
        input = input.unsqueeze(-1)
        # [batch, nodes, nodes, num_kernels]
        expanded_input = input.expand(-1, -1, -1, self.num_kernels)
        # [num_kernels]
        mean = self.means.weight.float().view(-1)
        # [num_kernels]
        std = self.stds.weight.float().view(-1).abs() + 0.01  # epsilon is 0.01 that matches gps++ value
        pre_exp_factor = (2 * math.pi) ** 0.5
        # [batch, nodes, nodes, num_kernels]
        tensor_with_kernel = torch.exp(-0.5 * (((expanded_input - mean) / std) ** 2)) / (pre_exp_factor * std)
        return tensor_with_kernel
