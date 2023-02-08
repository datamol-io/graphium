import torch
import torch.nn as nn
from goli.nn.base_layers import MLP
from goli.ipu.to_dense_batch import to_dense_batch, to_sparse_batch
from torch_geometric.data import Batch
from typing import Tuple
from torch import Tensor


class Preprocess3DPositions(nn.Module):
    """
    Compute 3D attention bias and 3D node features according to the 3D position information.
    """

    def __init__(self, num_heads, embed_dim, num_kernel):
        r"""
        Parameters:
            num_heads:
                Number of attention heads used in self-attention.
            embed_dim:
                Hidden dimension of node features.
            num_kernel:
                Number of gaussian kernels.

        """
        super().__init__()
        self.num_heads = num_heads
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        self.gaussian = GaussianLayer(self.num_kernel)
        self.gaussian_proj = MLP(
            in_dim=self.num_kernel,
            hidden_dim=self.num_kernel,
            out_dim=self.num_heads,
            layers=2,
            activation="gelu",
        )
        # make sure the 3D node feature has the same dimension as the embedding size
        # so that it can be added to the original node features
        self.node_proj = nn.Linear(self.num_kernel, self.embed_dim)

    def forward(self, batch: Batch, max_num_nodes_per_graph: int, on_ipu: bool) -> Tuple[Tensor, Tensor]:
        r"""
        Inputs:
            batch:
                Batch object.
            max_num_nodes_per_graph:
                Maximum number of nodes per graph.
            on_ipu:
                If model rus on IPU.
        """

        pos = batch.positions_3d # uncomment this when 3D positions are available
        #pos = torch.rand(
        #    batch.h.size()[0], 3, device=batch.h.device.type
        #)  # remove this when 3D positions are available
        batch_size = None if batch.h.device.type != "ipu" else batch.graph_is_true.shape[0]
        # pos: [batch, nodes, 3]
        # padding_mask: [batch, nodes]
        # idx: [totoal_nodes]
        pos, padding_mask, idx = to_dense_batch(
            pos,
            batch=batch.batch,
            batch_size=batch_size,
            max_num_nodes_per_graph=max_num_nodes_per_graph,
            drop_nodes_last_graph=on_ipu,
        )
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
        # unsqueezed mask size: [batch, 1, 1, nodes] apply on tensor [batch, num_heads, nodes, nodes]
        attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2),
            float("-10000"),
        )
        # unsqueezed mask size: [batch, 1, nodes, 1] apply on tensor [batch, nodes, nodes, num_kernel]
        distance_feature.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0)
        # [batch, nodes, num_kernel]
        distance_feature_sum = distance_feature.sum(dim=-2)
        # [batch, nodes, embed_dim]
        node_feature = self.node_proj(distance_feature_sum)
        # [total_nodes, embed_dim]
        node_feature = to_sparse_batch(node_feature, idx)

        return attn_bias, node_feature


class GaussianLayer(nn.Module):
    def __init__(self, num_kernels=128):
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
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, input: Tensor) -> Tensor:
        # [batch, nodes, nodes, 1]
        input = input.unsqueeze(-1)
        # [batch, nodes, nodes, num_kernels]
        expanded_input = input.expand(-1, -1, -1, self.num_kernels)
        # [num_kernels]
        mean = self.means.weight.float().view(-1)
        # [num_kernels]
        std = self.stds.weight.float().view(-1).abs() + 0.01  # epsilon is 0.01 that matches gps++ value
        pi = 3.141592653
        pre_exp_factor = (2 * pi) ** 0.5
        # [batch, nodes, nodes, num_kernels]
        tensor_with_kernel = torch.exp(-0.5 * (((expanded_input - mean) / std) ** 2)) / (pre_exp_factor * std)
        return tensor_with_kernel
