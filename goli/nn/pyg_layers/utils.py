import torch
import torch.nn as nn
from goli.nn.base_layers import MLP
from goli.ipu.to_dense_batch import to_dense_batch, to_sparse_batch


class Preprocess3DPositions(nn.Module):
    """
        Compute 3D attention bias and 3D node features according to the 3D position information.
        """

    def __init__(self, num_heads, embed_dim, max_num_nodes_per_graph, num_kernel, on_ipu):
        super().__init__()
        self.num_heads = num_heads
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        self.on_ipu = on_ipu
        self.max_num_nodes_per_graph = max_num_nodes_per_graph

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


    def forward(self, batch):

        pos= batch.positions_3d
        batch_size = None if batch.h.device.type != "ipu" else batch.graph_is_true.shape[0]
        pos, mask, idx = to_dense_batch(
                pos,
                batch=batch.batch,
                batch_size=batch_size,
                max_num_nodes_per_graph=self.max_num_nodes_per_graph,
                drop_nodes_last_graph=self.on_ipu,
            )
        padding_mask = mask
        batch, n_node, _ = pos.shape # TODO: figure out if there is an extra dimension for pos 
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        # [batch, nodes, nodes] we may not have the batch dimension here as it is a sparse batch
        distance = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        distance /= distance.unsqueeze(-1) + 1e-5 # in order to avoid nans, change to 1e-3 with FP16
        # [batch, nodes, nodes, num_kernel]
        distance_feature = self.gaussian(distance)
        # [batch, nodes, nodes, num_heads]
        attn_bias = self.gaussian_proj(distance_feature)
        # [batch, num_heads, nodes, nodes]
        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()
        attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float('-inf') # in order to avoid nans, change -inf to -1000 for FP16
        )
        distance_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )
        distance_feature_sum = distance_feature.sum(dim=-2)
        node_feature = self.node_proj(distance_feature_sum)
        node_feature = to_sparse_batch(node_feature, idx)

        return attn_bias, node_feature

class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x):
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        pi = 3.14159
        a = (2*pi) ** 0.5   
        out = torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)
        return out
