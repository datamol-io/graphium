"""
Unit tests for the attention layer
"""

from ast import Assert
import numpy as np
import torch
import unittest as ut
from torch_geometric.data import Data, Batch
from copy import deepcopy
from graphium.ipu.to_dense_batch import to_dense_batch

from graphium.nn.base_layers import MultiheadAttentionMup


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class test_MultiHeadAttention(ut.TestCase):
    seed_everything(42)
    in_dim = 12
    out_dim = 12
    in_dim_edges = 10
    out_dim_edges = 10
    edge_idx1 = torch.stack([torch.tensor([0, 1, 2, 3, 2]), torch.tensor([1, 2, 3, 0, 0])])
    edge_idx2 = torch.stack([torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])])
    x1 = torch.randn(edge_idx1.max() + 1, in_dim, dtype=torch.float32)
    e1 = torch.randn(edge_idx1.shape[-1], in_dim_edges, dtype=torch.float32)
    x2 = torch.randn(edge_idx2.max() + 1, in_dim, dtype=torch.float32)
    e2 = torch.randn(edge_idx2.shape[-1], in_dim_edges, dtype=torch.float32)
    g1 = Data(feat=x1, edge_index=edge_idx1, edge_feat=e1)
    g2 = Data(feat=x2, edge_index=edge_idx2, edge_feat=e2)
    bg = Batch.from_data_list([g1, g2])

    attn_kwargs = {"embed_dim": in_dim, "num_heads": 2, "batch_first": True}

    def test_attention_class(self):
        bg = deepcopy(self.bg)
        seed_everything(42)
        attention_layer = MultiheadAttentionMup(biased_attention=False, **self.attn_kwargs)
        attention_layer.eval()
        seed_everything(42)
        attention_layer_bias = MultiheadAttentionMup(biased_attention=True, **self.attn_kwargs)
        attention_layer_bias.eval()

        h_dense, mask, _ = to_dense_batch(
            bg.feat,
            batch=bg.batch,
            batch_size=None,
            max_num_nodes_per_graph=None,
            drop_nodes_last_graph=False,
        )
        # attn_bias [batch, num_heads, nodes, nodes]
        nodes = h_dense.size()[1]
        attn_bias_3d = torch.zeros(2, 2, nodes, nodes)
        h_dense.nodepair_gaussian_bias_3d = attn_bias_3d
        # Apply attention layer and attention layer with bias.
        h_attn_output = attention_layer(
            h_dense,
            h_dense,
            h_dense,
            attn_bias=None,
            attn_mask=None,
            key_padding_mask=~mask,
            need_weights=False,
        )[0]
        h_attn_output_biased = attention_layer_bias(
            h_dense,
            h_dense,
            h_dense,
            attn_bias=attn_bias_3d,
            attn_mask=None,
            key_padding_mask=~mask,
            need_weights=False,
        )[0]
        self.assertEqual(h_attn_output.size(), h_attn_output_biased.size())
        np.testing.assert_array_almost_equal(
            h_attn_output.detach().numpy(), h_attn_output_biased.detach().numpy(), decimal=0
        )


if __name__ == "__main__":
    ut.main()
