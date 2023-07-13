"""
Unit tests for the different layers of graphium/nn/pyg_layers/...

The layers are not thoroughly tested due to the difficulty of testing them
"""

import numpy as np
import torch
import unittest as ut
from torch_geometric.data import Data, Batch
from copy import deepcopy
import pytest

from graphium.nn.pyg_layers import (
    GINConvPyg,
    GINEConvPyg,
    MPNNPlusPyg,
    GatedGCNPyg,
    PNAMessagePassingPyg,
    GPSLayerPyg,
    VirtualNodePyg,
    DimeNetPyg,
)

from graphium.nn.pyg_layers.utils import (
    PreprocessPositions,
    GaussianLayer,
)


class test_Pyg_Layers(ut.TestCase):
    in_dim = 21
    out_dim = 11
    in_dim_edges = 13
    out_dim_edges = 17

    edge_idx1 = torch.stack([torch.tensor([0, 1, 2, 3, 2]), torch.tensor([1, 2, 3, 0, 0])])
    edge_idx2 = torch.stack([torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])])
    x1 = torch.randn(edge_idx1.max() + 1, in_dim, dtype=torch.float32)
    e1 = torch.randn(edge_idx1.shape[-1], in_dim_edges, dtype=torch.float32)
    x2 = torch.randn(edge_idx2.max() + 1, in_dim, dtype=torch.float32)
    e2 = torch.randn(edge_idx2.shape[-1], in_dim_edges, dtype=torch.float32)
    # edge_idx1, e1 = add_self_loops(edge_idx1, e1)
    # edge_idx2, e2 = add_self_loops(edge_idx2, e2)
    g1 = Data(feat=x1, edge_index=edge_idx1, edge_feat=e1)
    g2 = Data(feat=x2, edge_index=edge_idx2, edge_feat=e2)
    bg = Batch.from_data_list([g1, g2])

    kwargs = {
        "activation": "relu",
        "dropout": 0.1,
        "normalization": "batch_norm",
        "droppath_rate": 0.1,
        "layer_idx": 1,
        "layer_depth": 10,
    }

    def test_gpslayer(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        kwargs = deepcopy(self.kwargs)
        kwargs.pop("droppath_rate")
        kwargs["droppath_rate_attn"] = 0.2
        kwargs["droppath_rate_ffn"] = 0.3
        kwargs["mpnn_kwargs"] = {"droppath_rate_ffn": 0.4}

        layer = GPSLayerPyg(in_dim=self.in_dim, out_dim=self.out_dim, **kwargs)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer. Should crash due to different dim for nodes vs edges
        with self.assertRaises(ValueError):
            bg = layer.forward(bg)

        # Create new edge attributes with same dim and check that it works
        bg.edge_feat = torch.zeros((bg.edge_feat.shape[0], self.in_dim), dtype=torch.float32)
        bg = layer.forward(bg)
        self.assertEqual(bg.feat.shape[0], feat_in.shape[0])
        self.assertEqual(bg.feat.shape[1], self.out_dim * layer.out_dim_factor)

    def test_ginlayer(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        layer = GINConvPyg(in_dim=self.in_dim, out_dim=self.out_dim, **self.kwargs)

        # Check the re-implementation of abstract methods
        self.assertFalse(layer.layer_supports_edges)
        self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        bg = layer.forward(bg)
        self.assertEqual(bg.feat.shape[0], feat_in.shape[0])
        self.assertEqual(bg.feat.shape[1], self.out_dim * layer.out_dim_factor)

    def test_ginelayer(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        layer = GINEConvPyg(in_dim=self.in_dim, out_dim=self.out_dim, **self.kwargs)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer. Should crash due to different dim for nodes vs edges
        with self.assertRaises(ValueError):
            bg = layer.forward(bg)

        # Create new edge attributes with same dim and check that it works
        bg.edge_feat = torch.zeros((bg.edge_feat.shape[0], self.in_dim), dtype=torch.float32)
        bg = layer.forward(bg)
        self.assertEqual(bg.feat.shape[0], feat_in.shape[0])
        self.assertEqual(bg.feat.shape[1], self.out_dim * layer.out_dim_factor)

    def test_mpnnlayer(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        # need in_dim = out_dim for skip connection
        # mpnn layer accept different dimension for node and edge features
        layer = MPNNPlusPyg(
            in_dim=self.in_dim,
            out_dim=self.in_dim,
            use_edges=True,
            in_dim_edges=self.in_dim_edges,
            out_dim_edges=self.in_dim_edges,
            **self.kwargs,
        )

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertTrue(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Create new edge attributes with same dim and check that it works
        bg.edge_feat = torch.zeros((bg.edge_feat.shape[0], self.in_dim_edges), dtype=torch.float32)
        bg = layer.forward(bg)
        self.assertEqual(bg.feat.shape[0], feat_in.shape[0])
        self.assertEqual(bg.feat.shape[1], self.in_dim * layer.out_dim_factor)

    def test_gatedgcnlayer(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        e_in = bg.edge_feat
        layer = GatedGCNPyg(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            in_dim_edges=self.in_dim_edges,
            out_dim_edges=self.out_dim_edges,
            **self.kwargs,
        )

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertTrue(layer.layer_outputs_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        self.assertIsInstance(layer.layer_inputs_edges, bool)
        self.assertIsInstance(layer.layer_outputs_edges, bool)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer with edges
        bg2 = layer.forward(bg)
        self.assertEqual(bg2.feat.shape[0], feat_in.shape[0])
        self.assertEqual(bg2.feat.shape[1], self.out_dim * layer.out_dim_factor)

    def test_pnamessagepassinglayer(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        aggregators = ["mean", "max", "min", "std", "sum"]
        scalers = ["identity", "amplification", "attenuation"]

        layer = PNAMessagePassingPyg(
            in_dim=self.in_dim, out_dim=self.out_dim, aggregators=aggregators, scalers=scalers, **self.kwargs
        )

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        # self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        bg2 = layer.forward(bg)
        self.assertEqual(bg2.feat.shape[0], feat_in.shape[0])
        self.assertEqual(bg2.feat.shape[1], self.out_dim * layer.out_dim_factor)

        # Now try with edges
        bg = deepcopy(self.bg)
        layer = PNAMessagePassingPyg(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            aggregators=aggregators,
            scalers=scalers,
            in_dim_edges=self.in_dim_edges,
            **self.kwargs,
        )

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        # self.assertTrue(layer.layer_inputs_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        # self.assertIsInstance(layer.layer_inputs_edges, bool)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        bg2 = layer.forward(bg)
        self.assertEqual(bg2.feat.shape[0], feat_in.shape[0])
        self.assertEqual(bg2.feat.shape[1], self.out_dim * layer.out_dim_factor)
        self.assertTrue((bg2.edge_feat == self.bg.edge_feat).all)

    @pytest.mark.skip_ipu
    def test_dimenetlayer(self):
        from graphium.nn.encoders.bessel_pos_encoder import BesselSphericalPosEncoder

        bg = deepcopy(self.bg)
        # dummy input pos
        bg.pos = torch.randn((bg.feat.shape[0], 3), dtype=torch.float32)

        # position 3d encoder
        pos_enc = BesselSphericalPosEncoder(
            input_keys=["pos"],
            output_keys=[
                "node_feat",
                "edge_feat",
                "edge_rbf",
                "triplet_sbf",
                "radius_edge_index",
            ],  # The keys to return
            in_dim=3,
            out_dim=self.in_dim,
            out_dim_edges=self.in_dim_edges,
            num_output_layers=2,
            num_layers=2,
            num_spherical=4,
            num_radial=32,
        )

        enc_output = pos_enc(bg, None)  # forward requires: pos, edge_index, batch

        bg.feat = bg.feat + enc_output["node_feat"]  # [num_nodes, in_dim]
        bg.edge_feat = enc_output["edge_feat"]  # [num_edges, out_dim_edges]
        bg.radius_edge_index = enc_output["radius_edge_index"]  # [2, num_edges]
        bg.edge_rbf = enc_output["edge_rbf"]  # [num_edges, num_radial]
        bg.triplet_sbf = enc_output["triplet_sbf"]  # [num_triplets, num_spherical * num_radial]

        kwargs = deepcopy(self.kwargs)
        kwargs["num_bilinear"] = 32
        kwargs["num_spherical"] = 4
        kwargs["num_radial"] = 32

        layer = DimeNetPyg(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            in_dim_edges=self.in_dim_edges,
            out_dim_edges=self.out_dim_edges,
            **kwargs,
        )

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertTrue(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer with encoded feats as input
        bg2 = layer.forward(bg)
        # output sanity check
        self.assertEqual(bg2.edge_feat.shape, bg.edge_feat.shape)
        self.assertEqual(bg2.edge_feat.shape[1], self.out_dim_edges)
        self.assertEqual(bg2.feat.shape[1], self.out_dim)
        # not change rbf/sbf embedding
        self.assertTrue((bg2.edge_rbf == bg.edge_rbf).all)
        self.assertTrue((bg2.triplet_sbf == bg.triplet_sbf).all)

    def test_preprocess3Dfeaturelayer(self):
        bg = deepcopy(self.bg)
        num_heads = 2
        num_kernel = 2
        in_dim = 3
        bg.positions_3d = torch.zeros(bg.feat.size()[0], in_dim)
        layer = PreprocessPositions(
            num_heads=num_heads,
            embed_dim=self.out_dim,
            num_kernel=num_kernel,
            in_dim=in_dim,
            first_normalization="layer_norm",
        )
        # bias: [batch, num_heads, nodes, nodes]
        # node_feature: [total_nodes, embed_dim]
        bias, node_feature = layer.forward(
            bg, max_num_nodes_per_graph=4, on_ipu=False, positions_3d_key="positions_3d"
        )
        self.assertEqual(bias.size(), torch.Size([2, num_heads, 4, 4]))
        self.assertFalse(np.isnan(bias.detach().numpy()).any())
        self.assertEqual(node_feature.size(), torch.Size([7, self.out_dim]))

    def test_gaussianlayer(self):
        num_kernels = 3
        input = torch.zeros(2, 4, 4)
        layer = GaussianLayer(num_kernels=num_kernels)
        # tensor_with_kernel: [batch, nodes, nodes, num_kernel]
        tensor_with_kernel = layer.forward(input)
        self.assertEqual(tensor_with_kernel.size(), torch.Size([2, 4, 4, 3]))

    def test_pooling_virtual_node(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        e_in = bg.edge_feat
        vn_h = 0

        vn_types = ["sum", "mean"]
        for vn_type, use_edges, expected_v_node in zip(vn_types, [False, True], [(2, 21), (2, 34)]):
            with self.subTest(
                vn_type=vn_type,
                use_edges=use_edges,
                expected_v_node=expected_v_node,
            ):
                vn_h = 0.0
                print(vn_h, self.out_dim, feat_in.size())
                layer = VirtualNodePyg(
                    in_dim=self.in_dim,
                    out_dim=self.in_dim,
                    in_dim_edges=self.in_dim_edges,
                    out_dim_edges=self.in_dim_edges,
                    vn_type=vn_type,
                    use_edges=use_edges,
                    **self.kwargs,
                )

                feat_out, vn_out, e_out = layer.forward(bg, feat_in, vn_h, edge_feat=e_in)
                assert vn_out.shape == expected_v_node
                assert feat_out.shape == (7, 21)
                assert e_out.shape == (9, 13)
                if use_edges is False:
                    # i.e. that the node features have been updated
                    assert torch.equal(feat_out, feat_in) == False
                    # And that the edge features have not
                    assert torch.equal(e_out, e_in) == True
                else:
                    assert torch.equal(feat_out, feat_in) == False
                    assert torch.equal(e_out, e_in) == False


if __name__ == "__main__":
    ut.main()
