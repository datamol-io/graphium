"""
Unit tests for the different layers of goli/dgl/dgl_layers/...

The layers are not thoroughly tested due to the difficulty of testing them
"""

import numpy as np
import torch
import unittest as ut
import dgl
from copy import deepcopy

from goli.dgl.architectures import FeedForwardNN, FeedForwardDGL
from goli.dgl.base_layers import FCLayer


class test_FeedForwardNN(ut.TestCase):

    kwargs = {
        "activation": "relu",
        "last_activation": "none",
        "batch_norm": False,
        "dropout": 0.2,
        "name": "LNN",
        "layer_type": FCLayer,
    }

    def test_forward_no_residual(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [5, 6, 7]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="none",
            residual_skip_steps=1,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_simple_residual_1(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="simple",
            residual_skip_steps=1,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_simple_residual_2(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="simple",
            residual_skip_steps=2,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, hidden_dims[4])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_concat_residual_1(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="concat",
            residual_skip_steps=1,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, 2 * hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, 2 * hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, 2 * hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, 2 * hidden_dims[4])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_concat_residual_2(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="concat",
            residual_skip_steps=2,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, 1 * hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, 1 * hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, 2 * hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, 1 * hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, 2 * hidden_dims[4])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_densenet_residual_1(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="densenet",
            residual_skip_steps=1,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, 2 * hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, 3 * hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, 4 * hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, 5 * hidden_dims[4])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_densenet_residual_2(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="densenet",
            residual_skip_steps=2,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, 1 * hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, 2 * hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, 1 * hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, 3 * hidden_dims[4])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_weighted_residual_1(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="weighted",
            residual_skip_steps=1,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, hidden_dims[4])

        self.assertEqual(len(lnn.residual_layer.residual_list), len(hidden_dims))

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_weighted_residual_2(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="weighted",
            residual_skip_steps=2,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, hidden_dims[4])

        self.assertEqual(len(lnn.residual_layer.residual_list), (len(hidden_dims) // 2 + 1))

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])


if __name__ == "__main__":
    ut.main()
