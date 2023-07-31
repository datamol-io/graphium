import pytest
import torch
from torch_geometric.data import Data, Batch
from graphium.ipu.to_dense_batch import to_dense_batch
from warnings import warn


# General imports
import yaml
import unittest as ut
import numpy as np
from copy import deepcopy
from warnings import warn
from lightning import Trainer, LightningModule
from lightning_graphcore import IPUStrategy
from functools import partial

import torch
from torch.utils.data.dataloader import default_collate

# Current library imports
from graphium.config._loader import load_datamodule, load_metrics, load_architecture, load_accelerator


@pytest.mark.ipu
class TestIPUBatch:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.in_dim = 12
        self.out_dim = 12
        self.in_dim_edges = 10
        self.out_dim_edges = 10
        self.edge_idx1 = torch.stack(
            [torch.tensor([0, 1, 2, 3, 2], dtype=torch.int), torch.tensor([1, 2, 3, 0, 0], dtype=torch.int)]
        )
        self.edge_idx2 = torch.stack(
            [torch.tensor([0, 0, 0, 1], dtype=torch.int), torch.tensor([0, 1, 2, 0], dtype=torch.int)]
        )
        self.x1 = torch.randn(self.edge_idx1.max().item() + 1, self.in_dim, dtype=torch.float32)
        self.e1 = torch.randn(self.edge_idx1.shape[-1], self.in_dim_edges, dtype=torch.float32)
        self.x2 = torch.randn(self.edge_idx2.max().item() + 1, self.in_dim, dtype=torch.float32)
        self.e2 = torch.randn(self.edge_idx2.shape[-1], self.in_dim_edges, dtype=torch.float32)
        self.g1 = Data(feat=self.x1, edge_index=self.edge_idx1, edge_feat=self.e1)
        self.g2 = Data(feat=self.x2, edge_index=self.edge_idx2, edge_feat=self.e2)
        self.bg = Batch.from_data_list([self.g1, self.g2])
        self.attn_kwargs = {"embed_dim": self.in_dim, "num_heads": 2, "batch_first": True}

    # @pytest.mark.skip
    @pytest.mark.parametrize("max_num_nodes_per_graph, batch_size", [(10, 5), (20, 10), (30, 15)])
    def test_ipu_to_dense_batch(self, max_num_nodes_per_graph, batch_size):
        # Run this test only if poptorch is available
        try:
            import poptorch

            opts = poptorch.Options()
            opts.useIpuModel(True)

            class MyModel(torch.nn.Module):
                def __init__(self):
                    super(MyModel, self).__init__()

                def forward(self, x, batch):
                    return to_dense_batch(
                        x,
                        batch=batch,
                        batch_size=batch_size,
                        max_num_nodes_per_graph=max_num_nodes_per_graph,
                        drop_nodes_last_graph=False,
                    )

            model = MyModel()
            model = model.eval()
            poptorch_model_inf = poptorch.inferenceModel(model, options=opts)
            # for data in train_dataloader:
            out, mask, idx = poptorch_model_inf(self.bg.feat, self.bg.batch)
            # Check the output sizes
            assert out.size() == torch.Size([batch_size, max_num_nodes_per_graph, 12])
            # Check the mask for true / false values
            assert mask.size() == torch.Size([batch_size, max_num_nodes_per_graph])
            assert torch.sum(mask) == 7
            assert (mask[0][:4] == True).all()
            assert (mask[0][4:] == False).all()
            assert (mask[1][:3] == True).all()
            assert (mask[1][3:] == False).all()
            assert (mask[2:] == False).all()

            # Check the idx are all the true values in the mask
            assert (mask.flatten()[idx] == True).all()
            poptorch_model_inf.detachFromDevice()
        except ImportError:
            pytest.skip("Skipping this test because poptorch is not available")

    def test_ipu_to_dense_batch_no_batch_no_max_nodes(self):
        h_dense, mask = to_dense_batch(
            self.bg.feat,
            batch=None,
            batch_size=None,
            max_num_nodes_per_graph=None,
            drop_nodes_last_graph=False,
        )
        # Add assertions to check the output as needed
        assert torch.allclose(h_dense, self.bg.feat.unsqueeze(0), atol=1e-5), "Tensors are not equal"
        assert mask.size(1) == h_dense.size(1)
        assert mask.all().item(), "Not all values in the tensor are True"

    def test_ipu_to_dense_batch_no_batch(self):
        max_nodes_per_graph = 10
        h_dense, mask, id = to_dense_batch(
            self.bg.feat,
            batch=None,
            batch_size=None,
            max_num_nodes_per_graph=max_nodes_per_graph,
            drop_nodes_last_graph=False,
        )
        assert mask.size() == (1, max_nodes_per_graph)
        assert torch.sum(mask) == 7
        assert torch.equal(id, torch.arange(7))
        assert h_dense.size() == (1, max_nodes_per_graph, self.bg.feat.size(-1))

    def test_ipu_to_dense_batch_drop_last(self):
        out, mask, idx = to_dense_batch(
            self.bg.feat,
            batch=None,
            batch_size=None,
            max_num_nodes_per_graph=3,
            drop_nodes_last_graph=True,
        )
        # Add assertions to check the output as needed
        assert mask.size(1) == out.size(1)
        # Check the mask and output have been clipped
        assert mask.size() == torch.Size([1, 3])
        assert mask.all().item(), "Not all values in the tensor are True"
