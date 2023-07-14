import pytest
import torch
from torch_geometric.data import Data, Batch
from graphium.ipu.to_dense_batch import to_dense_batch
from warnings import warn
try:
    import poptorch
except Exception as e:
    warn(f"Skipping this test because poptorch is not available.\n{e}")
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

from graphium.ipu.ipu_wrapper import PredictorModuleIPU



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

    def test_ipu_to_dense_batch(self):
        opts = poptorch.Options()

        # train_dataloader = poptorch.DataLoader(
        #     opts, [self.g1, self.g2], batch_size=1, shuffle=True, num_workers=1
        # )
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                
            def forward(self, x, batch):
                # Implement your function here
                return to_dense_batch(x, batch=batch)

        model = MyModel()
        model = model.eval()
        poptorch_model_inf = poptorch.inferenceModel(model, options=opts)
        # for data in train_dataloader:
        import ipdb; ipdb.set_trace()
        # poptorch_model_inf(data.x, data.batch)
        import ipdb; ipdb.set_trace()
        h_dense, mask, _ = to_dense_batch(
            self.bg.feat,
            batch=self.bg.batch,
            batch_size=None,
            max_num_nodes_per_graph=None,
            drop_nodes_last_graph=False,
        )
        # import ipdb; ipdb.set_trace()
        # Add assertions to check the output as needed

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


