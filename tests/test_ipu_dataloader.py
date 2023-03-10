# General imports
import yaml
import unittest as ut
import numpy as np
from copy import deepcopy
from warnings import warn
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.strategies import IPUStrategy
from functools import partial

import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch

# Current library imports
from goli.ipu.ipu_dataloader import (
    smart_packing,
    get_pack_sizes,
    fast_packing,
    hybrid_packing,
    node_to_pack_indices_mask,
)
from goli.config._loader import load_datamodule, load_metrics, load_architecture


def random_packing(num_nodes, batch_size):
    ipu_batch_size = int(len(num_nodes) / batch_size)
    indices = np.arange(len(num_nodes))
    np.random.shuffle(indices)
    indices = np.reshape(indices, (ipu_batch_size, batch_size)).tolist()
    return indices


def global_batch_collator(batch_size, batches):
    packs = []
    for pack_idx in range(0, len(batches), batch_size):
        packs.append(default_collate(batches[pack_idx : pack_idx + batch_size]))
    global_batch = default_collate(packs)
    global_batch = (global_batch[0], tuple(global_batch[1]))
    return global_batch


class test_Packing(ut.TestCase):
    def test_smart_packing(self):
        np.random.seed(42)

        batch_sizes = [2, 4, 8, 16, 32, 64]
        ipu_batch_sizes = [2, 3, 4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            for ipu_batch_size in ipu_batch_sizes:
                err_msg = f"bz={batch_size}, ipu_bz={ipu_batch_size}"

                # Generate random batch size
                global_batch = batch_size * ipu_batch_size
                num_nodes = np.abs(np.random.gamma(2, 20, size=global_batch)).astype(int)

                # Use the smart packing
                packed_indices = smart_packing(num_nodes=num_nodes, batch_size=batch_size)
                pack_num_nodes = get_pack_sizes(packed_indices, num_nodes)

                # Use the random packing
                rand_packed_indices = random_packing(num_nodes=num_nodes, batch_size=batch_size)
                rand_pack_num_nodes = get_pack_sizes(rand_packed_indices, num_nodes)

                # Assert that the smart packing is better than the random packing
                self.assertLessEqual(max(pack_num_nodes), max(rand_pack_num_nodes), msg=err_msg)
                self.assertGreaterEqual(min(pack_num_nodes), min(rand_pack_num_nodes), msg=err_msg)

                # Assert that the total number of atoms is right
                self.assertEqual(sum(pack_num_nodes), sum(num_nodes), msg=err_msg)
                self.assertEqual(sum(rand_pack_num_nodes), sum(num_nodes), msg=err_msg)

                # Assert that all index are there
                self.assertListEqual(
                    np.sort(np.asarray(packed_indices).flatten()).tolist(), np.arange(len(num_nodes)).tolist()
                )
                self.assertListEqual(
                    np.sort(np.asarray(rand_packed_indices).flatten()).tolist(),
                    np.arange(len(num_nodes)).tolist(),
                )

    def test_fast_packing(self):
        np.random.seed(42)

        # Start at 4 for fast_packing for better statistical significance
        batch_sizes = [4, 8, 16, 32, 64]
        ipu_batch_sizes = [4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            for ipu_batch_size in ipu_batch_sizes:
                err_msg = f"bz={batch_size}, ipu_bz={ipu_batch_size}"

                # Generate random batch size
                global_batch = batch_size * ipu_batch_size
                num_nodes = np.abs(np.random.gamma(2, 20, size=global_batch)).astype(int)

                # Use the smart packing
                packed_indices = fast_packing(num_nodes=num_nodes, batch_size=batch_size)
                pack_num_nodes = get_pack_sizes(packed_indices, num_nodes)

                # Use the random packing
                rand_packed_indices = random_packing(num_nodes=num_nodes, batch_size=batch_size)
                rand_pack_num_nodes = get_pack_sizes(rand_packed_indices, num_nodes)

                # Assert that the smart packing is better than the random packing
                self.assertLessEqual(max(pack_num_nodes), max(rand_pack_num_nodes), msg=err_msg)
                self.assertGreaterEqual(min(pack_num_nodes), min(rand_pack_num_nodes), msg=err_msg)

                # Assert that the total number of atoms is right
                self.assertEqual(sum(pack_num_nodes), sum(num_nodes), msg=err_msg)
                self.assertEqual(sum(rand_pack_num_nodes), sum(num_nodes), msg=err_msg)

                # Assert that all index are there
                self.assertListEqual(
                    np.sort(np.asarray(packed_indices).flatten()).tolist(), np.arange(len(num_nodes)).tolist()
                )
                self.assertListEqual(
                    np.sort(np.asarray(rand_packed_indices).flatten()).tolist(),
                    np.arange(len(num_nodes)).tolist(),
                )

    def test_hybrid_packing(self):
        np.random.seed(42)

        batch_sizes = [2, 4, 8, 16, 32, 64]
        ipu_batch_sizes = [2, 3, 4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            for ipu_batch_size in ipu_batch_sizes:
                err_msg = f"bz={batch_size}, ipu_bz={ipu_batch_size}"

                # Generate random batch size
                global_batch = batch_size * ipu_batch_size
                num_nodes = np.abs(np.random.gamma(2, 20, size=global_batch)).astype(int)

                # Use the smart packing
                packed_indices = hybrid_packing(num_nodes=num_nodes, batch_size=batch_size)
                pack_num_nodes = get_pack_sizes(packed_indices, num_nodes)

                # Use the random packing
                rand_packed_indices = random_packing(num_nodes=num_nodes, batch_size=batch_size)
                rand_pack_num_nodes = get_pack_sizes(rand_packed_indices, num_nodes)

                # Assert that the smart packing is better than the random packing
                self.assertLessEqual(max(pack_num_nodes), max(rand_pack_num_nodes), msg=err_msg)
                self.assertGreaterEqual(min(pack_num_nodes), min(rand_pack_num_nodes), msg=err_msg)

                # Assert that the total number of atoms is right
                self.assertEqual(sum(pack_num_nodes), sum(num_nodes), msg=err_msg)
                self.assertEqual(sum(rand_pack_num_nodes), sum(num_nodes), msg=err_msg)

                # Assert that all index are there
                self.assertListEqual(
                    np.sort(np.asarray(packed_indices).flatten()).tolist(), np.arange(len(num_nodes)).tolist()
                )
                self.assertListEqual(
                    np.sort(np.asarray(rand_packed_indices).flatten()).tolist(),
                    np.arange(len(num_nodes)).tolist(),
                )

    def test_node_to_pack_indices_mask(self):
        # Create a dummy batch
        in_dim = 7
        in_dim_edges = 11
        max_num_nodes_per_graph = 20
        batch_size_per_pack = 5

        torch.manual_seed(42)

        # Create a dummy batch of graphs
        batch, all_num_nodes = [], []
        for ii in range(100):
            num_nodes = torch.randint(1, max_num_nodes_per_graph, (1,)).item()
            all_num_nodes.append(num_nodes)
            num_edges = abs(round(2.2 * num_nodes) + torch.randint(-2, 2, (1,)).item()) + 1
            x = torch.randn(num_nodes, in_dim, dtype=torch.float32)
            edge_idx = torch.randint(0, num_nodes, (2, num_edges))
            e = torch.randn(edge_idx.shape[-1], in_dim_edges, dtype=torch.float32)
            g = Data(h=x, edge_index=edge_idx, edge_attr=e)
            batch.append(g)
        batch = Batch.from_data_list(batch)

        # Get the packing
        packed_graph_idx = fast_packing(all_num_nodes, batch_size_per_pack)
        pack_sizes = get_pack_sizes(packed_graph_idx, all_num_nodes)
        max_pack_size = max(pack_sizes)
        num_packs = len(pack_sizes)

        # Get the node to pack indices and the mask
        node_to_pack_idx, pack_attn_mask = node_to_pack_indices_mask(packed_graph_idx, all_num_nodes)

        # Assert that the nodes to pack indices are correct
        h = torch.arange(batch.num_nodes, dtype=torch.float32)
        packed_shape = [num_packs, max_pack_size]
        h_packed = torch.zeros(packed_shape)
        h_packed[node_to_pack_idx[:, 0], node_to_pack_idx[:, 1]] = h
        h_packed_unique = torch.sort(torch.unique(h_packed))[0]
        np.testing.assert_array_equal(h_packed_unique, torch.arange(batch.num_nodes))
        self.assertEqual(h_packed.sum(), h.sum())

        # Test again with additional h dimension
        h = batch.h
        packed_shape = [num_packs, max_pack_size] + list(h.shape[1:])
        h_packed = torch.zeros(packed_shape)
        h_packed[node_to_pack_idx[:, 0], node_to_pack_idx[:, 1]] = h
        h_packed_unique = torch.sort(torch.unique(h_packed))[0]
        h_packed_unique = h_packed_unique[h_packed_unique != 0]
        np.testing.assert_array_almost_equal(h_packed_unique, torch.unique(h))
        self.assertAlmostEqual(h_packed.sum().item(), h.sum().item(), places=3)

        # Assert that the mask is correct by counting the number of False values (the sum of squared number of nodes per pack)
        num_false = (~pack_attn_mask).sum([1, 2])
        num_expected = torch.as_tensor(
            [sum([all_num_nodes[graph_idx] ** 2 for graph_idx in pack]) for pack in packed_graph_idx]
        )
        np.testing.assert_array_equal(num_false, num_expected)

        # Assert that the mask is correct by counting the number of elements in each row and column
        num_expected = []
        for pack in packed_graph_idx:
            pack_num_expected = []
            for graph_idx in pack:
                num_nodes = all_num_nodes[graph_idx]
                for ii in range(num_nodes):
                    pack_num_expected.append(num_nodes)
            pack_num_expected.extend([0] * (max_pack_size - len(pack_num_expected)))
            num_expected.append(pack_num_expected)
        num_expected = torch.as_tensor(num_expected)
        num_false_row = (~pack_attn_mask).sum([2])
        num_false_col = (~pack_attn_mask).sum([1])
        np.testing.assert_array_equal(num_false_row, num_expected)
        np.testing.assert_array_equal(num_false_col, num_expected)


# class test_DataLoading(ut.TestCase):
#     class TestSimpleLightning(LightningModule):
#         # Create a basic Ligthning for testing the batch sizes
#         def __init__(self, batch_size, node_feat_size, edge_feat_size, num_batch) -> None:
#             super().__init__()
#             self.batch_size = batch_size
#             self.node_feat_size = node_feat_size
#             self.edge_feat_size = edge_feat_size
#             self.layer = torch.nn.Linear(node_feat_size, 1)
#             self.loss_fn = torch.nn.L1Loss()
#             self.num_batch = num_batch

#         def validation_step(self, batch, batch_idx):
#             self.assert_shapes(batch, batch_idx, "val")
#             loss = self.forward(batch)
#             return loss

#         def training_step(self, batch, batch_idx):
#             self.assert_shapes(batch, batch_idx, "train")
#             loss = self.forward(batch)
#             return loss

#         def forward(self, batch):
#             out = self.layer(batch[1][0]).squeeze(-1)
#             loss = self.loss_fn(out, batch[0])
#             return loss

#         def assert_shapes(self, batch, batch_idx, step):
#             # Test the shape of the labels
#             this_shape = list(batch[0].shape)
#             true_shape = [1, self.batch_size]
#             assert (
#                 this_shape == true_shape
#             ), f"Shape of the labels is `{this_shape}` but should be {true_shape}"

#             # Test the shape of the first feature
#             this_shape = list(batch[1][0].shape)
#             true_shape = [1, self.batch_size, self.node_feat_size]
#             assert (
#                 this_shape == true_shape
#             ), f"Shape of the feature 0 is `{this_shape}` but should be {true_shape}"

#             # Test the shape of the second feature
#             this_shape = list(batch[1][1].shape)
#             true_shape = [1, self.batch_size, self.edge_feat_size]
#             assert (
#                 this_shape == true_shape
#             ), f"Shape of the feature 0 is `{this_shape}` but should be {true_shape}"

#         def configure_optimizers(self):
#             return torch.optim.Adam(self.parameters(), lr=1e-3)

#     class TestDataset(torch.utils.data.Dataset):
#         # Create a simple dataset for testing the Lightning integration
#         def __init__(self, labels, node_features, edge_features):
#             self.labels = labels
#             self.node_features = node_features
#             self.edge_features = edge_features

#         def __len__(self):
#             return len(self.labels)

#         def __getitem__(self, idx):
#             # [label, [feat1, feat2]]
#             return [self.labels[idx], [self.node_features[idx], self.edge_features[idx]]]

#     def test_poptorch_simple_deviceiterations_gradient_accumulation(self):
#         """
#         Test a simple version of the device-iterations and gradient accumulation
#         to make sure that the dataloader and models handle them correcly.
#         """

#         # Run this test only if poptorch is available
#         try:
#             import poptorch
#         except Exception as e:
#             warn(f"Skipping this test because poptorch is not available.\n{e}")
#             return

#         # Initialize constants
#         gradient_accumulation = 2
#         device_iterations = 3
#         batch_size = 5
#         num_replicate = 7
#         node_feat_size = 11
#         edge_feat_size = 13

#         # Initialize the batch info and poptorch options
#         opts = poptorch.Options()
#         opts.deviceIterations(device_iterations)
#         training_opts = deepcopy(opts)
#         training_opts.Training.gradientAccumulation(gradient_accumulation)
#         inference_opts = deepcopy(opts)

#         # Initialize the dataset
#         num_batch = device_iterations * gradient_accumulation * num_replicate
#         data_size = num_batch * batch_size
#         dataset = self.TestDataset(
#             labels=np.random.rand(data_size).astype(np.float32),
#             node_features=[np.random.rand(node_feat_size).astype(np.float32) for ii in range(data_size)],
#             edge_features=[np.random.rand(edge_feat_size).astype(np.float32) for ii in range(data_size)],
#         )

#         # Initialize the dataloader
#         train_dataloader = poptorch.DataLoader(
#             options=training_opts,
#             dataset=deepcopy(dataset),
#             batch_size=batch_size,
#             collate_fn=partial(global_batch_collator, batch_size),
#         )

#         val_dataloader = poptorch.DataLoader(
#             options=inference_opts,
#             dataset=deepcopy(dataset),
#             batch_size=batch_size,
#             collate_fn=partial(global_batch_collator, batch_size),
#         )

#         # Build the model, and run it on IPU
#         model = self.TestSimpleLightning(batch_size, node_feat_size, edge_feat_size, num_batch)
#         strategy = IPUStrategy(training_opts=training_opts, inference_opts=inference_opts)
#         trainer = Trainer(
#             logger=False,
#             enable_checkpointing=False,
#             max_epochs=2,
#             strategy=strategy,
#             num_sanity_val_steps=0,
#             ipus=1,
#         )
#         trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

#     def test_poptorch_goli_deviceiterations_gradient_accumulation(self):
#         """
#         Test the device-iterations and gradient accumulation in a way
#         that is very similar to the Goli code
#         to make sure that the dataloader and models handle them correcly.
#         """

#         try:
#             import poptorch
#         except Exception as e:
#             warn(f"Skipping this test because poptorch is not available.\n{e}")
#             return

#         from goli.ipu.ipu_wrapper import PredictorModuleIPU

#         class TestPredictor(PredictorModuleIPU):
#             # Create a basic Ligthning for testing the batch sizes
#             def __init__(self, batch_size, node_batch_size, edge_batch_size, in_dims, **kwargs) -> None:
#                 super().__init__(**kwargs)
#                 self.in_dims = in_dims
#                 self.batch_size = batch_size
#                 self.node_batch_size = node_batch_size
#                 self.edge_batch_size = edge_batch_size

#                 import poptorch

#                 self.pop = poptorch

#             def validation_step(self, features, labels):
#                 features, labels = self.squeeze_input_dims(features, labels)
#                 dict_input = {"features": features, "labels": labels}
#                 self.assert_shapes(dict_input, "val")
#                 preds = self.forward(dict_input)["preds"]
#                 loss = self.compute_loss(
#                     preds, dict_input["labels"], weights=None, loss_fun=self.loss_fun, target_nan_mask=None
#                 )
#                 loss = loss[0]
#                 return loss

#             def training_step(self, features, labels):
#                 features, labels = self.squeeze_input_dims(features, labels)
#                 dict_input = {"features": features, "labels": labels}
#                 preds = self.forward(dict_input)["preds"]
#                 loss = self.compute_loss(
#                     preds, dict_input["labels"], weights=None, loss_fun=self.loss_fun, target_nan_mask=None
#                 )
#                 loss = self.pop.identity_loss(loss[0], reduction="mean")
#                 return loss

#             def assert_shapes(self, dict_input, step):
#                 msg = f", step=`{step}`"

#                 # Test the shape of the labels
#                 this_shape = list(dict_input["labels"]["homo"].shape)
#                 true_shape = [self.batch_size, 2]
#                 assert (
#                     this_shape == true_shape
#                 ), f"Shape of the labels is `{this_shape}` but should be {true_shape}. {msg}"

#                 # Test the shape of the node feature
#                 this_shape = list(dict_input["features"]["feat"].shape)
#                 true_shape = [self.node_batch_size, self.in_dims["feat"]]
#                 assert (
#                     this_shape == true_shape
#                 ), f"Shape of the feature 0 is `{this_shape}` but should be {true_shape}. {msg}"

#                 # Test the shape of the node feature
#                 this_shape = list(dict_input["features"]["edge_feat"].shape)
#                 true_shape = [self.edge_batch_size, self.in_dims["edge_feat"]]
#                 assert (
#                     this_shape == true_shape
#                 ), f"Shape of the feature 0 is `{this_shape}` but should be {true_shape}. {msg}"

#             def get_progress_bar_dict(self):
#                 return {}

#             def on_train_batch_end(self, *args, **kwargs):
#                 return

#             def on_validation_batch_end(self, *args, **kwargs):
#                 return

#             def validation_epoch_end(self, *args, **kwargs):
#                 return

#             def on_train_epoch_end(self) -> None:
#                 return

#             def configure_optimizers(self):
#                 return torch.optim.Adam(self.parameters(), lr=1e-3)

#             def squeeze_input_dims(self, features, labels):
#                 for key, tensor in features:
#                     if isinstance(tensor, torch.Tensor):
#                         features[key] = features[key].squeeze(0)

#                 for key in labels:
#                     labels[key] = labels[key].squeeze(0)

#                 return features, labels

#         from goli.ipu.ipu_wrapper import DictIPUStrategy

#         gradient_accumulation = 3
#         device_iterations = 5
#         batch_size = 7

#         # Initialize the batch info and poptorch options
#         opts = poptorch.Options()
#         opts.deviceIterations(device_iterations)
#         training_opts = deepcopy(opts)
#         training_opts.Training.gradientAccumulation(gradient_accumulation)
#         inference_opts = deepcopy(opts)

#         # Load the configuration file for the model
#         CONFIG_FILE = "tests/config_test_ipu_dataloader.yaml"
#         with open(CONFIG_FILE, "r") as f:
#             cfg = yaml.safe_load(f)
#         cfg["datamodule"]["args"]["batch_size_training"] = batch_size
#         cfg["datamodule"]["args"]["batch_size_inference"] = batch_size
#         node_factor = cfg["datamodule"]["args"]["ipu_dataloader_training_opts"]["max_num_nodes_per_graph"]
#         edge_factor = cfg["datamodule"]["args"]["ipu_dataloader_training_opts"]["max_num_edges_per_graph"]

#         # Load the datamodule, and prepare the data
#         datamodule = load_datamodule(cfg)
#         datamodule.prepare_data()
#         datamodule.setup()
#         datamodule.ipu_training_opts = training_opts
#         datamodule.ipu_inference_opts = inference_opts

#         model_class, model_kwargs = load_architecture(cfg, in_dims=datamodule.in_dims)
#         metrics = load_metrics(cfg)
#         predictor = TestPredictor(
#             batch_size=batch_size,
#             node_batch_size=node_factor * batch_size,
#             edge_batch_size=edge_factor * batch_size,
#             in_dims=datamodule.in_dims,
#             model_class=model_class,
#             model_kwargs=model_kwargs,
#             metrics=metrics,
#             **cfg["predictor"],
#         )
#         strategy = DictIPUStrategy(training_opts=training_opts, inference_opts=inference_opts)
#         trainer = Trainer(
#             logger=False, enable_checkpointing=False, max_epochs=2, strategy=strategy, num_sanity_val_steps=0
#         )
#         trainer.fit(model=predictor, datamodule=datamodule)


if __name__ == "__main__":
    ut.main()
