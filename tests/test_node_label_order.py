"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


import unittest as ut

from graphium.utils.fs import rm, exists
from graphium.data import MultitaskFromSmilesDataModule

import torch
import pandas as pd
import numpy as np

from torch_geometric.utils import unbatch

TEMP_CACHE_DATA_PATH = "tests/temp_cache_0000"


class Test_NodeLabelOrdering(ut.TestCase):
    def test_node_label_ordering(self):
        # Import node labels from parquet fole
        parquet_file = "tests/data/dummy_node_label_order_data.parquet"
        task_kwargs = {"df_path": parquet_file, "split_val": 0.0, "split_test": 0.0}

        # Look at raw data
        raw_data = pd.read_parquet("tests/data/dummy_node_label_order_data.parquet")
        raw_labels = {
            smiles: torch.from_numpy(np.stack([label_1, label_2])).T for (smiles, label_1, label_2) in zip(raw_data["ordered_smiles"], raw_data["node_charges_mulliken"], raw_data["node_charges_lowdin"])
        }

        # Check datamodule with single task and two labels
        task_specific_args = {
            "task": {"task_level": "node", "label_cols": ["node_charges_mulliken", "node_charges_lowdin"], "smiles_col": "ordered_smiles", "seed": 42, **task_kwargs},
        }

        ds = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), 10)

        dl = ds.train_dataloader()

        batch = next(iter(dl))
        
        smiles = batch["smiles"]
        unbatched_node_labels = unbatch(batch["labels"].node_task, batch["labels"].batch)
        
        processed_labels = {
            smiles[idx]: unbatched_node_labels[idx] for idx in range(len(smiles))
        }

        for key in raw_labels.keys():
            assert torch.abs(raw_labels[key] - processed_labels[key]).max() < 1e-3

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        # Check datamodule with two tasks with each one label
        task_specific_args = {
            "task_1": {"task_level": "node", "label_cols": ["node_charges_mulliken"], "smiles_col": "ordered_smiles", "seed": 41, **task_kwargs},
            "task_2": {"task_level": "node", "label_cols": ["node_charges_lowdin"], "smiles_col": "ordered_smiles", "seed": 43, **task_kwargs},
        }

        ds = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), 10)

        dl = ds.train_dataloader()

        batch = next(iter(dl))
        
        smiles = batch["smiles"]
        unbatched_node_labels_1 = unbatch(batch["labels"].node_task_1, batch["labels"].batch)
        unbatched_node_labels_2 = unbatch(batch["labels"].node_task_2, batch["labels"].batch)
        
        processed_labels = {
            smiles[idx]: torch.cat([unbatched_node_labels_1[idx], unbatched_node_labels_2[idx]], dim=-1) for idx in range(len(smiles))
        }

        for key in raw_labels.keys():
            assert torch.abs(raw_labels[key] - processed_labels[key]).max() < 1e-3


if __name__ == "__main__":
    ut.main()

    # Delete the cache
    if exists(TEMP_CACHE_DATA_PATH):
        rm(TEMP_CACHE_DATA_PATH, recursive=True)
