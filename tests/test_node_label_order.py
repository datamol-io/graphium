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

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        ###################################################################################################################
        ### Test I: Test if atom labels are ordered correctly for a single dataset that contains only a single molecule ###
        ###################################################################################################################
            
        # Import node labels from parquet file
        df = pd.DataFrame(
            {
                "ordered_smiles": ["[C:0][C:1][O:2]"],
                "node_labels": [[0., 0., 2.]],
            }
        )

        task_kwargs = {"df": df, "split_val": 0.0, "split_test": 0.0}

        # Check datamodule with single task and two labels
        task_specific_args = {
            "task": {"task_level": "node", "label_cols": ["node_labels"], "smiles_col": "ordered_smiles", "seed": 42, **task_kwargs},
        }

        dm = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, featurization={"atom_property_list_onehot": ["atomic-number"]})
        dm.prepare_data()
        dm.setup()

        dm.train_ds.return_smiles = True

        dl = dm.train_dataloader()

        batch = next(iter(dl))

        atom_types = batch["labels"].node_task.squeeze()
        atom_types_from_features = batch["features"].feat.argmax(1)
        
        np.testing.assert_array_equal(atom_types, atom_types_from_features)

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)
            
        ###################################################################################
        ### Test II: Two ordered SMILES representing the same molecule in same dataset ###
        ###################################################################################

        # Create input data
        df = pd.DataFrame(
            {
                "ordered_smiles": ["[C:0][C:1][O:2]", "[O:0][C:1][C:2]"],
                "node_labels": [[0., 0., 2.], [2., 0., 0.]],
            }
        )

        task_kwargs = {"df": df, "split_val": 0.0, "split_test": 0.0}

        # Check datamodule with single task and two labels
        task_specific_args = {
            "task": {"task_level": "node", "label_cols": ["node_labels"], "smiles_col": "ordered_smiles", "seed": 42, **task_kwargs},
        }

        dm = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, featurization={"atom_property_list_onehot": ["atomic-number"]})
        dm.prepare_data()
        dm.setup()

        dm.train_ds.return_smiles = True

        dl = dm.train_dataloader()

        batch = next(iter(dl))

        atom_types = batch["labels"].node_task.squeeze()
        atom_types_from_features = batch["features"].feat.argmax(1)
        
        np.testing.assert_array_equal(atom_types_from_features, atom_types)

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        #############################################################################################
        ### Test III: Merging two node-level tasks each with different ordering of ordered SMILES ###
        #############################################################################################

        # Create input data
        df1 = pd.DataFrame(
            {
                "ordered_smiles": ["[C:0][C:1][O:2]"],
                "node_labels": [[0., 0., 2.]],
            }
        )
        
        df2 = pd.DataFrame(
            {
                "ordered_smiles": ["[O:0][C:1][C:2]"],
                "node_labels": [[2., 0., 0.]],
            }
        )

        task1_kwargs = {"df": df1, "split_val": 0.0, "split_test": 0.0}
        task2_kwargs = {"df": df2, "split_val": 0.0, "split_test": 0.0}

        # Check datamodule with single task and two labels
        task_specific_args = {
            "task1": {"task_level": "node", "label_cols": ["node_labels"], "smiles_col": "ordered_smiles", "seed": 42, **task1_kwargs},
            "task2": {"task_level": "node", "label_cols": ["node_labels"], "smiles_col": "ordered_smiles", "seed": 42, **task2_kwargs},
        }

        dm = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, featurization={"atom_property_list_onehot": ["atomic-number"]})
        dm.prepare_data()
        dm.setup()

        dm.train_ds.return_smiles = True

        dl = dm.train_dataloader()

        batch = next(iter(dl))

        unbatched_node_labels1 = unbatch(batch["labels"].node_task1, batch["labels"].batch)
        unbatched_node_labels2 = unbatch(batch["labels"].node_task2, batch["labels"].batch)
        unbatched_node_features = unbatch(batch["features"].feat, batch["features"].batch)

        atom_types1 = unbatched_node_labels1[0].squeeze()
        atom_types2 = unbatched_node_labels2[0].squeeze()
        atom_types_from_features = unbatched_node_features[0].argmax(1)
        
        np.testing.assert_array_equal(atom_types_from_features, atom_types1)
        np.testing.assert_array_equal(atom_types_from_features, atom_types2)

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        ###############################################################################
        ### Test IV: Merging node-level task on graph-level task with no node order ###
        ### NOTE: Works as rdkit does not merge ordered_smiles vs. unordered smiles ###
        ###############################################################################

        # Create input data
        df1 = pd.DataFrame(
            {
                "ordered_smiles": ["CCO"],
                "graph_labels": [1.],
            }
        )
        
        df2 = pd.DataFrame(
            {
                "ordered_smiles": ["[O:0][C:1][C:2]"],
                "node_labels": [[2., 0., 0.]],
            }
        )

        task1_kwargs = {"df": df1, "split_val": 0.0, "split_test": 0.0}
        task2_kwargs = {"df": df2, "split_val": 0.0, "split_test": 0.0}

        # Check datamodule with single task and two labels
        task_specific_args = {
            "task1": {"task_level": "graph", "label_cols": ["graph_labels"], "smiles_col": "ordered_smiles", "seed": 42, **task1_kwargs},
            "task2": {"task_level": "node", "label_cols": ["node_labels"], "smiles_col": "ordered_smiles", "seed": 42, **task2_kwargs},
        }

        dm = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, featurization={"atom_property_list_onehot": ["atomic-number"]})
        dm.prepare_data()
        dm.setup()

        dm.train_ds.return_smiles = True

        dl = dm.train_dataloader()

        batch = next(iter(dl))

        atom_types = batch["labels"].node_task2.squeeze()
        atom_types_from_features = batch["features"].feat.argmax(1)

        # Ignore NaNs
        nan_indices = atom_types.isnan()
        atom_types_from_features[nan_indices] = 333
        atom_types[nan_indices] = 333
        
        np.testing.assert_array_equal(atom_types, atom_types_from_features)

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        #####################################################################################
        ### Test V: Merging node-level task on graph-level task with different node order ###
        #####################################################################################

        # Create input data
        df1 = pd.DataFrame(
            {
                "ordered_smiles": ["[C:0][C:1][O:2]"],
                "graph_labels": [1.],
            }
        )
        
        df2 = pd.DataFrame(
            {
                "ordered_smiles": ["[O:0][C:1][C:2]"],
                "node_labels": [[2., 0., 0.]],
            }
        )

        task1_kwargs = {"df": df1, "split_val": 0.0, "split_test": 0.0}
        task2_kwargs = {"df": df2, "split_val": 0.0, "split_test": 0.0}

        # Check datamodule with single task and two labels
        task_specific_args = {
            "task1": {"task_level": "graph", "label_cols": ["graph_labels"], "smiles_col": "ordered_smiles", "seed": 42, **task1_kwargs},
            "task2": {"task_level": "node", "label_cols": ["node_labels"], "smiles_col": "ordered_smiles", "seed": 42, **task2_kwargs},
        }

        dm = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, featurization={"atom_property_list_onehot": ["atomic-number"]})
        dm.prepare_data()
        dm.setup()

        dm.train_ds.return_smiles = True

        dl = dm.train_dataloader()

        batch = next(iter(dl))

        atom_types = batch["labels"].node_task2.squeeze()
        atom_types_from_features = batch["features"].feat.argmax(1)
        
        np.testing.assert_array_equal(atom_types, atom_types_from_features)

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        ############################
        ### Test VI: ...         ###
        ### TODO: To be finished ###
        ############################

        # Create input data
        df = pd.DataFrame(
            {
                "smiles": ["CCO", "OCC", "COC", "[C:0][C:1][O:2]", "[O:0][C:1][C:2]", "[C:0][O:1][C:2]"],
                "graph_labels": [0., 0., 1., 0., 0., 1.],
            }
        )

        task_kwargs = {"df": df, "split_val": 0.0, "split_test": 0.0}

        # Check datamodule with single task and two labels
        task_specific_args = {
            "task": {"task_level": "graph", "label_cols": ["graph_labels"], "smiles_col": "smiles", "seed": 42, **task_kwargs},
        }

        dm = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, featurization={"atom_property_list_onehot": ["atomic-number"]})
        dm.prepare_data()
        dm.setup()

        dm.train_ds.return_smiles = True

        dl = dm.train_dataloader()

        batch = next(iter(dl))

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)


if __name__ == "__main__":
    ut.main()

    # Delete the cache
    if exists(TEMP_CACHE_DATA_PATH):
        rm(TEMP_CACHE_DATA_PATH, recursive=True)
