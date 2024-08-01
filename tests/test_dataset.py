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

from graphium.data import load_micro_zinc
from graphium.data.datamodule import MultitaskFromSmilesDataModule
from graphium.data.dataset import MultitaskDataset
from graphium.features import mol_to_pyggraph
from graphium.data.smiles_transform import smiles_to_unique_mol_ids
from graphium.data.utils import get_keys

import graphium_cpp

import numpy as np
import os.path as osp

TEMP_CACHE_DATA_PATH = "tests/temp_cache_0000"


def dataframes_to_dataset(dataframes_dict, case_num):
    task_names = [key for key in dataframes_dict.keys()]

    task_dataset_args = {}
    task_train_indices = {}
    task_val_indices = {}
    task_test_indices = {}
    for task in task_names:
        (
            smiles,
            labels,
            label_offsets,
            sample_idx,
            extras,
        ) = MultitaskFromSmilesDataModule._extract_smiles_labels(
            df=dataframes_dict[task],
            task_level="graph",
            smiles_col="SMILES",
            label_cols=task,
            idx_col=None,
            weights_col=None,
            weights_type=None,
        )
        num_molecules = len(smiles)
        task_dataset_args[task] = {
            "smiles": smiles,
            "labels": labels,
            "label_offsets": label_offsets,
            "extras": extras,
        }

        task_train_indices[task] = np.arange(num_molecules).tolist()
        task_val_indices[task] = []
        task_test_indices[task] = []

    fake_data_hash = "a1b2c3testdataset" + str(case_num)

    # The rest of the data preparation and caching is done in graphium_cpp.prepare_and_save_data
    normalizations = {task: {} for task in task_names}  # No normalization
    stage_data, all_stats, label_num_cols, label_dtypes = graphium_cpp.prepare_and_save_data(
        task_names,
        task_dataset_args,
        normalizations,
        TEMP_CACHE_DATA_PATH,
        fake_data_hash,
        task_train_indices,
        task_val_indices,
        task_test_indices,
        False,  # add_self_loop
        False,  # explicit_H
        0,  # preprocessing_n_jobs
        True,  # merge_equivalent_mols
    )

    stage_data = stage_data["train"]

    data_offsets = None
    if MultitaskFromSmilesDataModule.data_offsets_tensor_index() < len(stage_data):
        data_offsets = stage_data[MultitaskFromSmilesDataModule.data_offsets_tensor_index()]

    multitask_dataset = MultitaskDataset(
        about="test_dataset case" + str(case_num),
        data_path=osp.join(TEMP_CACHE_DATA_PATH, "train_" + fake_data_hash),
        featurize_smiles=mol_to_pyggraph,
        task_names=task_names,
        label_num_cols=label_num_cols,
        label_dtypes=label_dtypes,
        mol_file_data_offsets=data_offsets,
        concat_smiles_tensor=stage_data[MultitaskFromSmilesDataModule.concat_smiles_tensor_index()],
        smiles_offsets_tensor=stage_data[MultitaskFromSmilesDataModule.smiles_offsets_tensor_index()],
        num_nodes_tensor=stage_data[MultitaskFromSmilesDataModule.num_nodes_tensor_index()],
        num_edges_tensor=stage_data[MultitaskFromSmilesDataModule.num_edges_tensor_index()],
    )

    return multitask_dataset


class Test_Multitask_Dataset(ut.TestCase):
    # Then we can choose different rows and columns for the tests as we see fit.
    # Remember tests are supposed to be FAST, and reading from the file system multiple times slows things down.

    # Make sure that the inputs to single task datasets are always lists!
    # Do not pass a data frame itself, but turn it into a list to satisfy the type required.

    def test_multitask_dataset_case_1(self):
        """Case: different tasks, all with the same smiles set.
        - Check that for each task, all smiles are received from the initial DF.
        - Check that for each task, you have the same label values as the initial DF.
        """

        df_micro_zinc = load_micro_zinc()  # Has about 1000 molecules
        df = df_micro_zinc.iloc[0:4]
        num_unique_mols = 4

        # Here we take the microzinc dataset and split the labels up into 'SA', 'logp' and 'score' in order to simulate having multiple single-task datasets
        df_micro_zinc_SA = df[["SMILES", "SA"]]
        df_micro_zinc_logp = df[["SMILES", "logp"]]
        df_micro_zinc_score = df[["SMILES", "score"]]

        # We need to prepare the data for these dataframes.
        # We don't need to do featurization yet.
        dataframes = {
            "SA": df_micro_zinc_SA,
            "logp": df_micro_zinc_logp,
            "score": df_micro_zinc_score,
        }
        multitask_dataset = dataframes_to_dataset(dataframes, 1)

        # Check: The number of unique molecules equals the number of datapoints in the multitask dataset.
        self.assertEqual(num_unique_mols, multitask_dataset.__len__())

        # Check that for each task, you have the same label values as the initial DF.
        for idx in range(multitask_dataset.__len__()):
            smiles = df[["SMILES"]].iloc[idx].values[0]

            label_SA = df_micro_zinc_SA["SA"][idx]
            label_logp = df_micro_zinc_logp["logp"][idx]
            label_score = df_micro_zinc_score["score"][idx]

            # Search for the smiles string in the multitask dataset
            found_idx = -1
            for i in range(multitask_dataset.__len__()):
                if (
                    graphium_cpp.extract_string(
                        multitask_dataset.smiles_tensor, multitask_dataset.smiles_offsets_tensor, i
                    )
                    == smiles
                ):
                    found_idx = i
                    break

            item = multitask_dataset[found_idx]["labels"]

            # Compare labels
            self.assertEqual(label_SA, item["SA"])
            self.assertEqual(label_logp, item["logp"])
            self.assertEqual(label_score, item["score"])

    def test_multitask_dataset_case_2(self):
        """Case: Different tasks, but with no intersection in the smiles (each task has a unique set of smiles)
        - Check that the total dataset has as much smiles as all tasks together
        - Check that, for each task, only the smiles related to that task have values, and ensure the value is what's expected from the initial DF
        """
        df = load_micro_zinc()  # Has about 1000 molecules

        # Choose non-overlapping smiles by choosing specific rows from the original dataframe.
        df_rows_SA = df.iloc[0:200]  # 200 data points
        df_rows_logp = df.iloc[200:400]  # 200 data points
        df_rows_score = df.iloc[400:750]  # 350 data points
        total_data_points = 750

        dataframes = {
            "SA": df_rows_SA,
            "logp": df_rows_logp,
            "score": df_rows_score,
        }
        multitask_microzinc = dataframes_to_dataset(dataframes, 2)

        # The total dataset has as many molecules as there are smiles in all tasks put together
        self.assertEqual(total_data_points, multitask_microzinc.__len__())

        # For each task, only the smiles related to that task have values, and the value is what's expected from the initial DF.
        for idx in range(len(multitask_microzinc)):
            smiles = df[["SMILES"]].iloc[idx].values[0]

            task = "task"
            if idx in range(0, 200):
                task = "SA"
            elif idx in range(200, 400):
                task = "logp"
            elif idx in range(400, 750):
                task = "score"

            # Labels of that molecule
            label_df = df[[task]].iloc[idx].values[0]

            # Search for the smiles string in the multitask dataset
            found_idx = -1
            for i in range(multitask_microzinc.__len__()):
                if (
                    graphium_cpp.extract_string(
                        multitask_microzinc.smiles_tensor, multitask_microzinc.smiles_offsets_tensor, i
                    )
                    == smiles
                ):
                    found_idx = i
                    break

            item = multitask_microzinc[found_idx]["labels"]
            multitask_microzinc_labels = item.keys()

            assert task in multitask_microzinc_labels
            self.assertEqual(label_df, item[task])

            if task == "SA":
                self.assertFalse("score" in multitask_microzinc_labels)
                self.assertFalse("logp" in multitask_microzinc_labels)
            elif task == "logp":
                self.assertFalse("score" in multitask_microzinc_labels)
                self.assertFalse("SA" in multitask_microzinc_labels)
            elif task == "score":
                self.assertFalse("SA" in multitask_microzinc_labels)
                self.assertFalse("logp" in multitask_microzinc_labels)

    def test_multitask_dataset_case_3(self):
        """Case: Different tasks, but with semi-intersection (some smiles unique per task, some intersect)
        - Check that the total dataset has as much smiles as the unique number of smiles.
        - Check that for each task, you retrieve the same smiles as expected from the initial DF
        """
        df_micro_zinc = load_micro_zinc()  # Has about 1000 molecules
        df = df_micro_zinc.iloc[0:5]

        # Choose OVERLAPPING smiles by choosing specific rows from the original dataframe. The tasks will not necessarily have unique smiles.
        df_rows_SA = df.iloc[0:3]
        df_rows_logp = df.iloc[1:4]
        df_rows_score = df.iloc[3:5]
        total_data_points = 5

        dataframes = {
            "SA": df_rows_SA,
            "logp": df_rows_logp,
            "score": df_rows_score,
        }
        multitask_microzinc = dataframes_to_dataset(dataframes, 3)

        # The multitask dataset has as many molecules as there are unique smiles across the single task datasets.
        self.assertEqual(total_data_points, multitask_microzinc.__len__())


if __name__ == "__main__":
    ut.main()
