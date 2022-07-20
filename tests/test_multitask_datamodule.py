import pathlib
import tempfile

import unittest as ut
import numpy as np
from omegaconf import OmegaConf
import torch
import pandas as pd


import goli
from goli.data import load_micro_zinc, DGLDataset, SingleTaskDataset, MultitaskDGLDataset
from goli.data.datamodule import smiles_to_unique_mol_ids

class Test_Multitask_DataModule(ut.TestCase):
    
    def test_multitask_dglfromsmiles_dm(self):
        """Cover similar testing as for the original data module."""
        df = goli.data.load_tiny_zinc()     # 100 molecules

        # Here we take the microzinc dataset and split the labels up into 'SA', 'logp' and 'score' in order to simulate having multiple single-task datasets
        df_micro_zinc_SA = df[['SMILES', 'SA']]
        df_micro_zinc_logp = df[['SMILES', 'logp']]
        df_micro_zinc_score = df[['SMILES', 'score']]

        # Setup the featurization. This will be the same across all tasks.
        featurization_args = {}
        featurization_args["atom_property_list_float"] = []  # ["weight", "valence"]
        featurization_args["atom_property_list_onehot"] = ["atomic-number", "degree"]
        featurization_args["edge_property_list"] = ["in-ring", "bond-type-onehot"]
        featurization_args["add_self_loop"] = False
        featurization_args["use_bonds_weights"] = False
        featurization_args["explicit_H"] = False

        # Config for multitask datamodule. 

        # Per-task arguments.
        dm_task_args_SA = {}
        dm_task_args_SA["df"] = df_micro_zinc_SA
        dm_task_args_SA["smiles_col"] = "SMILES"
        dm_task_args_SA["label_cols"] = ["SA"]
        dm_task_args_SA["split_val"] = 0.2
        dm_task_args_SA["split_test"] = 0.2
        dm_task_args_SA["split_seed"] = 19
        dm_task_args_SA["splits_path"] = None                   # This may not always be provided
        dm_task_args_SA["sample_size"] = None                   # This may not always be provided
        dm_task_args_SA["idx_col"] = None                       # This may not always be provided
        dm_task_args_SA["weights_col"] = None                   # This may not always be provided
        dm_task_args_SA["weights_type"] = None                  # This may not always be provided

        dm_task_args_logp = {}
        dm_task_args_logp["df"] = df_micro_zinc_logp
        dm_task_args_logp["smiles_col"] = "SMILES"
        dm_task_args_logp["label_cols"] = ["logp"]
        dm_task_args_logp["split_val"] = 0.2
        dm_task_args_logp["split_test"] = 0.2
        dm_task_args_logp["split_seed"] = 19
        dm_task_args_logp["splits_path"] = None                 # This may not always be provided
        dm_task_args_logp["sample_size"] = None                 # This may not always be provided
        dm_task_args_logp["idx_col"] = None                     # This may not always be provided
        dm_task_args_logp["weights_col"] = None                 # This may not always be provided
        dm_task_args_logp["weights_type"] = None                # This may not always be provided

        dm_task_args_score = {}
        dm_task_args_score["df"] = df_micro_zinc_score
        dm_task_args_score["smiles_col"] = "SMILES"
        dm_task_args_score["label_cols"] = ["score"]
        dm_task_args_score["split_val"] = 0.2
        dm_task_args_score["split_test"] = 0.2
        dm_task_args_score["split_seed"] = 19
        dm_task_args_score["splits_path"] = None                # This may not always be provided
        dm_task_args_score["sample_size"] = None                # This may not always be provided
        dm_task_args_score["idx_col"] = None                    # This may not always be provided
        dm_task_args_score["weights_col"] = None                # This may not always be provided
        dm_task_args_score["weights_type"] = None               # This may not always be provided

        dm_task_kwargs = {}
        dm_task_kwargs["SA"] = dm_task_args_SA
        dm_task_kwargs["logp"] = dm_task_args_logp
        dm_task_kwargs["score"] = dm_task_args_score

        dm_args = {}

        # Task-specific arguments for the datamodule
        dm_args["task_specific_args"] = dm_task_kwargs

        # Task-independent arguments
        dm_args["featurization"] = featurization_args
        dm_args["featurization_n_jobs"] = 16
        dm_args["featurization_progress"] = True
        dm_args["featurization_backend"] = "multiprocessing"
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True
        dm_args["cache_data_path"] = None
        dm_args["batch_size_train_val"] = 16
        dm_args["batch_size_test"] = 16

        # Create the data module
        dm = goli.data.MultitaskDGLFromSmilesDataModule(**dm_args)

        #self.assertEqual(50, dm.num_node_feats)    # Not implemeneted error
        #self.assertEqual(6, dm.num_edge_feats)

        dm.prepare_data()
        dm.setup()

        #self.assertEqual(len(dm), 100)                      # Should this have a fixed value for when it's initialized? MTL dataset only gets created after.
        self.assertEqual(len(dm.train_ds), 60)  # type: ignore
        self.assertEqual(len(dm.val_ds), 20)    # type: ignore
        self.assertEqual(len(dm.test_ds), 20)   # type: ignore
        #assert dm.num_node_feats == 50
        #assert dm.num_edge_feats == 6

        for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            it = iter(dl)
            batch = next(it)
            
            assert set(batch.keys()) == {"labels", "features", "smiles", "mol_ids"}
            assert len(batch["mol_ids"]) == 16
            #assert len(batch["smiles"]) == 16
            #assert len(batch["features"])                      # This is not a list, but a graph.

            #assert batch["labels"].shape == (16, 1)            # Single-task case
            assert batch["labels"]["SA"].shape == (16, 1)
            assert batch["labels"]["logp"].shape == (16, 1)
            assert batch["labels"]["score"].shape == (16, 1)

    def test_multitask_dglfromsmiles_from_config(self):

        config = goli.load_config(name="zinc_default_multitask_fulldgl")

        df = goli.data.load_tiny_zinc()     # 100 molecules

        # Here we take the microzinc dataset and split the labels up into 'SA', 'logp' and 'score' in order to simulate having multiple single-task datasets
        df_micro_zinc_SA = df[['SMILES', 'SA']]
        df_micro_zinc_logp = df[['SMILES', 'logp']]
        df_micro_zinc_score = df[['SMILES', 'score']]

        #dm_args = dict(config.datamodule.args)
        dm_args = OmegaConf.to_container(config.datamodule.args, resolve=True)
        #dm_args["task_specific_args"]["SA"]["df"] = df
        dm_args["task_specific_args"]["SA"]["df"] = df_micro_zinc_SA
        dm_args["task_specific_args"]["logp"]["df"] = df_micro_zinc_logp
        dm_args["task_specific_args"]["score"]["df"] = df_micro_zinc_score

        dm_args["task_specific_args"]["SA"]["df_path"] = None
        dm_args["task_specific_args"]["logp"]["df_path"] = None
        dm_args["task_specific_args"]["score"]["df_path"] = None

        dm = goli.data.MultitaskDGLFromSmilesDataModule(**dm_args)

        #assert dm.num_node_feats == 50
        #assert dm.num_edge_feats == 6

        dm.prepare_data()
        dm.setup()

        #self.assertEqual(len(dm), 100)                      # Should this have a fixed value for when it's initialized? MTL dataset only gets created after.
        self.assertEqual(len(dm.train_ds), 60)  # type: ignore
        self.assertEqual(len(dm.val_ds), 20)    # type: ignore
        self.assertEqual(len(dm.test_ds), 20)   # type: ignore
        #assert dm.num_node_feats == 50
        #assert dm.num_edge_feats == 6

        for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            it = iter(dl)
            batch = next(it)
            
            assert set(batch.keys()) == {"labels", "features", "smiles", "mol_ids"}
            assert len(batch["mol_ids"]) == 16
            #assert len(batch["smiles"]) == 16
            #assert len(batch["features"])                      # This is not a list, but a graph.

            #assert batch["labels"].shape == (16, 1)            # Single-task case
            assert batch["labels"]["SA"].shape == (16, 1)
            assert batch["labels"]["logp"].shape == (16, 1)
            assert batch["labels"]["score"].shape == (16, 1)

    def test_multitask_dglfromsmiles_from_config_csv(self):
        config = goli.load_config(name="zinc_default_multitask_fulldgl")
     
        dm_args = OmegaConf.to_container(config.datamodule.args, resolve=True)
        dm = goli.data.MultitaskDGLFromSmilesDataModule(**dm_args)

        dm.prepare_data()
        dm.setup()

        #self.assertEqual(len(dm), 100)                      # Should this have a fixed value for when it's initialized? MTL dataset only gets created after.
        self.assertEqual(len(dm.train_ds), 60)  # type: ignore
        self.assertEqual(len(dm.val_ds), 20)    # type: ignore
        self.assertEqual(len(dm.test_ds), 20)   # type: ignore
        #assert dm.num_node_feats == 50
        #assert dm.num_edge_feats == 6

        for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            it = iter(dl)
            batch = next(it)
            
            assert set(batch.keys()) == {"labels", "features", "smiles", "mol_ids"}
            assert len(batch["mol_ids"]) == 16
            #assert len(batch["smiles"]) == 16
            #assert len(batch["features"])                      # This is not a list, but a graph.

            #assert batch["labels"].shape == (16, 1)            # Single-task case
            assert batch["labels"]["SA"].shape == (16, 1)
            assert batch["labels"]["logp"].shape == (16, 1)
            assert batch["labels"]["score"].shape == (16, 1)

class Test_Multitask_Dataset(ut.TestCase):

    def test_multitask_dataset_case_1(self):
        """Case: different tasks, all with the same smiles set.
            - Check that for each task, all smiles are received from the initial DF.
            - Check that for each task, you have the same label values as the initial DF.
        """

        df_micro_zinc = load_micro_zinc() # Has about 1000 molecules
        df = df_micro_zinc.iloc[0:4]
        num_unique_mols = 4

        # Here we take the microzinc dataset and split the labels up into 'SA', 'logp' and 'score' in order to simulate having multiple single-task datasets
        df_micro_zinc_SA = df[['SMILES', 'SA']]
        df_micro_zinc_logp = df[['SMILES', 'logp']]
        df_micro_zinc_score = df[['SMILES', 'score']]

        # We need to turn these dataframes into single-task datasets.
        # We don't need to do featurization yet.
        ds_micro_zinc_SA = SingleTaskDataset(
            smiles=df_micro_zinc_SA.loc[:,'SMILES'],
            labels=df_micro_zinc_SA.loc[:,'SA']
            )

        ds_micro_zinc_logp = SingleTaskDataset(
            smiles=df_micro_zinc_logp.loc[:, "SMILES"], labels=df_micro_zinc_logp.loc[:, "logp"]
        )
        ds_micro_zinc_score = SingleTaskDataset(

            smiles=df_micro_zinc_score.loc[:,'SMILES'],
            labels=df_micro_zinc_score.loc[:,'score']
            )

        # Create the multitask dataset
        datasets_dict = {'SA': ds_micro_zinc_SA, 'logp': ds_micro_zinc_logp, 'score': ds_micro_zinc_score}
        multitask_microzinc = MultitaskDGLDataset(datasets_dict) # Can optionally have features

        # Check: The number of unique molecules equals the number of datapoints in the multitask dataset.
        self.assertEqual(num_unique_mols, multitask_microzinc.__len__())

        # Check that for each task, you have the same label values as the initial DF.
        for idx in range(multitask_microzinc.__len__()):
            smiles = df[['SMILES']].iloc[idx].values[0]
            #label = df[['SA']].iloc[idx]
            label_SA = ds_micro_zinc_SA.labels[idx]
            label_logp = ds_micro_zinc_logp.labels[idx]
            label_score = ds_micro_zinc_score.labels[idx]

            # Search for the mol id in the multitask dataset
            mol_ids = smiles_to_unique_mol_ids([smiles])
            mol_id = mol_ids[0]
            found_idx = -1
            for i, id in enumerate(multitask_microzinc.mol_ids):
                if mol_id == id: found_idx = i

            # Compare labels
            self.assertEqual(label_SA, multitask_microzinc.labels[found_idx]['SA'])
            self.assertEqual(label_logp, multitask_microzinc.labels[found_idx]['logp'])
            self.assertEqual(label_score, multitask_microzinc.labels[found_idx]['score'])
<<<<<<< HEAD

        # Using a data loader
        #multitask_loader = torch.utils.data.DataLoader(multitask_microzinc, shuffle=False, num_workers=0, batch_size=1)
        #self.assertEqual(len(smiles), len(multitask_loader))
=======
>>>>>>> origin/master

    # def test_multitask_dataset_case_2(self):
    #     """Case: Different tasks, but with no intersection in the smiles (each task has a unique set of smiles)
    #         - Check that the total dataset has as much smiles as all tasks together
    #         - Check that, for each task, only the smiles related to that task have values, and ensure the value is what's expected from the initial DF
    #     """
    #     df = load_micro_zinc() # Has about 1000 molecules

    #     # Choose non-overlapping smiles by choosing specific rows from the original dataframe.
    #     df_rows_SA = df.iloc[0:200]         # 200 data points
    #     df_rows_logp = df.iloc[200:400]     # 200 data points
    #     df_rows_score = df.iloc[400:750]    # 350 data points
    #     total_data_points = 750

    #     # Here we split the data according to the task we care about.
    #     df_micro_zinc_SA = df_rows_SA[['SMILES', 'SA']]
    #     df_micro_zinc_logp = df_rows_logp[['SMILES', 'logp']]
    #     df_micro_zinc_score = df_rows_score[['SMILES', 'score']]

    #     # We need to turn these dataframes into single-task datasets.
    #     # We don't need to do featurization yet.
    #     ds_micro_zinc_SA = SingleTaskDataset(
    #         smiles=df_micro_zinc_SA.loc[:,'SMILES'], 
    #         labels=df_micro_zinc_SA.loc[:,'SA']
    #         )
    #     ds_micro_zinc_logp = SingleTaskDataset(
    #         smiles=df_micro_zinc_logp.loc[:,'SMILES'], 
    #         labels=df_micro_zinc_logp.loc[:,'logp']
    #         )
    #     ds_micro_zinc_score = SingleTaskDataset(
    #         smiles=df_micro_zinc_score.loc[:,'SMILES'], 
    #         labels=df_micro_zinc_score.loc[:,'score']
    #         )

<<<<<<< HEAD
        # We need to turn these dataframes into single-task datasets.
        # We don't need to do featurization yet.
        ds_micro_zinc_SA = SingleTaskDataset(
            smiles=df_micro_zinc_SA.loc[:,'SMILES'],
            labels=df_micro_zinc_SA.loc[:,'SA']
            )
        ds_micro_zinc_logp = SingleTaskDataset(
            smiles=df_micro_zinc_logp.loc[:,'SMILES'],
            labels=df_micro_zinc_logp.loc[:,'logp']
            )
        ds_micro_zinc_score = SingleTaskDataset(
            smiles=df_micro_zinc_score.loc[:,'SMILES'],
            labels=df_micro_zinc_score.loc[:,'score']
            )
=======
    #     # Create the multitask dataset
    #     datasets_dict = {'SA': ds_micro_zinc_SA, 'logp': ds_micro_zinc_logp, 'score': ds_micro_zinc_score}
    #     multitask_microzinc = MultitaskDGLDataset(datasets_dict) # Can optionally have features
>>>>>>> origin/master

    #     # The total dataset has as many molecules as there are smiles in all tasks put together
    #     self.assertEqual(total_data_points, multitask_microzinc.__len__())

    #     # For each task, only the smiles related to that task have values, and the value is what's expected from the initial DF.
    #     for idx in range(len(ds_micro_zinc_SA)):
    #         smiles = df[['SMILES']].iloc[idx].values[0]

    #         task = 'task'
    #         if idx in range(0, 200): task = 'SA'
    #         elif idx in range(200, 400): task = 'logp'
    #         elif idx in range(400, 750): task = 'score'

    #         # Labels of that molecule
    #         label_SA = df[['SA']].iloc[idx].values[0]
    #         label_logp = df[['logp']].iloc[idx].values[0]
    #         label_score = df[['score']].iloc[idx].values[0]

    #         # Search for that molecule in the multitask dataset
    #         mol_ids = smiles_to_unique_mol_ids([smiles])
    #         mol_id = mol_ids[0]
    #         found_idx = -1
    #         for i, id in enumerate(multitask_microzinc.mol_ids):
    #             if mol_id == id: found_idx = i

    #         if task == 'SA':
    #             self.assertEqual(label_SA, multitask_microzinc.labels[found_idx]['SA'])
    #             self.assertFalse('score' in multitask_microzinc.labels[found_idx].keys())
    #             self.assertFalse('logp' in multitask_microzinc.labels[found_idx].keys())
    #         elif task == 'logp':
    #             self.assertEqual(label_logp, multitask_microzinc.labels[found_idx]['logp'])
    #             self.assertFalse('score' in multitask_microzinc.labels[found_idx].keys())
    #             self.assertFalse('SA' in multitask_microzinc.labels[found_idx].keys())
    #         elif task == 'score':
    #             self.assertEqual(label_score, multitask_microzinc.labels[found_idx]['score'])
    #             self.assertFalse('SA' in multitask_microzinc.labels[found_idx].keys())
    #             self.assertFalse('logp' in multitask_microzinc.labels[found_idx].keys())

    # TODO (Gabriela): Fix this test case.
    # def test_multitask_dataset_case_3(self):
    #     """Case: Different tasks, but with semi-intersection (some smiles unique per task, some intersect)
    #         - Check that the total dataset has as much smiles as the unique number of smiles.
    #         - Check that for each task, you retrieve the same smiles as expected from the initial DF
    #     """
    #     df_micro_zinc = load_micro_zinc() # Has about 1000 molecules
    #     df = df_micro_zinc.iloc[0:5]


    #     # Choose OVERLAPPING smiles by choosing specific rows from the original dataframe. The tasks will not necessarily have unique smiles.
    #     df_rows_SA = df.iloc[0:3]         # 200 data points in 'SA' task, but 170-199 overlap with 'logp' task
    #     df_rows_logp = df.iloc[1:4]     # 200 data points in 'logp' task, but 170-199 overlap with 'SA' task, and 370-399 overlap with 'score' task
    #     df_rows_score = df.iloc[3:5]    # 350 data points in 'score' task, but 370-399 overlap with 'logp'
    #     total_data_points = 3             # There are 750 rows, but 60 smiles overlap, giving 690 unique molecules

    #     # Here we split the data according to the task we care about.
    #     df_micro_zinc_SA = df_rows_SA[['SMILES', 'SA']]
    #     df_micro_zinc_logp = df_rows_logp[['SMILES', 'logp']]
    #     df_micro_zinc_score = df_rows_score[['SMILES', 'score']]     

    #     # We need to turn these dataframes into single-task datasets.
    #     # We don't need to do featurization yet.
    #     ds_micro_zinc_SA = SingleTaskDataset(
    #         smiles=df_micro_zinc_SA.loc[:,'SMILES'], 
    #         labels=df_micro_zinc_SA.loc[:,'SA']
    #         )
    #     ds_micro_zinc_logp = SingleTaskDataset(
    #         smiles=df_micro_zinc_logp.loc[:,'SMILES'], 
    #         labels=df_micro_zinc_logp.loc[:,'logp']
    #         )
    #     ds_micro_zinc_score = SingleTaskDataset(
    #         smiles=df_micro_zinc_score.loc[:,'SMILES'], 
    #         labels=df_micro_zinc_score.loc[:,'score']
    #         )
   
    #     # Create the multitask dataset
    #     datasets_dict = {'SA': ds_micro_zinc_SA, 'logp': ds_micro_zinc_logp, 'score': ds_micro_zinc_score}
    #     multitask_microzinc = MultitaskDGLDataset(datasets_dict) # Can optionally have features

    #     multitask_microzinc.print_this()
    #     pprint(df)

    #     # The multitask dataset has as many molecules as there are unique smiles across the single task datasets.
    #     self.assertEqual(total_data_points, multitask_microzinc.__len__())

    # # TODO (Gabriela): After fixing case 3, implement case with bad smiles.
    # def test_multitask_dataset_case_4(self):
    #     """Case: Different tasks, semi-intersection, some bad smiles
    #         - For each task, add a few random smiles that won’t work, such as “XYZ” or simply “X”
    #         - Check that the smiles are filtered, alongside their label (ensure the right label has been filtered)
    #     """
    #     pass


if __name__ == "__main__":
    ut.main()
