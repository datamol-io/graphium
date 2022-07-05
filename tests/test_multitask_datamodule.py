import pathlib
import tempfile

import unittest as ut
import numpy as np
import torch
import pandas as pd


import goli
from goli.data import load_micro_zinc, DGLDataset, SingleTaskDataset, MultitaskDGLDataset
from goli.data.datamodule import smiles_to_unique_mol_ids

class Test_Multitask_Dataset(ut.TestCase):

    def test_multitask_dataset_case_1(self):
        """Case: different tasks, all with the same smiles set.
            - Check that for each task, all smiles are received from the initial DF.
            - Check that for each task, you have the same label values as the initial DF.
        """

        df = load_micro_zinc() # Has about 1000 molecules

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

        # Test the MultitaskDGLDataset class itself, without the data module for now.
        #smiles = multitask_microzinc.smiles
        #self.assertEqual(len(smiles), len(df.loc[:,'SMILES']))
        #self.assertEqual(smiles, df.loc[:'SMILES'])

        # Check that for each task, you have the same label values as the initial DF.
        # SA task
        for idx in range(len(ds_micro_zinc_SA)):
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

        # Using a data loader
        #multitask_loader = torch.utils.data.DataLoader(multitask_microzinc, shuffle=False, num_workers=0, batch_size=1)
        #self.assertEqual(len(smiles), len(multitask_loader))

        #for data in multitask_loader:
            # Check that for each task, you have the same label values as the initial DF.
            #data["unique_ids"]
            #print(data)

    def test_multitask_dataset_case_2(self):
        """Case: Different tasks, but with no intersection in the smiles (each task has a unique set of smiles)
            - Check that the total dataset has as much smiles as all tasks together
            - Check that, for each task, only the smiles related to that task have values, and ensure the value is what's expected from the initial DF
        """
        df = load_micro_zinc() # Has about 1000 molecules

        # Choose non-overlapping smiles by choosing specific rows from the original dataframe.
        df_rows_SA = df.iloc[0:200]         # 200 data points
        df_rows_logp = df.iloc[200:400]     # 200 data points
        df_rows_score = df.iloc[400:750]    # 350 data points
        total_data_points = 750

        # Here we split the data according to the task we care about.
        df_micro_zinc_SA = df_rows_SA[['SMILES', 'SA']]
        df_micro_zinc_logp = df_rows_logp[['SMILES', 'logp']]
        df_micro_zinc_score = df_rows_score[['SMILES', 'score']]

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

        # Create the multitask dataset
        datasets_dict = {'SA': ds_micro_zinc_SA, 'logp': ds_micro_zinc_logp, 'score': ds_micro_zinc_score}
        multitask_microzinc = MultitaskDGLDataset(datasets_dict) # Can optionally have features

        # The total dataset has as many molecules as there are smiles in all tasks put together
        self.assertEqual(total_data_points, multitask_microzinc.__len__())

        # For each task, only the smiles related to that task have values, and the value is what's expected from the initial DF.
        for idx in range(len(ds_micro_zinc_SA)):
            smiles = df[['SMILES']].iloc[idx].values[0]

            task = 'task'
            if idx in range(0, 200): task = 'SA'
            elif idx in range(200, 400): task = 'logp'
            elif idx in range(400, 750): task = 'score'

            # Labels of that molecule
            label_SA = df[['SA']].iloc[idx].values[0]
            label_logp = df[['logp']].iloc[idx].values[0]
            label_score = df[['score']].iloc[idx].values[0]

            # Search for that molecule in the multitask dataset
            mol_ids = smiles_to_unique_mol_ids([smiles])
            mol_id = mol_ids[0]
            found_idx = -1
            for i, id in enumerate(multitask_microzinc.mol_ids):
                if mol_id == id: found_idx = i

            if task == 'SA':
                self.assertEqual(label_SA, multitask_microzinc.labels[found_idx]['SA'])
                self.assertFalse('score' in multitask_microzinc.labels[found_idx].keys())
                self.assertFalse('logp' in multitask_microzinc.labels[found_idx].keys())




if __name__ == "__main__":
    ut.main()
