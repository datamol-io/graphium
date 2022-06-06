import pathlib
import tempfile

import unittest as ut
import numpy as np
import torch
import pandas as pd

#import goli
from goli.data.datamodule import SingleTaskDataset, MultitaskDGLDataset
from goli.data import load_micro_zinc

# Try using self.assert
class Test_Multitask_DataModule(ut.TestCase):

    def test_multitask_dataset(self):
        # Make sure that the merging of datasets happens as expected
        # Try datasets which have a lot of overlap
        # Little overlap
        # Removing certain columns (i.e. when featurization has not been successful for those molecules)
        
        df = load_micro_zinc() # Has about 1000 molecules

        # Here we take the microzinc dataset and split the labels up into 'SA', 'logp' and 'score'
        # in order to simulate having multiple single-task datasets
        df_micro_zinc_SA = df[['SMILES', 'SA']]
        df_micro_zinc_logp = df[['SMILES', 'logp']]
        df_micro_zinc_score = df[['SMILES', 'score']]

        # We need to turn these dataframes into single-task DGL datasets
        # We don't need to do featurization yet
        ds_micro_zinc_SA = SingleTaskDataset(      # Doesn't run yet, "AttributeError", maybe because there are changes to DataModule right now that don't compile
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
        
        dataset_dict = {'SA': ds_micro_zinc_SA, 'logp': ds_micro_zinc_logp, 'score': ds_micro_zinc_score}
        multitask_microzinc = MultitaskDGLDataset(dataset_dict) # Can optionally have features

        # Test the MultitaskDGLDataset class itself, without the data module for now.
        smiles = multitask_microzinc.smiles
        self.assertEqual(len(smiles), len(df.loc[:,'SMILES']))
        self.assertEqual(smiles, df.loc[:'SMILES'])

        #for idx in multitask_microzinc.__len__():
        #    datapoint = multitask_microzinc.__getitem__(idx)
        #    smiles = datapoint["smiles"]
            # Check the values in the multitask dataset against the values of the original dataset
            # Re-write after making changes to multitask dataset

if __name__ == "__main__":
    ut.main()