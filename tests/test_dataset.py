import unittest as ut

from graphium.data import load_micro_zinc
from graphium.data.dataset import SingleTaskDataset, MultitaskDataset
from graphium.data.smiles_transform import smiles_to_unique_mol_ids
from graphium.data.utils import get_keys


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

        # We need to turn these dataframes into single-task datasets.
        # We don't need to do featurization yet.
        ds_micro_zinc_SA = SingleTaskDataset(
            smiles=df_micro_zinc_SA.loc[:, "SMILES"].tolist(), labels=df_micro_zinc_SA.loc[:, "SA"].tolist()
        )

        ds_micro_zinc_logp = SingleTaskDataset(
            smiles=df_micro_zinc_logp.loc[:, "SMILES"].tolist(),
            labels=df_micro_zinc_logp.loc[:, "logp"].tolist(),
        )
        ds_micro_zinc_score = SingleTaskDataset(
            smiles=df_micro_zinc_score.loc[:, "SMILES"].tolist(),
            labels=df_micro_zinc_score.loc[:, "score"].tolist(),
        )

        # Create the multitask dataset
        datasets_dict = {"SA": ds_micro_zinc_SA, "logp": ds_micro_zinc_logp, "score": ds_micro_zinc_score}
        multitask_microzinc = MultitaskDataset(
            datasets_dict, save_smiles_and_ids=True
        )  # Can optionally have features

        # Check: The number of unique molecules equals the number of datapoints in the multitask dataset.
        self.assertEqual(num_unique_mols, multitask_microzinc.__len__())

        # Check that for each task, you have the same label values as the initial DF.
        for idx in range(multitask_microzinc.__len__()):
            smiles = df[["SMILES"]].iloc[idx].values[0]
            # label = df[['SA']].iloc[idx]
            label_SA = ds_micro_zinc_SA.labels[idx]
            label_logp = ds_micro_zinc_logp.labels[idx]
            label_score = ds_micro_zinc_score.labels[idx]

            # Search for the mol id in the multitask dataset
            mol_ids = smiles_to_unique_mol_ids([smiles])
            mol_id = mol_ids[0]
            found_idx = -1
            for i, id in enumerate(multitask_microzinc.mol_ids):
                if mol_id == id:
                    found_idx = i

            # Compare labels
            self.assertEqual(label_SA, multitask_microzinc.labels[found_idx]["SA"])
            self.assertEqual(label_logp, multitask_microzinc.labels[found_idx]["logp"])
            self.assertEqual(label_score, multitask_microzinc.labels[found_idx]["score"])

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

        # Here we split the data according to the task we care about.
        df_micro_zinc_SA = df_rows_SA[["SMILES", "SA"]]
        df_micro_zinc_logp = df_rows_logp[["SMILES", "logp"]]
        df_micro_zinc_score = df_rows_score[["SMILES", "score"]]

        # We need to turn these dataframes into single-task datasets.
        # We don't need to do featurization yet.
        ds_micro_zinc_SA = SingleTaskDataset(
            smiles=df_micro_zinc_SA.loc[:, "SMILES"].tolist(), labels=df_micro_zinc_SA.loc[:, "SA"].tolist()
        )
        ds_micro_zinc_logp = SingleTaskDataset(
            smiles=df_micro_zinc_logp.loc[:, "SMILES"].tolist(),
            labels=df_micro_zinc_logp.loc[:, "logp"].tolist(),
        )
        ds_micro_zinc_score = SingleTaskDataset(
            smiles=df_micro_zinc_score.loc[:, "SMILES"].tolist(),
            labels=df_micro_zinc_score.loc[:, "score"].tolist(),
        )

        # Create the multitask dataset
        datasets_dict = {"SA": ds_micro_zinc_SA, "logp": ds_micro_zinc_logp, "score": ds_micro_zinc_score}
        multitask_microzinc = MultitaskDataset(
            datasets_dict, save_smiles_and_ids=True
        )  # Can optionally have features

        # The total dataset has as many molecules as there are smiles in all tasks put together
        self.assertEqual(total_data_points, multitask_microzinc.__len__())

        # For each task, only the smiles related to that task have values, and the value is what's expected from the initial DF.
        for idx in range(len(ds_micro_zinc_SA)):
            smiles = df[["SMILES"]].iloc[idx].values[0]

            task = "task"
            if idx in range(0, 200):
                task = "SA"
            elif idx in range(200, 400):
                task = "logp"
            elif idx in range(400, 750):
                task = "score"

            # Labels of that molecule
            label_SA = df[["SA"]].iloc[idx].values[0]
            label_logp = df[["logp"]].iloc[idx].values[0]
            label_score = df[["score"]].iloc[idx].values[0]

            # Search for that molecule in the multitask dataset
            mol_ids = smiles_to_unique_mol_ids([smiles])
            mol_id = mol_ids[0]
            found_idx = -1
            for i, id in enumerate(multitask_microzinc.mol_ids):
                if mol_id == id:
                    found_idx = i
            multitask_microzinc_labels = get_keys(multitask_microzinc.labels[found_idx])
            if task == "SA":
                self.assertEqual(label_SA, multitask_microzinc.labels[found_idx]["SA"])
                self.assertFalse("score" in multitask_microzinc_labels)
                self.assertFalse("logp" in multitask_microzinc_labels)
            elif task == "logp":
                self.assertEqual(label_logp, multitask_microzinc.labels[found_idx]["logp"])
                self.assertFalse("score" in multitask_microzinc_labels)
                self.assertFalse("SA" in multitask_microzinc_labels)
            elif task == "score":
                self.assertEqual(label_score, multitask_microzinc.labels[found_idx]["score"])
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

        # Here we split the data according to the task we care about.
        df_micro_zinc_SA = df_rows_SA[["SMILES", "SA"]]
        df_micro_zinc_logp = df_rows_logp[["SMILES", "logp"]]
        df_micro_zinc_score = df_rows_score[["SMILES", "score"]]

        # We need to turn these dataframes into single-task datasets.
        # We don't need to do featurization yet.
        ds_micro_zinc_SA = SingleTaskDataset(
            smiles=df_micro_zinc_SA.loc[:, "SMILES"].tolist(), labels=df_micro_zinc_SA.loc[:, "SA"].tolist()
        )
        ds_micro_zinc_logp = SingleTaskDataset(
            smiles=df_micro_zinc_logp.loc[:, "SMILES"].tolist(),
            labels=df_micro_zinc_logp.loc[:, "logp"].tolist(),
        )
        ds_micro_zinc_score = SingleTaskDataset(
            smiles=df_micro_zinc_score.loc[:, "SMILES"].tolist(),
            labels=df_micro_zinc_score.loc[:, "score"].tolist(),
        )

        # Create the multitask dataset
        datasets_dict = {"SA": ds_micro_zinc_SA, "logp": ds_micro_zinc_logp, "score": ds_micro_zinc_score}
        multitask_microzinc = MultitaskDataset(
            datasets_dict, save_smiles_and_ids=True
        )  # Can optionally have features

        # The multitask dataset has as many molecules as there are unique smiles across the single task datasets.
        self.assertEqual(total_data_points, multitask_microzinc.__len__())


if __name__ == "__main__":
    ut.main()
