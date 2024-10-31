from typing import Any, List, Dict, Literal, Union

import os

import torch

import pandas as pd
import numpy as np

from pytorch_lightning import LightningDataModule

from torch.utils.data import Dataset, DataLoader

from graphium.data.datamodule import MultitaskFromSmilesDataModule, ADMETBenchmarkDataModule
from graphium.trainer.predictor import PredictorModule
from graphium.fingerprinting.fingerprinter import Fingerprinter


class FingerprintDataset(Dataset):
    def __init__(
            self,
            smiles: List[str],
            labels: torch.Tensor,
            fingerprints: Dict[str, torch.Tensor],
    ):
        self.smiles = smiles
        self.labels = labels
        self.fingerprints = fingerprints

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        fp_list = []
        for val in self.fingerprints.values():
            fp_list.append(val[index])
        return fp_list, self.labels[index]


class FingerprintDatamodule(LightningDataModule):
    def __init__(
        self,
        pretrained_models: Dict[str, List[str]],
        task: str = "herg",
        benchmark: Literal["tdc", None] = "tdc",
        df_path: str = None,
        batch_size: int = 64,
        split_type: str = "random",
        splits_path: str = None,
        split_val: float = 0.1,
        split_test: float = 0.1,
        data_seed: int = 42,
        num_workers: int = 0,
        device: str = "cpu",
        mol_cache_dir: str = "./expts/data/cache",
        fps_cache_dir: str = "./expts/data/cache",
    ):
        super().__init__()

        assert benchmark is not None or df_path is not None, "Either benchmark or df_path must be provided"

        self.pretrained_models = pretrained_models
        self.task = task
        self.benchmark = benchmark
        self.df_path = df_path
        self.batch_size = batch_size
        self.split_type = split_type
        self.splits_path = splits_path
        self.split_val = split_val
        self.split_test = split_test
        self.data_seed = data_seed
        self.num_workers = num_workers
        self.device = device
        self.mol_cache_dir = mol_cache_dir
        self.fps_cache_dir = fps_cache_dir
        if benchmark is not None:
            # Check if benchmark naming is already implied in config
            if f"{benchmark}/{task}" not in mol_cache_dir:
                self.mol_cache_dir = f"{mol_cache_dir}/{benchmark}/{task}"
            if f"{benchmark}/{task}" not in fps_cache_dir:
                self.fps_cache_dir = f"{fps_cache_dir}/{benchmark}/{task}"

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.splits = []

    def prepare_data(self) -> None:  
        if self.fps_cache_dir is not None and os.path.exists(f"{self.fps_cache_dir}/fps.pt"):
            self.smiles, self.labels, self.fps_dict = torch.load(f"{self.fps_cache_dir}/fps.pt").values()
            self.splits = list(self.smiles.keys())
        
        else:
            # Check which splits are needed
            self.splits = []
            add_all = self.benchmark is not None or self.splits_path is not None
            if add_all or self.split_val + self.split_test < 1:
                self.splits.append("train")
            if add_all or self.split_val > 0:
                self.splits.append("valid")
            if add_all or self.split_test > 0:
                self.splits.append("test")

            self.data = {
                split: {
                    "smiles": [],
                    "labels": [],
                    "fps": {},
                }
                for split in self.splits
            }

            for model, layers in self.pretrained_models.items():
                predictor = PredictorModule.load_pretrained_model(model, device=self.device)
                predictor.featurization.pop("max_num_atoms", None)

                # Featurization
                if self.benchmark is None:
                    assert self.df_path is not None, "df_path must be provided if not using an integrated benchmark"

                    # Add a dummy task column (filled with NaN values) in case no such column is provided
                    smiles_df = pd.read_csv(self.df_path)
                    task_cols = [col for col in smiles_df if col.startswith("task_")]
                    if len(task_cols) == 0:
                        df_path = ".".join(self.df_path.split(".")[:-1])
                        smiles_df["task_dummy"] = np.nan
                        smiles_df.to_csv(f"{df_path}_with_dummy_task_col.csv", index=False)
                        self.df_path = f"{df_path}_with_dummy_task_col.csv"

                    task_specific_args = {
                        "fingerprinting": {
                            "df_path": self.df_path,
                            "smiles_col": "smiles",
                            "label_cols": "task_*",
                            "task_level": "graph",
                            "splits_path": self.splits_path,
                            "split_type": self.split_type,
                            "split_val": self.split_val,
                            "split_test": self.split_test,
                            "seed": self.data_seed,
                        }
                    }
                    label_key = "graph_fingerprinting"

                    datamodule = MultitaskFromSmilesDataModule(
                        task_specific_args=task_specific_args,
                        batch_size_inference=128,
                        featurization=predictor.featurization,
                        featurization_n_jobs=0,
                        processed_graph_data_path=f"{self.mol_cache_dir}/mols/",
                    )

                elif self.benchmark == "tdc":
                    datamodule = ADMETBenchmarkDataModule(
                        tdc_benchmark_names=[self.task],
                        tdc_train_val_seed=self.data_seed,
                        batch_size_inference=128,
                        featurization=predictor.featurization,
                        featurization_n_jobs=self.num_workers,
                        processed_graph_data_path=f"{self.mol_cache_dir}/mols/",
                    )
                    label_key = f"graph_{self.task}"
                
                else:
                    raise ValueError(f"Invalid benchmark: {self.benchmark}")

                datamodule.prepare_data()
                datamodule.setup()

                loader_dict = {}
                if "train" in self.splits:
                    datamodule.train_ds.return_smiles = True
                    loader_dict["train"] = datamodule.get_dataloader(datamodule.train_ds, shuffle=False, stage="predict")
                if "valid" in self.splits:
                    datamodule.val_ds.return_smiles = True
                    loader_dict["valid"] = datamodule.get_dataloader(datamodule.val_ds, shuffle=False, stage="predict")
                if "test" in self.splits:
                    datamodule.test_ds.return_smiles = True
                    loader_dict["test"] = datamodule.get_dataloader(datamodule.test_ds, shuffle=False, stage="predict")

                for split, loader in loader_dict.items():
                    if len(self.data[split]["smiles"]) == 0:
                        for batch in loader:
                            self.data[split]["smiles"] += [item for item in batch["smiles"]]
                            self.data[split]["labels"] += batch["labels"][label_key]

                    with Fingerprinter(predictor, layers, out_type="torch") as fp:
                        fps = fp.get_fingerprints_for_dataset(loader, store_dict=True)
                        for fp_name, fp in fps.items():
                            self.data[split]["fps"][f"{model}/{fp_name}"] = fp

            os.makedirs(self.fps_cache_dir, exist_ok=True)
            torch.save(self.data, f"{self.fps_cache_dir}/fps.pt")

    def setup(self, stage: str) -> None:
        # Creating datasets
        if stage == "fit":
            self.train_dataset = FingerprintDataset(self.smiles["train"], self.labels["train"], self.fps_dict["train"])
            self.valid_dataset = FingerprintDataset(self.smiles["valid"], self.labels["valid"], self.fps_dict["valid"])
        else:
            self.test_dataset = FingerprintDataset(self.smiles["test"], self.labels["test"], self.fps_dict["test"])

    def get_fp_dims(self):
        fp_dict = next(iter(self.fps_dict.values()))

        return [fp.size(1) for fp in fp_dict.values()]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset), shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)