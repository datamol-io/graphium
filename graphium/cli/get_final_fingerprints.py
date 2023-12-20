from typing import List, Literal, Union
import os
import time
import timeit
from datetime import datetime

import fsspec
import hydra
import numpy as np
import torch
import wandb
import yaml
from datamol.utils import fs
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from lightning.pytorch.utilities.model_summary import ModelSummary
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from graphium.config._loader import (
    load_accelerator,
    load_architecture,
    load_datamodule,
    load_metrics,
    load_predictor,
    load_trainer,
    save_params_to_wandb,
    get_checkpoint_path,
)
from graphium.finetuning import (
    FINETUNING_CONFIG_KEY,
    GraphFinetuning,
    modify_cfg_for_finetuning,
)
from graphium.hyper_param_search import (
    HYPER_PARAM_SEARCH_CONFIG_KEY,
    extract_main_metric_for_hparam_search,
)
from graphium.trainer.predictor import PredictorModule
from graphium.utils.safe_run import SafeRun

import graphium.cli.finetune_utils

from tqdm import tqdm
from copy import deepcopy
from tdc.benchmark_group import admet_group
import datamol as dm
import sys
from torch_geometric.data import Batch
import random

TESTING_ONLY_CONFIG_KEY = "testing_only"


@hydra.main(version_base=None, config_path="../../expts/hydra-configs", config_name="main")
def cli(cfg: DictConfig) -> None:
    """
    The main CLI endpoint for training, fine-tuning and evaluating Graphium models.
    """
    return get_final_fingerprints(cfg)


def get_final_fingerprints(cfg: DictConfig) -> None:
    """
    The main (pre-)training and fine-tuning loop.
    """

    # Get ADMET SMILES strings
    
    if not os.path.exists("saved_admet_smiles.pt"):
        admet = admet_group(path="admet-data/")
        admet_mol_ids = set()
        for dn in tqdm(admet.dataset_names, desc="Getting IDs for ADMET", file=sys.stdout):
            admet_mol_ids |= set(admet.get(dn)["train_val"]["Drug"].apply(dm.unique_id))
            admet_mol_ids |= set(admet.get(dn)["test"]["Drug"].apply(dm.unique_id))

        smiles_to_process = []
        admet_mol_ids_to_find = deepcopy(admet_mol_ids)

        for dn in tqdm(admet.dataset_names, desc="Matching molecules to IDs", file=sys.stdout):
            for key in ["train_val", "test"]:
                train_mols = set(admet.get(dn)[key]["Drug"])
                for smiles in train_mols:
                    mol_id = dm.unique_id(smiles)
                    if mol_id in admet_mol_ids_to_find:
                        smiles_to_process.append(smiles)
                        admet_mol_ids_to_find.remove(mol_id)

        assert set(dm.unique_id(s) for s in smiles_to_process) == admet_mol_ids
        torch.save(smiles_to_process, "saved_admet_smiles.pt")
    else:
        smiles_to_process = torch.load("saved_admet_smiles.pt")

    unresolved_cfg = OmegaConf.to_container(cfg, resolve=False)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    st = timeit.default_timer()

    ## == Instantiate all required objects from their respective configs ==
    # Accelerator
    cfg, accelerator_type = load_accelerator(cfg)
    assert accelerator_type == "cpu", "get_final_fingerprints script only runs on CPU for now"

    ## Data-module
    datamodule = load_datamodule(cfg, accelerator_type)


    # Featurize SMILES strings
    
    input_features_save_path = "input_features.pt"
    idx_none_save_path = "idx_none.pt"
    if not os.path.exists(input_features_save_path):
        input_features, idx_none = datamodule._featurize_molecules(smiles_to_process)

        torch.save(input_features, input_features_save_path)
        torch.save(idx_none, idx_none_save_path)
    else:
        input_features = torch.load(input_features_save_path)

    '''
    for _ in range(100):

        index = random.randint(0, len(smiles_to_process) - 1)
        features_single, idx_none_single = datamodule._featurize_molecules([smiles_to_process[index]])

        def _single_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, torch.Tensor):
                return val.all()
            raise ValueError(f"Type {type(val)} not accounted for")

        assert all(_single_bool(features_single[0][k] == input_features[index][k]) for k in features_single[0].keys())

    import sys; sys.exit(0)
    '''

    failures = 0

    # Cast to FP32
    
    for input_feature in tqdm(input_features, desc="Casting to FP32"):
        try:
            if not isinstance(input_feature, str):
                for k, v in input_feature.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype == torch.half:
                            input_feature[k] = v.float()
                        elif v.dtype == torch.int32:
                            input_feature[k] = v.long()
            else:
                failures += 1
        except Exception as e:
            print(f"{input_feature = }")
            raise e

    print(f"{failures = }")
                    

    # Load pre-trained model
    predictor = PredictorModule.load_pretrained_model(
        name_or_path=get_checkpoint_path(cfg), device=accelerator_type
    )

    logger.info(predictor.model)
    logger.info(ModelSummary(predictor, max_depth=4))

    batch_size = 100

    results = []


    # Run the model to get fingerprints
    
    for index in tqdm(range(0, len(input_features), batch_size)):
        batch = Batch.from_data_list(input_features[index:(index + batch_size)])
        model_fp32 = predictor.model.float()
        output, extras = model_fp32.forward(batch, extra_return_names=["pre_task_heads"])
        fingerprint = extras['pre_task_heads']['graph_feat']
        results.extend([fingerprint[i] for i in range(batch_size)])

        if index == 0:
            print(fingerprint.shape)

    torch.save(results, "results.pt")

    # Generate dictionary SMILES -> fingerprint vector
    smiles_to_fingerprint = dict(zip(smiles_to_process, results))
    torch.save(smiles_to_fingerprint, "smiles_to_fingerprint.pt")

    # Generate dictionary unique IDs -> fingerprint vector
    ids = [dm.unique_id(smiles) for smiles in smiles_to_process]
    ids_to_fingerprint = dict(zip(ids, results))
    torch.save(ids_to_fingerprint, "ids_to_fingerprint.pt")
    

if __name__ == "__main__":
    cli()
