# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.utilities.model_summary import ModelSummary

# Current project imports
import graphium
from graphium.config._loader import load_datamodule, load_trainer
from graphium.utils.fs import mkdir
from graphium.trainer.predictor import PredictorModule


# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(graphium.__file__)))
os.chdir(MAIN_DIR)

DATA_NAME = "molhiv"
MODEL_FILE = "models_checkpoints/ogb-molpcba/model-v2.ckpt"
CONFIG_FILE = f"expts/config_{DATA_NAME}_pretrained.yaml"

# MODEL_FILE = "models_checkpoints/micro_ZINC/model.ckpt"
# CONFIG_FILE = "expts/config_micro_ZINC.yaml"


def main(cfg: DictConfig) -> None:
    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)
    print("\ndatamodule:\n", datamodule, "\n")

    export_df_path = f"predictions/preds-{DATA_NAME}.csv.gz"

    predictor = PredictorModule.load_from_checkpoint(MODEL_FILE)

    print(predictor.model)
    print(ModelSummary(predictor, max_depth=4))

    trainer = load_trainer(cfg)

    # Run the model prediction
    preds = trainer.predict(model=predictor, datamodule=datamodule)
    if isinstance(preds[0], torch.Tensor):
        preds = [p.detach().cpu().numpy() for p in preds]
    preds = np.concatenate(preds, axis=0)

    # Generate output dataframe
    df = {"SMILES": datamodule.dataset.smiles}

    target = datamodule.dataset.labels
    for ii in range(target.shape[1]):
        df[f"Target-{ii}"] = target[:, ii]

    for ii in range(preds.shape[1]):
        df[f"Preds-{ii}"] = preds[:, ii]
    df = pd.DataFrame(df)
    mkdir("predictions")
    df.to_csv(export_df_path)
    print(df)
    print(f"file saved to:`{export_df_path}`")


if __name__ == "__main__":
    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
