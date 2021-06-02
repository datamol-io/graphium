# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
import numpy as np
import pandas as pd

# Current project imports
import goli
from goli.config._loader import load_datamodule, load_trainer
from goli.utils.fs import mkdir
from goli.trainer.predictor import PredictorModule


# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
os.chdir(MAIN_DIR)

MODEL_FILE = "models_checkpoints/micro_ZINC/model.ckpt"
CONFIG_FILE = "expts/config_micro_ZINC.yaml"


SKIP_OUTPUT_LAYERS = 1
EXPORT_DATAFRAME_PATH = f"predictions/fingerprint-skip-output-{SKIP_OUTPUT_LAYERS}.csv"


def main(cfg: DictConfig) -> None:

    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)
    print("\ndatamodule:\n", datamodule, "\n")

    predictor = PredictorModule.load_from_checkpoint(MODEL_FILE)
    predictor.metrics = {}
    predictor.skip_output_layers = SKIP_OUTPUT_LAYERS

    print(predictor.model)
    print(predictor.summarize(mode=4, to_print=False))

    trainer = load_trainer(cfg)

    # Run the model prediction
    preds = trainer.predict(model=predictor, datamodule=datamodule)
    preds = np.concatenate(preds, axis=0)

    # Generate output dataframe
    df = {"SMILES": datamodule.dataset.smiles}

    target = datamodule.dataset.labels
    for ii in range(target.shape[1]):
        df[f"target-{ii}"] = target[:, ii]

    for ii in range(preds.shape[1]):
        df[f"Predictions-{ii}"] = preds[:, ii]
    df = pd.DataFrame(df)
    mkdir("predictions")
    df.to_csv(EXPORT_DATAFRAME_PATH)
    print(df)


if __name__ == "__main__":
    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
