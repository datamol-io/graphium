# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
from lightning.pytorch.utilities.model_summary import ModelSummary

# Current project imports
import graphium
from graphium.config._loader import load_datamodule, load_metrics, load_trainer

from graphium.trainer.predictor import PredictorModule


# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(graphium.__file__)))
os.chdir(MAIN_DIR)

MODEL_FILE = "models_checkpoints/ogb-molpcba/model-v2.ckpt"

CONFIG_FILE = "expts/config_molPCBA.yaml"


def main(cfg: DictConfig) -> None:
    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)
    print("\ndatamodule:\n", datamodule, "\n")

    metrics = load_metrics(cfg)
    print(metrics)

    predictor = PredictorModule.load_from_checkpoint(MODEL_FILE)
    predictor.metrics = metrics

    print(predictor.model)
    print(ModelSummary(predictor, max_depth=4))

    trainer = load_trainer(cfg)

    # Run the model testing
    trainer.test(model=predictor, datamodule=datamodule, ckpt_path=MODEL_FILE)


if __name__ == "__main__":
    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
