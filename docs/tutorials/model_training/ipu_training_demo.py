# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
import timeit
from loguru import logger
from pytorch_lightning.utilities.model_summary import ModelSummary

# Current project imports
import goli
from goli.config._loader import load_datamodule, load_metrics, load_architecture, load_predictor, load_trainer, get_max_num_nodes_edges_datamodule
from goli.utils.safe_run import SafeRun

# from torch_geometric.nn.aggr import Aggregation
# Aggregation.set_validate_args(False)

# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
# CONFIG_FILE= "expts/configs/config_ipu_allsizes.yaml"
CONFIG_FILE = "docs/tutorials/model_training/config_ipu_tutorials.yaml"
os.chdir(MAIN_DIR)

with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
    cfg = yaml.safe_load(f)

# Load and initialize the dataset
datamodule = load_datamodule(cfg)
datamodule.prepare_data()
datamodule.setup(stage=None)
max_nodes, max_edges = get_max_num_nodes_edges_datamodule(datamodule, stages=["train", "val"])

print(f'datamodule type {type(datamodule)}')

# Initialize the network
model_class, model_kwargs = load_architecture(
    cfg,
    in_dims=datamodule.in_dims,
)

print(f'model class {model_class}')
print(f'model args {model_kwargs}')

metrics = load_metrics(cfg)
logger.info(metrics)

predictor = load_predictor(cfg, model_class, model_kwargs, metrics)
predictor.model.set_max_num_nodes_edges_per_graph(max_nodes, max_edges)

logger.info(predictor.model)
logger.info(ModelSummary(predictor, max_depth=4))

trainer = load_trainer(cfg, "tutorial-run")

print(trainer)

# Run the model training
with SafeRun(name="TRAINING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
    trainer.fit(model=predictor, datamodule=datamodule)

# Exit WandB
wandb.finish()