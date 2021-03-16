# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
import hydra
from omegaconf import DictConfig


# Current project imports
import goli
from goli.commons.config_loader import (
    config_load_constants,
    config_load_dataset,
    config_load_architecture,
    config_load_metrics,
    config_load_predictor,
    config_load_training,
)


# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
os.chdir(MAIN_DIR)


# @hydra.main(config_name="config_micro_ZINC.yaml")
def main(cfg: DictConfig) -> None:
    cfg = dict(deepcopy(cfg))

    # Get the general parameters and generate the train/val/test datasets
    data_device, model_device, dtype, exp_name, seed, raise_train_error = config_load_constants(
        **cfg["constants"], main_dir=MAIN_DIR
    )

    # Load and initialize the dataset
    datamodule, num_node_feats, num_edge_feats = config_load_dataset(
        **cfg["datasets"],
        main_dir=MAIN_DIR,
        data_device=data_device,
        model_device=model_device,
        seed=seed,
        dtype=dtype,
    )
    print("\ndatamodule:\n", datamodule, "\n")

    # Initialize the network
    model = config_load_architecture(
        **cfg["architecture"],
        in_dim_nodes=num_node_feats,
        in_dim_edges=num_edge_feats,
        model_device=model_device,
        dtype=dtype,
    )

    print("\nmodel:\n", model, "\n")
    pass

    # metrics, metrics_on_progress_bar = config_load_metrics(cfg["metrics"])
    # predictor = config_load_predictor(
    #     cfg["predictor"],
    #     metrics,
    #     metrics_on_progress_bar,
    #     model,
    #     layer_name,
    #     train_dt,
    #     val_dt,
    #     device,
    #     dtype,
    # )
    # trainer = config_load_training(cfg["training"], predictor)

    # # Run the model training
    # try:
    #     trainer.fit(predictor)
    #     print("\n------------ TRAINING COMPLETED ------------\n\n")
    # except Exception as e:
    #     if not cfg["constants"]["raise_train_error"]:
    #         print("\n------------ TRAINING ERROR: ------------\n\n", e)
    #     else:
    #         raise


if __name__ == "__main__":
    with open(os.path.join(MAIN_DIR, "goli/expts/config_micro_ZINC.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
