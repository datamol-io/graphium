# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig


# Current project imports
import goli
from goli.config._loader import load_datamodule, load_metrics, load_architecture, load_predictor, load_trainer


# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
os.chdir(MAIN_DIR)


def main(cfg: DictConfig) -> None:
    cfg = dict(deepcopy(cfg))

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)
    print("\ndatamodule:\n", datamodule, "\n")

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dim_nodes=datamodule.num_node_feats,
        in_dim_edges=datamodule.num_edge_feats,
    )

    metrics = load_metrics(cfg)
    print(metrics)

    predictor = load_predictor(cfg, model_class, model_kwargs, metrics)

    print(predictor.model)
    print(predictor.summarize(mode=4, to_print=False))

    trainer = load_trainer(cfg, metrics)

    # Run the model training
    try:
        trainer.fit(model=predictor, datamodule=datamodule)
        print("\n------------ TRAINING COMPLETED ------------\n\n")
    except Exception as e:
        if not cfg["constants"]["raise_train_error"]:
            print("\n------------ TRAINING ERROR: ------------\n\n", e)
        else:
            raise


if __name__ == "__main__":
    with open(os.path.join(MAIN_DIR, "expts/config_molHIV.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
