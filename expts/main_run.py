# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
# from ogb.lsc import PCQM4MDataset



# Current project imports
import goli
from goli.config._loader import load_datamodule, load_metrics, load_architecture, load_predictor, load_trainer


# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
# CONFIG_FILE = "expts/config_molPCBA.yaml"
# CONFIG_FILE = "expts/config_molHIV.yaml"
# CONFIG_FILE = "expts/config_molPCQM4M.yaml"
# CONFIG_FILE = "expts/config_micro_ZINC.yaml"
CONFIG_FILE = "expts/config_micro-PCBA.yaml"
# CONFIG_FILE = "expts/config_ZINC_bench_gnn.yaml"
os.chdir(MAIN_DIR)


def main(cfg: DictConfig) -> None:
    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)
    print("\ndatamodule:\n", datamodule, "\n")

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dim_nodes=datamodule.num_node_feats_with_positional_encoding,
        in_dim_edges=datamodule.num_edge_feats,
    )

    metrics = load_metrics(cfg)
    print(metrics)

    predictor = load_predictor(cfg, model_class, model_kwargs, metrics)

    print(predictor.model)
    print(predictor.summarize(max_depth=4))

    trainer = load_trainer(cfg)

    # Run the model training
    print("\n------------ TRAINING STARTED ------------")
    try:
        trainer.fit(model=predictor, datamodule=datamodule)
        print("\n------------ TRAINING COMPLETED ------------\n\n")

    except Exception as e:
        if not cfg["constants"]["raise_train_error"]:
            print("\n------------ TRAINING ERROR: ------------\n\n", e)
        else:
            raise e

    print("\n------------ TESTING STARTED ------------")
    try:
        ckpt_path = trainer.checkpoint_callbacks[0].best_model_path
        trainer.test(model=predictor, datamodule=datamodule, ckpt_path=ckpt_path)
        print("\n------------ TESTING COMPLETED ------------\n\n")

    except Exception as e:
        if not cfg["constants"]["raise_train_error"]:
            print("\n------------ TESTING ERROR: ------------\n\n", e)
        else:
            raise e


if __name__ == "__main__":
    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
