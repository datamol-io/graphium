# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
import poptorch

# Current project imports
import goli
from goli.config._loader import load_datamodule, load_metrics, load_metrics_mtl, load_architecture, load_predictor, load_trainer

import faulthandler

faulthandler.enable()

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
# CONFIG_FILE = "tests/mtl/config_micro_ZINC_mtl_test.yaml"
CONFIG_FILE = "tests/mtl/config_ipu_test.yaml"
os.chdir(MAIN_DIR)

#! adding IPU options here
ipu_options = poptorch.Options()
ipu_options.deviceIterations(1) #not sure how to set this number yet, start small
ipu_options.replicationFactor(1)  #use 1 IPU for now in testing
ipu_options.Jit.traceModel(False)
ipu_options._jit._values["trace_model"] = False

def main(cfg: DictConfig) -> None:

    #! need to define the IPU options
    #? where is the best place to put this
    # this is required for trainer and the data module
    # I put it in config/_loader.py first


    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg, ipu_options=ipu_options )

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dim_nodes=datamodule.num_node_feats_with_positional_encoding,
        in_dim_edges=datamodule.num_edge_feats,
    )

    metrics = load_metrics_mtl(cfg)
    print(metrics)

    predictor = load_predictor(cfg, model_class, model_kwargs, metrics)

    print(predictor.model)
    print(predictor.summarize(max_depth=4))

    trainer = load_trainer(cfg, ipu_options=ipu_options)

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
