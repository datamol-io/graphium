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
# CONFIG_FILE = "tests/mtl/config_micro_ZINC_mtl_test_3_tasks_pyg.yaml"
#CONFIG_FILE = "tests/mtl/config_ipu_9atoms.yaml"
CONFIG_FILE = "tests/mtl/config_ipu_allsizes.yaml"
# CONFIG_FILE = "tests/mtl/config_ipu_reproduce.yaml"
os.chdir(MAIN_DIR)

# '''
# Andy's helper function
# check if there is any missing values in a csv
# '''
# def nan_checker(fname):
#     with open(fname) as f:
#         lines = f.readlines()
#         prevsize = 0
#         for line in lines:
#             txts = line.split(",")
#             if (len(txts) > prevsize):
#                 prevsize = len(txts)
#             if (len(txts) < prevsize):
#                 print ("missing entry")
#                 print (line)

#             if (len(txts) != 21):
#                 print ("missing entry")


def main(cfg: DictConfig) -> None:

    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)

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

    datamodule.prepare_data()
    trainer.fit(model=predictor, datamodule=datamodule)
    # Run the model training
    print("\n------------ TRAINING STARTED ------------")
    try:
        print("\n------------ TRAINING COMPLETED ------------\n\n")

    except Exception as e:
        if not cfg["constants"]["raise_train_error"]:
            print("\n------------ TRAINING ERROR: ------------\n\n", e)
        else:
            raise e

    print("\n------------ TESTING STARTED ------------")
    try:
        ckpt_path = trainer.checkpoint_callbacks[0].best_model_path
        #ckpt_path = "models_checkpoints/micro_ZINC_mtl/model-v3.ckpt"
        #error here
        #TypeError: iteration over a 0-d tensor
        trainer.test(model=predictor, datamodule=datamodule) #, ckpt_path=ckpt_path)
        print("\n------------ TESTING COMPLETED ------------\n\n")

    except Exception as e:
        if not cfg["constants"]["raise_train_error"]:
            print("\n------------ TESTING ERROR: ------------\n\n", e)
        else:
            raise e


if __name__ == "__main__":
    #nan_checker("goli/data/QM9/micro_qm9.csv") #can be deleted
    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
