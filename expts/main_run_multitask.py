# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
from functools import partial
from datetime import date

# Current project imports
import goli
from goli.config._loader import load_datamodule, load_metrics, load_architecture, load_predictor, load_trainer

# Optuna
import optuna
from optuna.trial import TrialState

# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
# CONFIG_FILE = "tests/mtl/config_micro_ZINC_mtl_test_3_tasks_pyg.yaml"
#CONFIG_FILE = "tests/mtl/config_ipu_9atoms.yaml"
CONFIG_FILE = "tests/mtl/config_ipu_allsizes.yaml"
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


def main(cfg: DictConfig, trial, run_name="main") -> None:

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

    trainer = load_trainer(cfg, run_name)

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

    return trainer.callback_metrics["cv/mae/test"].cpu().item()

if __name__ == "__main__":
    #nan_checker("goli/data/QM9/micro_qm9.csv") #can be deleted
    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)

    def objective(trial, cfg):
        cfg = deepcopy(cfg)

        # * Example of adding hyper parameter search with Optuna:
        # * https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
        cfg["architecture"]["gnn"]["hidden_dims"] = trial.suggest_int("gnn.hid", 16, 124, 16)
        cfg["architecture"]["gnn"]["depth"] = trial.suggest_int("gnn.depth", 1, 5)
        # normalization = trial.suggest_categorical("batch_norm", ["none", "batch_norm", "layer_norm"])
        # cfg["architecture"]["gnn"]["normalization"] = normalization
        # cfg["architecture"]["pre_nn"]["normalization"] = normalization
        # cfg["architecture"]["post_nn"]["normalization"] = normalization

        run_name = 'no_name_' if not "name" in cfg["constants"] else cfg["constants"]["name"] + "_"
        run_name = run_name + date.today().strftime("%d/%m/%Y") + "_"
        for key, value in trial.params.items():
            run_name = run_name + str(key) + "=" + str(value) + "_"

        accu = main(cfg, trial, run_name=run_name[:len(run_name) - 1])
        wandb.finish()  
        return accu
    
    study = optuna.create_study()
    study.optimize(partial(objective, cfg=cfg), n_trials=1, timeout=600)

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
