# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
from functools import partial
from datetime import date
import timeit
from loguru import logger

# Current project imports
import goli
from goli.config._loader import load_datamodule, load_metrics, load_architecture, load_predictor, load_trainer
from goli.utils.safe_run import SafeRun

# Optuna
import optuna
from optuna.trial import TrialState

# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
# CONFIG_FILE = "expts/configs/config_micro_ZINC_mtl_test_3_tasks_pyg.yaml"
CONFIG_FILE = "expts/configs/config_ipu_allsizes.yaml"
# CONFIG_FILE = "expts/configs/config_ipu_reproduce.yaml"
os.chdir(MAIN_DIR)


def main(cfg: DictConfig, trial, run_name="main") -> None:
    st = timeit.default_timer()

    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)

    #! Andy: we might want a dict of in_dim for pe encoders from datamodule
    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dim_nodes=datamodule.num_node_feats,
        in_dim_edges=datamodule.num_edge_feats,
        pe_in_dims=datamodule.pe_in_dims,
    )

    metrics = load_metrics(cfg)
    logger.info(metrics)

    predictor = load_predictor(cfg, model_class, model_kwargs, metrics)

    logger.info(predictor.model)
    logger.info(predictor.summarize(max_depth=4))

    trainer = load_trainer(cfg, run_name)

    datamodule.prepare_data()
    # Run the model training
    with SafeRun(name="TRAINING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.fit(model=predictor, datamodule=datamodule)

    with SafeRun(name="TESTING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.test(model=predictor, datamodule=datamodule)  # , ckpt_path=ckpt_path)

    logger.info("--------------------------------------------")
    logger.info("totoal computation used", timeit.default_timer() - st)
    logger.info("--------------------------------------------")

    return trainer.callback_metrics["cv/mae/test"].cpu().item()


if __name__ == "__main__":
    # nan_checker("goli/data/QM9/micro_qm9.csv") #can be deleted
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

        run_name = "no_name_" if not "name" in cfg["constants"] else cfg["constants"]["name"] + "_"
        run_name = run_name + date.today().strftime("%d/%m/%Y") + "_"
        for key, value in trial.params.items():
            run_name = run_name + str(key) + "=" + str(value) + "_"

        accu = main(cfg, trial, run_name=run_name[: len(run_name) - 1])
        wandb.finish()
        return accu

    study = optuna.create_study()
    study.optimize(partial(objective, cfg=cfg), n_trials=1, timeout=600)

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info("  Number of finished trials: ", len(study.trials))
    logger.info("  Number of complete trials: ", len(complete_trials))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: ", trial.value)

    params_str = "  Params: "
    for key, value in trial.params.items():
        params_str += "\n    {}: {}".format(key, value)
    logger.info(params_str)
