# General imports
import os
from os.path import dirname, abspath, join
import yaml
from copy import deepcopy
from functools import partial
from datetime import date
from loguru import logger
from pydoc import importfile

# Current project imports
import goli

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
OPTUNA_CONFIG_FILE = "expts/optuna/optuna_base_config.py"
os.chdir(MAIN_DIR)

from expts.main_run_multitask import main

if __name__ == "__main__":
    # nan_checker("goli/data/QM9/micro_qm9.csv") #can be deleted
    with open(join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)

    def objective(trial, cfg):
        cfg = deepcopy(cfg)

        optuna_config = importfile(join(MAIN_DIR, OPTUNA_CONFIG_FILE))
        cfg = optuna_config.update_configuration(trial, cfg)

        run_name = "no_name_" if not "name" in cfg["constants"] else cfg["constants"]["name"] + "_"
        run_name = run_name + date.today().strftime("%d/%m/%Y") + "_"
        for key, value in trial.params.items():
            run_name = run_name + str(key) + "=" + str(value) + "_"

        metrics = main(cfg, run_name=run_name[: len(run_name) - 1])
        final_loss = metrics["loss/test"]
        wandb.finish()
        return final_loss

    study = optuna.create_study()
    study.optimize(partial(objective, cfg=cfg), n_trials=5, timeout=600)

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
