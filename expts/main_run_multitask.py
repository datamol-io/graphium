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
from pydoc import importfile

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
OPTUNA_CONFIG_FILE = "expts/optuna/optuna_base_config.py"
os.chdir(MAIN_DIR)

def main(cfg: DictConfig, trial, run_name="main") -> None:
    st = timeit.default_timer()

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
        trainer.test(model=predictor, datamodule=datamodule) #, ckpt_path=ckpt_path)

    logger.info ("--------------------------------------------")
    logger.info("totoal computation used", timeit.default_timer() - st)
    logger.info ("--------------------------------------------")

    return trainer.callback_metrics["cv/mae/test"].cpu().item()

if __name__ == "__main__":
    #nan_checker("goli/data/QM9/micro_qm9.csv") #can be deleted
    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)

    def objective(trial, cfg):
        cfg = deepcopy(cfg)
    
        optuna_config = importfile(os.path.join(MAIN_DIR, OPTUNA_CONFIG_FILE))
        cfg = optuna_config.update_configuration(trial, cfg)

        run_name = 'no_name_' if not "name" in cfg["constants"] else cfg["constants"]["name"] + "_"
        run_name = run_name + date.today().strftime("%d/%m/%Y") + "_"
        for key, value in trial.params.items():
            run_name = run_name + str(key) + "=" + str(value) + "_"

        accu = main(cfg, trial, run_name=run_name[:len(run_name) - 1])
        wandb.log(data={"cv/mae/test": accu})
        wandb.finish()
        return accu

    study = optuna.create_study()
    study.optimize(partial(objective, cfg=cfg), n_trials=5, timeout=600)

    # TODO - plt.save() to export to WandB (optional)
    # optuna.visualization.plot_optimization_history(study)
    # optuna.visualization.plot_slice(study)
    # optuna.visualization.plot_contour(study, params=['gnn.hid', 'gnn.depth'])

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info("  Number of finished trials: ", len(study.trials))
    logger.info("  Number of complete trials: ", len(complete_trials))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: ", trial.value)

    params_str = "  Params: "
    for key, value in trial.params.items():
        params_str += ("\n    {}: {}".format(key, value))
    logger.info(params_str)
