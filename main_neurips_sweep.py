# General imports
import yaml
from loguru import logger
import wandb
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf

# from goli_wandb.utils import split_model_config_path, objective




wandb.login()

# General imports
from typing import Dict, Any, Tuple, List
import os
import yaml
from copy import deepcopy
from loguru import logger
import wandb

wandb.login()

from expts.main_run_multitask import (
    main,
)  # Move the `main` function from goli to the current directory

# Change the working directory to the current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def split_model_config_path(
    sweep_cfg: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    Split the `model_config_path` entry from the sweep configuration file
    by looping all elements with the `"parameters"` key.

    Parameters:
        sweep_cfg: Configuration file for the sweep.

    Returns:
        sweep_cfg: Updated configuration file for the sweep.
        model_config_path: Dictionary with the `model_config_path` entries,
            used to map the sweep parameters to the model configuration file.
    """

    # Remove the `model_config_path` entry from the sweep parameters
    sweep_cfg = deepcopy(sweep_cfg)
    model_config_path = {}
    for config_key, config_param in sweep_cfg["parameters"].items():
        model_config_path[config_key] = config_param.pop("model_config_path")

    return sweep_cfg, model_config_path


def update_cfg_with_sweep(
    sweep_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    model_config_path: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Update the configuration file with the parameters from the sweep.

    Parameters:
        sweep_cfg: Configuration dictionary for the sweep.
        model_cfg: Configuration dictionary for the model.
        model_config_path: Dictionary with the `model_config_path` entries,
            used to map the sweep parameters to the model configuration file.
            Each key should be a list of strings, splittable with the period character `'.'`,
            where each split is a sub-key. The list of subkeys are associated
            to the structure of the `model_cfg` dictionary.

    Returns:
        model_cfg: Updated configuration file for the model.
    """

    model_cfg = deepcopy(model_cfg)

    # Get the parameters from the sweep
    sweep_parameters = sweep_cfg["parameters"]

    # Update the configuration file with the parameters from the sweep
    for config_key in sweep_parameters.keys():
        this_model_config_path = model_config_path[config_key]
        if isinstance(this_model_config_path, str):
            this_model_config_path = [this_model_config_path]

        # Split the path into sub-path, and loop within the dictionary, to finaly update the last key
        for path in this_model_config_path:
            sub_path = path.split(".")  # Split the path into sub-path
            this_cfg = model_cfg
            for key in sub_path[:-1]:
                this_cfg = this_cfg[key]
            try:
                this_cfg[sub_path[-1]] = wandb.config[config_key]
            except KeyError as e:
                if sub_path[-1] not in this_cfg.keys():
                    raise KeyError(
                        f"Key `{config_key}` with path `{path}` not found in the sweep configuration file."
                    )
                else:
                    raise e

    return model_cfg


def objective(
    sweep_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    sweep_metric: str,
    model_config_path: Dict[str, List[str]],
):
    """
    Define the objective function for the sweep.

    Parameters:
        sweep_cfg: Configuration file for the sweep.
        model_config: Path to the base configuration file for the model.
        sweep_metric: Name of the metric that is used to optimize the sweep.
        model_config_path: Dictionary with the `model_config_path` entries,
            used to map the sweep parameters to the model configuration file.
            Each path should be a list of strings, splittable with the period character `'.'`,
            where each split is a sub-path. The list of subpaths are associated
            to the structure of the `model_cfg` dictionary.

    """

    run_name = sweep_cfg["name"]
    run = wandb.init(project=run_name)

    # Update the model config file, and save them as a YAML file
    model_cfg = update_cfg_with_sweep(sweep_cfg, model_cfg, model_config_path)
    with open(os.path.join(run.dir, "model_configs.yaml"), "w") as file:
        yaml.dump(model_cfg, file)

    metrics = main(model_cfg, run_name=run_name)
    final_metric = metrics[sweep_metric]

    logger.info(f"{sweep_metric}: {final_metric}")
    run.log({sweep_metric: final_metric})



@hydra.main(version_base=None, config_path="expts/neurips2023_configs", config_name="sweep_config")
def run_sweep(full_cfg: DictConfig) -> None:
    # Get the main entries from the configuration file
    # import ipdb; ipdb.set_trace()
    project_name = full_cfg["project_name"]
    sweep_cfg = OmegaConf.to_object(full_cfg["sweep_cfg"])
    sweep_metric = sweep_cfg["metric"]["name"]

    # Load the configuration files for the model
    with open(full_cfg["model_config"], "r") as f:
        model_cfg = yaml.safe_load(f)

    # Print the configuration files for the sweep
    logger.info(
        "full configuration: \n_________________________________\n"
        + OmegaConf.to_yaml(full_cfg)
    )

    # Start the sweep
    sweep_cfg, model_config_path = split_model_config_path(sweep_cfg)
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=project_name)
    objective_fn = partial(
        objective, sweep_cfg, model_cfg, sweep_metric, model_config_path
    )
    wandb.agent(sweep_id, function=objective_fn, count=1)
    wandb.finish()


if __name__ == "__main__":
    run_sweep()
