import os
import enum
import fsspec
import hydra
import torch
import yaml

from datamol.utils import fs


class _Keys(enum.Enum):
    OBJECTIVE = "objective"
    SAVE_DESTINATION = "save_destination"
    FORCE = "overwrite_destination"


def process_results_for_hyper_param_search(results: dict, cfg: dict):
    """Processes the results in the context of a hyper-parameter search."""

    # Save the results to the current work directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]

    results = {k: v.item() if torch.is_tensor(v) else v for k, v in results.items()}
    with fsspec.open(fs.join(output_dir, "trial_results.json"), "w") as f:
        yaml.dump(results, f)

    # Copy the current working directory to remote
    dst_dir = cfg.get(_Keys.SAVE_DESTINATION.value)
    if dst_dir is not None:
        relpath = os.path.relpath(output_dir, os.getcwd())
        dst = fs.join(dst_dir, relpath)
        fs.copy_dir(output_dir, dst, force=cfg.get(_Keys.FORCE, False))

    # Extract the objectives
    objectives = cfg[_Keys.OBJECTIVE.value]
    if isinstance(objectives, str):
        objectives = [objectives]

    # Extract the objective values
    objective_values = [results[k] for k in objectives]
    if len(objective_values) == 1:
        objective_values = objective_values[0]
    return objective_values
