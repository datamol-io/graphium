from copy import deepcopy
from typing import Any, Dict, List, Union

from loguru import logger

from graphium.trainer import PredictorModule

import graphium


def filter_cfg_based_on_admet_benchmark_name(config: Dict[str, Any], names: Union[List[str], str]):
    """
    Filter a base config for the full TDC ADMET benchmarking group to only
    have settings related to a subset of the endpoints
    """

    if config["datamodule"]["module_type"] != "ADMETBenchmarkDataModule":
        # NOTE (cwognum): For now, this implies we only support the ADMET benchmark from TDC.
        #    It is easy to extend this in the future to support more datasets.
        raise ValueError("You can only use this method for the `ADMETBenchmarkDataModule`")

    if isinstance(names, str):
        names = [names]

    def _filter(d):
        return {k: v for k, v in d.items() if k in names}

    cfg = deepcopy(config)

    # Update the datamodule arguments
    cfg["datamodule"]["args"]["tdc_benchmark_names"] = names

    # Filter the relevant config sections
    if "architecture" in cfg and "task_heads" in cfg["architecture"]:
        cfg["architecture"]["task_heads"] = _filter(cfg["architecture"]["task_heads"])
    if "predictor" in cfg and "metrics_on_progress_bar" in cfg["predictor"]:
        cfg["predictor"]["metrics_on_progress_bar"] = _filter(cfg["predictor"]["metrics_on_progress_bar"])
    if "predictor" in cfg and "loss_fun" in cfg["predictor"]:
        cfg["predictor"]["loss_fun"] = _filter(cfg["predictor"]["loss_fun"])
    if "metrics" in cfg:
        cfg["metrics"] = _filter(cfg["metrics"])

    return cfg


def modify_cfg_for_finetuning(cfg: Dict[str, Any]):
    """
    Function combining information from configuration and pretrained model for finetuning.
    """
    task = cfg["finetuning"]["task"]

    # Filter the config based on the task name
    # NOTE (cwognum): This prevents the need for having many different files for each of the tasks
    #    with lots and lots of config repetition.
    cfg = filter_cfg_based_on_admet_benchmark_name(cfg, task)
    cfg_finetune = cfg["finetuning"]

    # Load pretrained model
    pretrained_model = cfg_finetune["pretrained_model"]
    pretrained_predictor = PredictorModule.load_pretrained_model(pretrained_model, device="cpu")

    # Inherit shared configuration from pretrained
    # Architecture
    pretrained_architecture = pretrained_predictor.model_kwargs
    arch_keys = pretrained_architecture.keys()
    arch_keys = [key.replace("_kwargs", "") for key in arch_keys]
    cfg_arch = {arch_keys[idx]: value for idx, value in enumerate(pretrained_architecture.values())}

    cfg_arch_from_pretrained = deepcopy(cfg_arch)
    # Featurization
    cfg["datamodule"]["args"]["featurization"] = pretrained_predictor.featurization

    finetuning_module = cfg_finetune["finetuning_module"]
    sub_module_from_pretrained = cfg_finetune.get("sub_module_from_pretrained", None)
    new_sub_module = cfg_finetune.pop("new_sub_module", None)
    keep_modules_after_finetuning_module = cfg_finetune.pop("keep_modules_after_finetuning_module", None)

    # Find part of config of module to finetune from
    pretrained_predictor.model.create_module_map()
    module_map_from_pretrained = pretrained_predictor.model._module_map

    if not any([module.startswith(finetuning_module) for module in module_map_from_pretrained.keys()]):
        raise ValueError("Unkown module {finetuning_module}")
    elif sub_module_from_pretrained is None:
        new_module_kwargs = deepcopy(cfg_arch[finetuning_module])
    else:
        new_module_kwargs = deepcopy(cfg_arch[finetuning_module][sub_module_from_pretrained])

    # Modify config according to desired finetuning architecture
    out_dim = (
        cfg_arch[finetuning_module].get("out_dim")
        if sub_module_from_pretrained is None
        else cfg_arch[finetuning_module][sub_module_from_pretrained].get("out_dim")
    )

    if new_module_kwargs["depth"] is None:
        new_module_kwargs["depth"] = len(new_module_kwargs["hidden_dims"]) + 1

    upd_kwargs = {
        "out_dim": cfg_finetune.pop("new_out_dim", out_dim),
        "depth": new_module_kwargs["depth"]
        + cfg_finetune.get("added_depth", 0)
        - cfg_finetune.pop("drop_depth", 0),
    }

    new_last_activation = cfg_finetune.pop("new_last_activation", None)
    if new_last_activation is not None:
        upd_kwargs["last_activation"] = new_last_activation

    # Update config
    new_module_kwargs.update(upd_kwargs)

    if sub_module_from_pretrained is None:
        cfg_arch[finetuning_module] = new_module_kwargs
    else:
        cfg_arch[finetuning_module] = {new_sub_module: new_module_kwargs}

    # Remove modules of pretrained model after module to finetune from unless specified differently
    module_list = list(module_map_from_pretrained.keys())
    super_module_list = []
    for module in module_list:
        if module.split("-")[0] not in super_module_list:  # Only add each supermodule once
            super_module_list.append(module.split("-")[0])

    # Set configuration of modules after finetuning module to None
    cutoff_idx = (
        super_module_list.index(finetuning_module) + 1
    )  # Index of module after module to finetune from
    for module in super_module_list[cutoff_idx:]:
        cfg_arch[module] = None

    # If desired, we can keep specific modules after the finetuning module (specified in cfg/finetuning/keep_modules_after_finetuning_module)
    if keep_modules_after_finetuning_module is not None:
        for module_name, updates in keep_modules_after_finetuning_module.items():
            cfg_arch = update_cfg_arch_for_module(cfg_arch, cfg_arch_from_pretrained, module_name, updates)

    # Change architecture to FullGraphFinetuningNetwork
    cfg_arch["model_type"] = "FullGraphFinetuningNetwork"

    cfg["architecture"] = cfg_arch

    pretrained_overwriting_kwargs = deepcopy(cfg["finetuning"])
    drop_keys = [
        "task",
        "level",
        "finetuning_head",
        "unfreeze_pretrained_depth",
        "epoch_unfreeze_all",
    ]

    for key in drop_keys:
        pretrained_overwriting_kwargs.pop(key, None)

    finetuning_training_kwargs = deepcopy(cfg["finetuning"])
    drop_keys = ["task", "level", "pretrained_model", "sub_module_from_pretrained", "finetuning_head"]
    for key in drop_keys:
        finetuning_training_kwargs.pop(key, None)

    cfg["finetuning"].update(
        {"overwriting_kwargs": pretrained_overwriting_kwargs, "training_kwargs": finetuning_training_kwargs}
    )

    return cfg


def update_cfg_arch_for_module(
    cfg_arch: Dict[str, Any],
    cfg_arch_from_pretrained: Dict[str, Any],
    module_name: str,
    updates: Dict[str, Any],
):
    """
    Function to modify the key-word arguments of modules after the finetuning module if they are kept.

    Parameters:
        cfg_arch: Configuration of the architecture of the model used for finetuning
        cfg_arch_from_pretrained: Configuration of the architecture of the loaded pretrained model
        module_name: Module of loaded pretrained model
        updates: Changes to apply to key-work arguments of selected module
    """
    # We need to distinguish between modules with & without submodules
    if "-" not in module_name:
        if cfg_arch[module_name] is None:
            cfg_arch[module_name] = {}

        cfg_arch_from_pretrained[module_name].update({key: value for key, value in updates.items()})

        cfg_arch.update({module_name, cfg_arch_from_pretrained})

    else:
        module_name, sub_module = module_name.split("-")
        new_sub_module = updates.pop("new_sub_module", sub_module)

        if cfg_arch[module_name] is None:
            cfg_arch[module_name] = {}

        cfg_arch_from_pretrained[module_name][sub_module].update(
            {key: value for key, value in updates.items()}
        )
        cfg_arch[module_name].update({new_sub_module: cfg_arch_from_pretrained[module_name][sub_module]})

    return cfg_arch
