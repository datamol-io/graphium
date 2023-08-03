from copy import deepcopy
from loguru import logger
from typing import Union, List
from graphium.trainer import PredictorModule

from graphium.utils.spaces import GRAPHIUM_PRETRAINED_MODELS_DICT


def filter_cfg_based_on_admet_benchmark_name(config, names: Union[List[str], str]):
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
    cfg["architecture"]["task_heads"] = _filter(cfg["architecture"]["task_heads"])
    cfg["predictor"]["metrics_on_progress_bar"] = _filter(cfg["predictor"]["metrics_on_progress_bar"])
    cfg["predictor"]["loss_fun"] = _filter(cfg["predictor"]["loss_fun"])
    cfg["metrics"] = _filter(cfg["metrics"])

    return cfg


def modify_cfg_for_finetuning(cfg):
    cfg_finetune = cfg["finetuning"]
    task = cfg_finetune["task"]

    # Filter the config based on the task name
    # NOTE (cwognum): This prevents the need for having many different files for each of the tasks
    #    with lots and lots of config repetition.
    cfg = filter_cfg_based_on_admet_benchmark_name(cfg, task)

    # Load pretrained model
    pretrained_model = cfg_finetune["pretrained_model"]
    pretrained_predictor = PredictorModule.load_from_checkpoint(
        GRAPHIUM_PRETRAINED_MODELS_DICT[pretrained_model]
    )

    # Inherit architecture from pretrained
    pretrained_architecture = pretrained_predictor.model_kwargs
    arch_keys = pretrained_architecture.keys()
    arch_keys = [key.replace("_kwargs", "") for key in arch_keys]
    cfg_arch = {arch_keys[idx]: value for idx, value in enumerate(pretrained_architecture.values())}

    finetuning_module = cfg_finetune["finetuning_module"]
    level = cfg_finetune["level"]
    sub_module_from_pretrained = cfg_finetune.get("sub_module_from_pretrained", None)

    # Find part of config of module to finetune from        # not sure how to make the code below more general;
    # it is specific to FullGraphMultitaskNetwork
    if finetuning_module == "gnn":
        new_module_kwargs = deepcopy(cfg_arch[finetuning_module])
    elif finetuning_module == "graph_output_nn":
        new_module_kwargs = deepcopy(cfg_arch[finetuning_module][level])
    elif finetuning_module == "task_heads":
        new_module_kwargs = deepcopy(cfg_arch[finetuning_module][sub_module_from_pretrained])
    elif finetuning_module in ["pe_encoders", "pre_nn", "pre_nn_edges"]:
        raise NotImplementedError(f"Finetune from (edge) pre-NNs is not supported")
    else:
        raise NotImplementedError(f"This is an unknown module type")

    # Modify config according to desired finetuning architecture
    upd_kwargs = {
        "out_dim": cfg_finetune.pop("new_out_dim"),
        "depth": new_module_kwargs["depth"]
        + cfg_finetune.get("added_depth", 0)
        - cfg_finetune.pop("drop_depth", 0),
    }

    # Update config
    new_module_kwargs.update(upd_kwargs)

    if finetuning_module == "gnn":
        cfg_arch[finetuning_module] = new_module_kwargs
    elif finetuning_module == "graph_output_nn":
        cfg_arch[finetuning_module] = {level: new_module_kwargs}
    elif finetuning_module == "task_heads":
        cfg_arch[finetuning_module] = {task: new_module_kwargs}

    # Remove modules of pretrained model after module to finetune from
    module_list = ["pre_nn", "pre_nn_edges", "gnn", "graph_output_nn", "task_heads"]
    cutoff_idx = module_list.index(finetuning_module) + 1  # Index of module after module to finetune from
    for module in module_list[cutoff_idx:]:
        cfg_arch[module] = None

    # Change architecture to FullGraphFinetuningNetwork
    cfg_arch["model_type"] = "FullGraphFinetuningNetwork"

    cfg["architecture"] = cfg_arch

    pretrained_overwriting_kwargs = deepcopy(cfg["finetuning"])
    drop_keys = [
        "pretrained_model",
        "task",
        "level",
        "finetuning_head",
        "unfreeze_pretrained_depth",
        "epoch_unfreeze_all",
    ]
    for key in drop_keys:
        pretrained_overwriting_kwargs.pop(key)
    # pretrained_overwriting_kwargs.pop("pretrained_model")
    # pretrained_overwriting_kwargs.pop("level")
    # pretrained_overwriting_kwargs.pop("finetuning_head")
    # pretrained_overwriting_kwargs.pop("unfreeze_pretrained_depth", None)
    # pretrained_overwriting_kwargs.pop("epoch_unfreeze_all")

    finetuning_training_kwargs = deepcopy(cfg["finetuning"])
    drop_keys = ["pretrained_model", "sub_module_from_pretrained", "finetuning_head"]
    for key in drop_keys:
        finetuning_training_kwargs.pop(key)
    # finetuning_training_kwargs.pop("pretrained_model")
    # finetuning_training_kwargs.pop("sub_module_from_pretrained")
    # finetuning_training_kwargs.pop("finetuning_head")

    cfg["finetuning"].update(
        {"overwriting_kwargs": pretrained_overwriting_kwargs, "training_kwargs": finetuning_training_kwargs}
    )

    return cfg
