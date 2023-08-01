from copy import deepcopy


def modify_cfg_for_finetuning(cfg):
    cfg_finetune = cfg["finetuning"]
    finetuning_module = cfg_finetune["finetuning_module"]
    task = cfg_finetune["task"]
    level = cfg_finetune["level"]
    task_head_from_pretrained = cfg_finetune.get("task_head_from_pretrained", None)

    # Find part of config of module to finetune from
    if finetuning_module == "gnn":
        new_module_kwargs = deepcopy(cfg["architecture"][finetuning_module])
    elif finetuning_module == "graph_output_nn":
        new_module_kwargs = deepcopy(cfg["architecture"][finetuning_module][level])
    elif finetuning_module == "task_heads":
        new_module_kwargs = deepcopy(cfg["architecture"][finetuning_module][task_head_from_pretrained])
    elif finetuning_module in ["pre_nn", "pre_nn_edges"]:
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
        cfg["architecture"][finetuning_module] = new_module_kwargs
    elif finetuning_module == "graph_output_nn":
        cfg["architecture"][finetuning_module] = {level: new_module_kwargs}
    elif finetuning_module == "task_heads":
        cfg["architecture"][finetuning_module] = {task: new_module_kwargs}

    # Remove modules of pretrained model after module to finetune from
    module_list = ["pre_nn", "pre_nn_edges", "gnn", "graph_output_nn", "task_heads"]
    cutoff_idx = (
        module_list.index(finetuning_module) + 1
    )  # Index of module after module to finetune from
    for module in module_list[cutoff_idx:]:
        cfg["architecture"][module] = None

    finetuning_overwriting_kwargs = deepcopy(cfg["finetuning"])
    finetuning_overwriting_kwargs.pop("pretrained_model")
    finetuning_overwriting_kwargs.pop("level")
    finetuning_overwriting_kwargs.pop("added_finetuning_depth", None)
    finetuning_overwriting_kwargs.pop("epoch_unfreeze_all")
    # cfg["finetuning"]["overwriting_kwargs"] = finetuning_overwriting_kwargs

    finetuning_training_kwargs = deepcopy(cfg["finetuning"])
    finetuning_training_kwargs.pop("pretrained_model")
    finetuning_training_kwargs.pop("task_head_from_pretrained")
    # cfg["finetuning"]["training_kwargs"] = finetuning_training_kwargs

    cfg["finetuning"].update({
        "overwriting_kwargs": finetuning_overwriting_kwargs,
        "training_kwargs": finetuning_training_kwargs
    })

    return cfg
