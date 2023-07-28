from copy import deepcopy


def modify_cfg_for_finetuning(cfg):
    cfg_finetune = cfg["finetuning"]
    module_from_pretrained = cfg_finetune["module_from_pretrained"]
    task = cfg_finetune["task"]
    level = cfg_finetune["level"]
    task_head_from_pretrained = cfg_finetune.get("task_head_from_pretrained", None)

    # Find part of config of module to finetune from
    if module_from_pretrained == "gnn":
        new_module_kwargs = deepcopy(cfg["architecture"][module_from_pretrained])
    elif module_from_pretrained == "graph_output_nn":
        new_module_kwargs = deepcopy(cfg["architecture"][module_from_pretrained][level])
    elif module_from_pretrained == "task_heads":
        new_module_kwargs = deepcopy(cfg["architecture"][module_from_pretrained][task_head_from_pretrained])
    elif module_from_pretrained in ["pre_nn", "pre_nn_edges"]:
        raise NotImplementedError(f"Finetune from (edge) pre-NNs is not supported")
    else:
        raise NotImplementedError(f"This is an unknown module type")

    # Modify config according to desired finetuning architecture
    upd_kwargs = {
        "out_dim": cfg_finetune["new_out_dim"],
        "depth": new_module_kwargs["depth"] + cfg_finetune.get("added_depth", 0) - cfg_finetune.get("drop_depth", 0),
    }

    # Update config
    new_module_kwargs.update(upd_kwargs)

    if module_from_pretrained == "gnn":
        cfg["architecture"][module_from_pretrained] = new_module_kwargs
    elif module_from_pretrained == "graph_output_nn":
        cfg["architecture"][module_from_pretrained] = {level: new_module_kwargs}
    elif module_from_pretrained == "task_heads":
        cfg["architecture"][module_from_pretrained] = {task: new_module_kwargs}

    # Remove modules of pretrained model after module to finetune from
    module_list = ["pre_nn", "pre_nn_edges", "gnn", "graph_output_nn", "task_heads"]
    cutoff_idx = module_list.index(module_from_pretrained) + 1                          # Index of module after module to finetune from
    for module in module_list[cutoff_idx:]:
        cfg["architecture"][module] = None

    return cfg
