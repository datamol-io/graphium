from copy import deepcopy


def modify_cfg_for_finetuning(cfg):

    cfg_finetune = cfg['trainer']['finetuning']
    module = cfg_finetune['module']
    task = cfg_finetune['task']
    level = cfg_finetune['level']
    choice = cfg_finetune['choice']
    drop_depth = cfg_finetune['drop_depth']

    new_module_kwargs = deepcopy(cfg['architecture'][module][choice])

    upd_kwargs = {
        'task_level': level,
        'out_dim': cfg_finetune['added_layers']['out_dim'],
        'depth': new_module_kwargs['depth'] + cfg_finetune['added_layers']['depth'] - drop_depth
    }

    new_module_kwargs.update(upd_kwargs)

    cfg['architecture'][module] = {
        task: new_module_kwargs
    }

    return cfg