from copy import deepcopy


def modify_cfg_for_finetuning(cfg):

    cfg_finetune = cfg['finetuning']
    module = cfg_finetune['module_from_pretrained']
    task = cfg_finetune['task']
    level = cfg_finetune['level']
    task_head_from_pretrained = cfg_finetune['task_head_from_pretrained']

    new_module_kwargs = deepcopy(cfg['architecture'][module][task_head_from_pretrained])

    upd_kwargs = {
        'task_level': level,
        'out_dim': cfg_finetune['new_out_dim'],
        'depth': new_module_kwargs['depth'] + cfg_finetune['added_depth'] - cfg_finetune['drop_depth']
    }

    new_module_kwargs.update(upd_kwargs)

    cfg['architecture'][module] = {
        task: new_module_kwargs
    }

    return cfg