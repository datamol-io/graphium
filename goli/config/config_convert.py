import omegaconf


def recursive_config_reformating(configs):
    r"""
    For a given configuration file, convert all `DictConfig` to `dict`,
    all `ListConfig` to `list`, and all `byte` to `str`.

    This helps avoid errors when dumping a yaml file.
    """

    if isinstance(configs, omegaconf.DictConfig):
        configs = dict(configs)
    elif isinstance(configs, omegaconf.ListConfig):
        configs = list(configs)

    if isinstance(configs, dict):
        for k, v in configs.items():
            if isinstance(v, bytes):
                configs[k] = str(v)
            else:
                configs[k] = recursive_config_reformating(v)
    elif isinstance(configs, list):
        for k, v in enumerate(configs):
            if isinstance(v, bytes):
                configs[k] = str(v)
            else:
                configs[k] = recursive_config_reformating(v)

    return configs
