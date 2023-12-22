"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


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
