"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals.
Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


import importlib.resources

import omegaconf


def load_config(name: str):
    """Load a default config file by its name.

    Args:
        name: name of the config to load.
    """

    with importlib.resources.open_text("graphium.config", f"{name}.yaml") as f:
        config = omegaconf.OmegaConf.load(f)

    return config
