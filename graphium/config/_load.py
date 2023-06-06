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
