import hydra
import timeit

from omegaconf import DictConfig, OmegaConf
from loguru import logger

from graphium.config._loader import load_datamodule, load_accelerator


@hydra.main(version_base=None, config_path="../../expts/hydra-configs", config_name="main")
def cli(cfg: DictConfig) -> None:
    """
    CLI endpoint for preparing the data in advance.
    """
    run_prepare_data(cfg)


def run_prepare_data(cfg: DictConfig) -> None:
    """
    The main (pre-)training and fine-tuning loop.
    """

    cfg = OmegaConf.to_container(cfg, resolve=True)

    st = timeit.default_timer()

    ## == Instantiate all required objects from their respective configs ==
    # Accelerator
    cfg, accelerator_type = load_accelerator(cfg)

    ## Data-module
    datamodule = load_datamodule(cfg, accelerator_type)

    datamodule.prepare_data()

    logger.info(f"Data preparation took {timeit.default_timer() - st:.2f} seconds.")


if __name__ == "__main__":
    cli()
