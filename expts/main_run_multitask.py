import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="hydra-configs", config_name="main")
def main(cfg: DictConfig) -> None:
    raise DeprecationWarning(
        "This script is deprecated. Use `python graphium/cli/train_finetune.py` (or `graphium-train`) instead!"
    )


if __name__ == "__main__":
    main()
