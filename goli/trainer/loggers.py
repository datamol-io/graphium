import os
import yaml
from copy import deepcopy

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class TensorBoardLoggerGoli(TensorBoardLogger):
    def __init__(self, full_configs=None, *args, **kwargs):
        self.full_configs = deepcopy(full_configs)
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, *args, **kwargs) -> None:

        with open(os.path.join(self.log_dir, "full_configs.yaml"), 'w') as file:
            yaml.dump(self.full_configs, file)

        return super().log_hyperparams(*args, **kwargs)


class WandbLoggerGoli(WandbLogger):
    def __init__(self, full_configs=None, *args, **kwargs):
        self.full_configs = deepcopy(full_configs)
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, *args, **kwargs) -> None:

        # Save the full configs as well
        with open(os.path.join(self.experiment.dir, "full_configs.yaml"), 'w') as file:
            yaml.dump(self.full_configs, file)

        return super().log_hyperparams(*args, **kwargs)


