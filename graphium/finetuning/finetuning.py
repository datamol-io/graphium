from typing import Iterable, List, Dict, Tuple, Union, Callable, Any, Optional, Type

from collections import OrderedDict

import pytorch_lightning as pl

from torch.optim.optimizer import Optimizer
from pytorch_lightning.callbacks import BaseFinetuning


class GraphFinetuning(BaseFinetuning):
    def __init__(
        self,
        task: str,
        level: str,
        finetuning_module: str,
        added_depth: int,
        unfreeze_pretrained_depth: Optional[int] = None,
        epoch_unfreeze_all: int = 0,
        train_bn: bool = False,
    ):
        """
        Finetuning logic
        """
        super().__init__()

        self.task = task
        self.level = level
        self.finetuning_module = finetuning_module
        self.training_depth = added_depth
        if unfreeze_pretrained_depth is not None:
            self.training_depth += unfreeze_pretrained_depth
        self.epoch_unfreeze_all = epoch_unfreeze_all
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        # Freeze everything up to finetuning module (and parts of finetuning module)
        self.module_map = pl_module.model.pretrained_model.net._module_map

        for module_name in self.module_map.keys():
            self.freeze_module(pl_module, module_name)

            if module_name.startswith(self.finetuning_module):
                break

    def freeze_module(self, pl_module: pl.LightningModule, module_name: str):
        modules = self.module_map[module_name]

        if module_name.startswith(self.finetuning_module):
            modules = modules[: -self.training_depth]

        self.freeze(modules=modules, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer):
        if epoch == self.epoch_unfreeze_all:
            self.unfreeze_and_add_param_group(modules=pl_module, optimizer=optimizer, train_bn=self.train_bn)
