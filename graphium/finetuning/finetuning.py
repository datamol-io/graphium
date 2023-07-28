from typing import Iterable, List, Dict, Tuple, Union, Callable, Any, Optional, Type

import pytorch_lightning as pl

from torch.optim.optimizer import Optimizer
from pytorch_lightning.callbacks import BaseFinetuning


class MolecularFinetuning(BaseFinetuning):
    def __init__(self, cfg, train_bn: bool = False):
        """
        Finetuning logic
        """
        super().__init__()

        cfg_finetune = cfg["finetuning"]
        self.depth = cfg_finetune["added_depth"]
        try:
            self.depth += cfg_finetune["add_finetune_depth"]
        except:
            pass
        self.task = cfg_finetune["task"]
        self.train_bn = train_bn
        self.epoch_unfreeze_all = cfg_finetune["epoch_unfreeze_all"]

    def freeze_before_training(self, pl_module):
        # Initially freeze everything
        self.freeze(modules=pl_module, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer):
        # Unfreeze finetuning layers
        if epoch == 0:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.task_heads.task_heads[self.task].layers[-self.depth :],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )

        if epoch == self.epoch_unfreeze_all:
            self.unfreeze_and_add_param_group(modules=pl_module, optimizer=optimizer, train_bn=self.train_bn)
