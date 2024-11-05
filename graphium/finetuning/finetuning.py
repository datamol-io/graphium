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


from typing import Iterable, List, Dict, Tuple, Union, Callable, Any, Optional, Type

from collections import OrderedDict

import torch.nn as nn
import pytorch_lightning as pl

from torch.optim.optimizer import Optimizer
from pytorch_lightning.callbacks import BaseFinetuning


class GraphFinetuning(BaseFinetuning):
    def __init__(
        self,
        finetuning_module: str,
        added_depth: int = 0,
        unfreeze_pretrained_depth: Optional[int] = None,
        epoch_unfreeze_all: Optional[int] = 0,
        always_freeze_modules: Optional[Union[List, str]] = None,
        train_bn: bool = False,
    ):
        """
        Finetuning training callback that (un)freezes modules as specified in the configuration file.
        By default, the modified layers of the fineuning module and the finetuning head are unfrozen.

        Parameters:
            finetuning_module: Module to finetune from
            added_depth: Number of layers of finetuning module that have been modified rel. to pretrained model
            unfreeze_pretrained_depth: Number of additional layers to unfreeze before layers modified rel. to pretrained model
            epoch_unfreeze_all: Epoch to unfreeze entire model
            always_freeze_modules: Module that always stay frozen while finetuning
            train_bn: Boolean value indicating if batchnorm layers stay in training mode

        """
        super().__init__()

        self.finetuning_module = finetuning_module
        self.training_depth = added_depth
        if unfreeze_pretrained_depth is not None:
            self.training_depth += unfreeze_pretrained_depth
        self.epoch_unfreeze_all = epoch_unfreeze_all
        self.always_freeze_modules = always_freeze_modules
        if self.always_freeze_modules == 'none':
            self.always_freeze_modules = None
        if isinstance(self.always_freeze_modules, str):
            self.always_freeze_modules = [self.always_freeze_modules]
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        """
        Freeze everything up to finetuning module (and parts of finetuning module)

        Parameters:
            pl_module: PredictorModule used for finetuning
        """

        # Access module map of pretrained module
        module_map = pl_module.model.pretrained_model.net._module_map

        for module_name in module_map.keys():
            self.freeze_module(pl_module, module_name, module_map)

            if module_name.startswith(self.finetuning_module):
                # Do not freeze modules after finetuning module
                break

    def freeze_module(self, pl_module, module_name: str, module_map: Dict[str, Union[nn.ModuleList, Any]]):
        """
        Freeze specific modules

        Parameters:
            module_name: Name of module to (partally) freeze
            module_map: Dictionary mapping from module_name to corresponding module(s)
        """
        modules = module_map[module_name]

        if module_name == "pe_encoders":
            for param in pl_module.model.pretrained_model.net.encoder_manager.parameters():
                param.requires_grad = False

        # We only partially freeze the finetuning module
        if module_name.startswith(self.finetuning_module):
            if self.training_depth == 0:
                pass
            else:
                modules = modules[: -self.training_depth]

        self.freeze(modules=modules, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer):
        """
        Function unfreezing entire model at specified epoch

        Parameters:
            pl_module: PredictorModule used for finetuning
            epoch: Current training epoch
            optimizer: Optimizer used for finetuning
        """
        if epoch == self.epoch_unfreeze_all:
            self.unfreeze_and_add_param_group(modules=pl_module, optimizer=optimizer, train_bn=self.train_bn)

            if self.always_freeze_modules is not None:
                for module_name in self.always_freeze_modules:
                    self.freeze_module(pl_module, module_name, pl_module.model.pretrained_model.net._module_map)