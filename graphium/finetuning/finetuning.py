from typing import Iterable, List, Dict, Tuple, Union, Callable, Any, Optional, Type

from collections import OrderedDict

import pytorch_lightning as pl

from torch.optim.optimizer import Optimizer
from pytorch_lightning.callbacks import BaseFinetuning


class GraphFinetuning(BaseFinetuning):
    def __init__(
        self,
        cfg_arch: Dict[str, Any],
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
        # Freeze everything up to finetuning module (and potentially parts of finetuning module)
        self.module_map = pl_module.model.pretrained_model.net._module_map
        
        # Remove modules that are not in pretrained model (and hence neither in FullGraphFinetuningNetwork)
        self.drop_modules = [module_name for module_name in self.module_map.keys() if self.module_map[module_name] is None]
        for module_name in self.drop_modules:
            self.module_map.pop(module_name)

        for module_name, module in self.module_map.items():
            if module_name == self.finetuning_module:
                break

            # We need to filter out optional modules in case they are not present, e.g., pre-nn(-edges)
            if module is not None:
                self.freeze_complete_module(pl_module, module_name)

        self.freeze_partial_module(pl_module, module_name)

    def freeze_complete_module(self, pl_module: pl.LightningModule, module_name: str):
        modules = self.module_map[module_name]
        self.freeze(modules=modules, train_bn=self.train_bn)

    def freeze_partial_module(self, pl_module: pl.LightningModule, module_name: str):
        # Below code is still specific to finetuning a FullGraphMultitaskNetwork
        # A solution would be to create a second module_map_layers that maps to nn.ModuleDict
        if module_name in self.drop_modules:
            raise NotImplementedError(f"Finetune from pos. encoders or (edge) pre-NNs is not supported")
        elif module_name == "gnn":
            modules = self.module_map[module_name].layers
        elif module_name == "graph_output_nn":
            modules = self.module_map[module_name][self.level].graph_output_nn.layers
        elif module_name == "task_heads":
            modules = self.module_map[module_name][self.task].layers
        else:
            raise "Wrong module"

        modules = modules[: -self.training_depth]

        self.freeze(modules=modules, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer):
        if epoch == self.epoch_unfreeze_all:
            self.unfreeze_and_add_param_group(modules=pl_module, optimizer=optimizer, train_bn=self.train_bn)
