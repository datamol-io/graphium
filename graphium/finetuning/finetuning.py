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
        added_finetuning_depth: Optional[int] = None,
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
        if added_finetuning_depth is not None:
            self.training_depth += added_finetuning_depth
        self.epoch_unfreeze_all = epoch_unfreeze_all
        self.train_bn = train_bn

        module_list = ["pe_encoders", "pre_nn", "pre_nn_edges", "gnn", "graph_output_nn", "task_heads"]
        self.module_list = [module for module in module_list if cfg_arch[module] is not None]

    def freeze_before_training(self, pl_module: pl.LightningModule):
        # Freeze everything up to finetuning module (and potentially parts of finetuning module)
        for name, module in pl_module._module_map.items():
            if name == self.finetuning_module:
                break

            # We need to filter out optional modules in case they are not present, e.g., pre-nn(-edges)
            if module is not None:
                self.freeze_complete_module(pl_module, name)

        self.freeze_partial_module(pl_module, name)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer):
        if epoch == self.epoch_unfreeze_all:
            self.unfreeze_and_add_param_group(modules=pl_module, optimizer=optimizer, train_bn=self.train_bn)

    def freeze_complete_module(self, pl_module: pl.LightningModule, name: str):
        modules = pl_module._module_map[name]
        self.freeze(modules=modules, train_bn=self.train_bn)

    def freeze_partial_module(self, pl_module: pl.LightningModule, name: str):
        if name in ["pe_encoders", "pre_nn", "pre_nn_edges"]:
            raise NotImplementedError(f"Finetune from pos. encoders or (edge) pre-NNs is not supported")
        elif name == "gnn":
            modules = pl_module._module_map[name].layers
        elif name == "graph_output_nn":
            modules = pl_module._module_map[name][self.level].graph_output_nn.layers
        elif name == "task_heads":
            modules = pl_module._module_map[name][self.task].layers
        else:
            raise "Wrong module"

        modules = modules[: -self.training_depth]

        self.freeze(modules=modules, train_bn=self.train_bn)
