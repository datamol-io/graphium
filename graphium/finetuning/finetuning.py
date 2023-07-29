from typing import Iterable, List, Dict, Tuple, Union, Callable, Any, Optional, Type

import pytorch_lightning as pl

from torch.optim.optimizer import Optimizer
from pytorch_lightning.callbacks import BaseFinetuning


class GraphFinetuning(BaseFinetuning):
    def __init__(self, cfg, train_bn: bool = False):
        """
        Finetuning logic
        """
        super().__init__()

        cfg_finetune = cfg["finetuning"]
        cfg_arch = cfg["architecture"]
        self.finetuning_module = cfg_finetune["module_from_pretrained"]
        self.depth = cfg_finetune["added_depth"]
        self.depth += cfg_finetune.get("add_finetune_depth", 0)
        self.task = cfg_finetune["task"]
        self.level = cfg_finetune["level"]
        self.train_bn = train_bn
        self.epoch_unfreeze_all = cfg_finetune.get("epoch_unfreeze_all", None)

        module_list = ["pe_encoders", "pre_nn", "pre_nn_edges", "gnn", "graph_output_nn", "task_heads"]
        self.module_list = [module for module in module_list if cfg_arch[module] is not None]

    def freeze_before_training(self, pl_module):
        # Freeze everything up to finetuning module (and potentially parts of finetuning module)

        for module in self.module_list:
            if module == self.finetuning_module:
                break

            self.freeze_complete_module(pl_module, module)

        self.freeze_partial_module(pl_module, module)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer):
        if epoch == self.epoch_unfreeze_all:
            self.unfreeze_and_add_param_group(modules=pl_module, optimizer=optimizer, train_bn=self.train_bn)

    def freeze_complete_module(self, pl_module: pl.LightningModule, module: str):
        if module == "pe_encoders":
            modules = pl_module.model.encoder_manager
        elif module == "pre_nn":
            modules = pl_module.model.pre_nn
        elif module == "pre_nn_edges":
            modules = pl_module.model.pre_nn_edges
        elif module == "gnn":
            modules = pl_module.model.gnn
        elif module == "graph_output_nn":
            modules = pl_module.model.task_heads.graph_output_nn
        elif module == "task_heads":
            modules = pl_module.model.task_heads.task_heads

        self.freeze(modules=modules, train_bn=self.train_bn)

    def freeze_partial_module(self, pl_module: pl.LightningModule, module: str):
        if module in ["pe_encoders", "pre_nn", "pre_nn_edges"]:
            raise NotImplementedError(f"Finetune from pos. encoders or (edge) pre-NNs is not supported")
        elif module == "gnn":
            modules = pl_module.model.gnn.layers[: -self.depth]
        elif module == "graph_output_nn":
            modules = pl_module.model.task_heads.graph_output_nn[self.level].graph_output_nn.layers[
                : -self.depth
            ]
        elif module == "task_heads":
            modules = pl_module.model.task_heads.task_heads[self.task].layers[: -self.depth]

        self.freeze(modules=modules, train_bn=self.train_bn)
