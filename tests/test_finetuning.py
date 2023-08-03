import os
from os.path import dirname, abspath

import unittest as ut

import torch

from omegaconf import OmegaConf
import graphium

from graphium.finetuning import modify_cfg_for_finetuning
from graphium.trainer import PredictorModule

from graphium.config._loader import (
    load_datamodule,
    load_metrics,
    load_architecture,
    load_predictor,
    load_trainer,
    save_params_to_wandb,
    load_accelerator,
)


MAIN_DIR = dirname(dirname(abspath(graphium.__file__)))
CONFIG_FILE = "graphium/config/dummy_finetuning.yaml"

os.chdir(MAIN_DIR)


class Test_Multitask_DataModule(ut.TestCase):
    def test_cfg_modification(self):
        cfg = graphium.load_config(name="dummy_finetuning")
        cfg = OmegaConf.to_container(cfg, resolve=True)
        # cfg = load_yaml_config(CONFIG_FILE, MAIN_DIR)
        # dm_args = OmegaConf.to_container(cfg.datamodule.args, resolve=True)

        cfg = modify_cfg_for_finetuning(cfg)

        # Initialize the accelerator
        cfg, accelerator_type = load_accelerator(cfg)

        # Load and initialize the dataset
        datamodule = load_datamodule(cfg, accelerator_type)

        # Initialize the network
        model_class, model_kwargs = load_architecture(
            cfg,
            in_dims=datamodule.in_dims,
        )

        metrics = load_metrics(cfg)

        predictor = load_predictor(
            cfg, model_class, model_kwargs, metrics, accelerator_type, datamodule.task_norms
        )

        self.assertEqual(
            len(
                predictor.model.pretrained_model.net.task_heads.task_heads["lipophilicity_astrazeneca"].layers
            ),
            3,
        )
        self.assertEqual(
            predictor.model.pretrained_model.net.task_heads.task_heads["lipophilicity_astrazeneca"].out_dim, 8
        )
        self.assertEqual(predictor.model.finetuning_head.net.in_dim, 8)
        self.assertEqual(len(predictor.model.finetuning_head.net.layers), 2)
        self.assertEqual(predictor.model.finetuning_head.net.out_dim, 1)

        # Load pretrained & replace in predictor
        pretrained_model = PredictorModule.load_pretrained_models(cfg["finetuning"]["pretrained_model"]).model

        pretrained_model._create_module_map()
        module_map_from_pretrained = pretrained_model._module_map
        module_map = predictor.model.pretrained_model.net._module_map

        # GNN layers need to be the same
        pretrained_layers = module_map_from_pretrained["gnn"]
        overwritten_layers = module_map["gnn"]

        for pretrained, overwritten in zip(pretrained_layers, overwritten_layers):
            assert torch.equal(pretrained.model.lin.weight, overwritten.model.lin.weight)

        # Task head has only been partially overwritten
        pretrained_layers = pretrained_model.task_heads.task_heads["zinc"].layers
        overwritten_layers = predictor.model.pretrained_model.net.task_heads.task_heads[
            "lipophilicity_astrazeneca"
        ].layers

        for idx, (pretrained, overwritten) in enumerate(zip(pretrained_layers, overwritten_layers)):
            if idx < 1:
                assert torch.equal(pretrained.linear.weight, overwritten.linear.weight)
                assert torch.equal(pretrained.linear.bias, overwritten.linear.bias)
            else:
                assert not torch.equal(pretrained.linear.weight, overwritten.linear.weight)
                assert not torch.equal(pretrained.linear.bias, overwritten.linear.bias)

            if idx + 1 == min(len(pretrained_layers), len(overwritten_layers)):
                break


if __name__ == "__main__":
    ut.main()
