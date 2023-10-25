import os
import unittest as ut
from copy import deepcopy
from os.path import abspath, dirname

import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

import graphium
from graphium.config._loader import (
    load_accelerator,
    load_architecture,
    load_datamodule,
    load_metrics,
    load_predictor,
    load_trainer,
    save_params_to_wandb,
)
from graphium.finetuning import GraphFinetuning, modify_cfg_for_finetuning
from graphium.trainer import PredictorModule

MAIN_DIR = dirname(dirname(abspath(graphium.__file__)))
CONFIG_FILE = "graphium/config/dummy_finetuning.yaml"

os.chdir(MAIN_DIR)


class Test_Finetuning(ut.TestCase):
    def test_finetuning_from_task_head(self):
        # Skip test if PyTDC package not installed
        try:
            import tdc
        except ImportError:
            self.skipTest("PyTDC needs to be installed to run this test. Use `pip install PyTDC`.")

        ##################################################
        ### Test modification of config for finetuning ###
        ##################################################

        cfg = graphium.load_config(name="dummy_finetuning_from_task_head")
        cfg = OmegaConf.to_container(cfg, resolve=True)

        cfg = modify_cfg_for_finetuning(cfg)

        # Initialize the accelerator
        cfg, accelerator_type = load_accelerator(cfg)

        # Load and initialize the dataset
        datamodule = load_datamodule(cfg, accelerator_type)
        datamodule.task_specific_args["lipophilicity_astrazeneca"].sample_size = 100

        # Initialize the network
        model_class, model_kwargs = load_architecture(
            cfg,
            in_dims=datamodule.in_dims,
        )

        datamodule.prepare_data()

        metrics = load_metrics(cfg)

        predictor = load_predictor(
            cfg,
            model_class,
            model_kwargs,
            metrics,
            datamodule.get_task_levels(),
            accelerator_type,
            datamodule.featurization,
            datamodule.task_norms,
        )

        # Create module map
        module_map = deepcopy(predictor.model.pretrained_model.net._module_map)

        cfg_finetune = cfg["finetuning"]
        finetuning_module = "".join([cfg_finetune["finetuning_module"], "-", cfg_finetune["task"]])
        finetuning_module_from_pretrained = "".join(
            [cfg_finetune["finetuning_module"], "-", cfg_finetune["sub_module_from_pretrained"]]
        )

        # Test for correctly modified shapes and number of layers in finetuning module
        self.assertEqual(
            len(module_map[finetuning_module]),
            3,
        )
        self.assertEqual(module_map[finetuning_module][-1].linear.weight.size(0), 8)
        self.assertEqual(predictor.model.finetuning_head.net.in_dim, 8)
        self.assertEqual(len(predictor.model.finetuning_head.net.layers), 2)
        self.assertEqual(predictor.model.finetuning_head.net.out_dim, 1)

        ################################################
        ### Test overwriting with pretrained weights ###
        ################################################

        # Load pretrained & replace in predictor
        pretrained_model = PredictorModule.load_pretrained_model(
            cfg["finetuning"]["pretrained_model"], device="cpu"
        ).model

        pretrained_model.create_module_map()
        module_map_from_pretrained = deepcopy(pretrained_model._module_map)

        # Finetuning module has only been partially overwritten
        loaded_layers = module_map_from_pretrained[finetuning_module_from_pretrained]
        overwritten_layers = module_map[finetuning_module]

        for idx, (loaded, overwritten) in enumerate(zip(loaded_layers, overwritten_layers)):
            if idx < 1:
                assert torch.equal(loaded.linear.weight, overwritten.linear.weight)
                assert torch.equal(loaded.linear.bias, overwritten.linear.bias)
            else:
                assert not torch.equal(loaded.linear.weight, overwritten.linear.weight)
                assert not torch.equal(loaded.linear.bias, overwritten.linear.bias)

            if idx + 1 == min(len(loaded_layers), len(overwritten_layers)):
                break

        for module_name in module_map.keys():
            if module_name == finetuning_module:
                break

            loaded_module, overwritten_module = (
                module_map_from_pretrained[module_name],
                module_map[module_name],
            )
            for loaded_params, overwritten_params in zip(
                loaded_module.parameters(), overwritten_module.parameters()
            ):
                assert torch.equal(loaded_params.data, overwritten_params.data)

        #################################################
        ### Test correct (un)freezing during training ###
        #################################################

        # Define test callback that checks for correct (un)freezing
        class TestCallback(Callback):
            def __init__(self, cfg):
                super().__init__()

                self.cfg_finetune = cfg["finetuning"]

            def on_train_epoch_start(self, trainer, pl_module):
                module_map = pl_module.model.pretrained_model.net._module_map

                finetuning_module = "".join(
                    [self.cfg_finetune["finetuning_module"], "-", self.cfg_finetune["task"]]
                )
                training_depth = self.cfg_finetune["added_depth"] + self.cfg_finetune.pop(
                    "unfreeze_pretrained_depth", 0
                )

                frozen_parameters, unfrozen_parameters = [], []

                if trainer.current_epoch == 0:
                    frozen = True

                    for module_name, module in module_map.items():
                        if module_name == finetuning_module:
                            # After the finetuning module, all parameters are unfrozen
                            frozen = False

                            frozen_parameters.extend(
                                [
                                    parameter.requires_grad
                                    for parameter in module[:-training_depth].parameters()
                                ]
                            )
                            unfrozen_parameters.extend(
                                [
                                    parameter.requires_grad
                                    for parameter in module[-training_depth:].parameters()
                                ]
                            )
                            continue

                        if frozen:
                            frozen_parameters.extend(
                                [parameter.requires_grad for parameter in module.parameters()]
                            )
                        else:
                            unfrozen_parameters.extend(
                                [parameter.requires_grad for parameter in module.parameters()]
                            )

                    # Finetuning head is always unfrozen
                    unfrozen_parameters.extend(
                        [
                            parameter.requires_grad
                            for parameter in pl_module.model.finetuning_head.parameters()
                        ]
                    )

                    assert not True in frozen_parameters
                    assert not False in unfrozen_parameters

                if trainer.current_epoch == 2:
                    # All parameter are unfrozen starting from epoch_unfreeze_all
                    unfrozen_parameters = [
                        parameter.requires_grad for parameter in pl_module.model.parameters()
                    ]

                    assert not False in unfrozen_parameters

        trainer = load_trainer(cfg, accelerator_type)

        finetuning_training_kwargs = cfg["finetuning"]["training_kwargs"]
        trainer.callbacks.append(GraphFinetuning(**finetuning_training_kwargs))

        # Add test callback to trainer
        trainer.callbacks.append(TestCallback(cfg))

        predictor.set_max_nodes_edges_per_graph(datamodule, stages=["train", "val"])

        # Run the model training
        trainer.fit(model=predictor, datamodule=datamodule)

    def test_finetuning_from_gnn(self):
        # Skip test if PyTDC package not installed
        try:
            import tdc
        except ImportError:
            self.skipTest("PyTDC needs to be installed to run this test. Use `pip install PyTDC`.")

        ##################################################
        ### Test modification of config for finetuning ###
        ##################################################

        cfg = graphium.load_config(name="dummy_finetuning_from_gnn")
        cfg = OmegaConf.to_container(cfg, resolve=True)

        cfg = modify_cfg_for_finetuning(cfg)

        # Initialize the accelerator
        cfg, accelerator_type = load_accelerator(cfg)

        # Load and initialize the dataset
        datamodule = load_datamodule(cfg, accelerator_type)
        datamodule.task_specific_args["lipophilicity_astrazeneca"].sample_size = 100

        # Initialize the network
        model_class, model_kwargs = load_architecture(
            cfg,
            in_dims=datamodule.in_dims,
        )

        datamodule.prepare_data()

        metrics = load_metrics(cfg)

        predictor = load_predictor(
            cfg,
            model_class,
            model_kwargs,
            metrics,
            datamodule.get_task_levels(),
            accelerator_type,
            datamodule.featurization,
            datamodule.task_norms,
        )

        # Create module map
        module_map = deepcopy(predictor.model.pretrained_model.net._module_map)

        cfg_finetune = cfg["finetuning"]
        finetuning_module = cfg_finetune["finetuning_module"]

        # Test for correctly modified shapes and number of layers in finetuning module
        self.assertEqual(
            len(module_map[finetuning_module]),
            5,
        )
        self.assertEqual(module_map[finetuning_module][-1].model.lin.weight.size(0), 96)
        self.assertEqual(len(module_map["graph_output_nn-graph"]), 2)

        assert predictor.model.pretrained_model.net.task_heads.graph_output_nn[
            "graph"
        ].graph_output_nn_kwargs["graph"]["pooling"] == ["mean"]

        ################################################
        ### Test overwriting with pretrained weights ###
        ################################################

        # Load pretrained & replace in predictor
        pretrained_model = PredictorModule.load_pretrained_model(
            cfg["finetuning"]["pretrained_model"], device="cpu"
        ).model

        pretrained_model.create_module_map()
        module_map_from_pretrained = deepcopy(pretrained_model._module_map)

        # Finetuning module has only been partially overwritten
        loaded_layers = module_map_from_pretrained[finetuning_module]
        overwritten_layers = module_map[finetuning_module]

        for idx, (loaded, overwritten) in enumerate(zip(loaded_layers, overwritten_layers)):
            if idx < 2:
                assert torch.equal(loaded.model.lin.weight, overwritten.model.lin.weight)
            else:
                assert not torch.equal(loaded.model.lin.weight, overwritten.model.lin.weight)

            if idx + 1 == min(len(loaded_layers), len(overwritten_layers)):
                break

        for module_name in module_map.keys():
            if module_name == finetuning_module:
                break

            loaded_module, overwritten_module = (
                module_map_from_pretrained[module_name],
                module_map[module_name],
            )
            for loaded_params, overwritten_params in zip(
                loaded_module.parameters(), overwritten_module.parameters()
            ):
                assert torch.equal(loaded_params.data, overwritten_params.data)

        #################################################
        ### Test correct (un)freezing during training ###
        #################################################

        # Define test callback that checks for correct (un)freezing
        class TestCallback(Callback):
            def __init__(self, cfg):
                super().__init__()

                self.cfg_finetune = cfg["finetuning"]

            def on_train_epoch_start(self, trainer, pl_module):
                module_map = pl_module.model.pretrained_model.net._module_map

                training_depth = self.cfg_finetune["added_depth"] + self.cfg_finetune.pop(
                    "unfreeze_pretrained_depth", 0
                )

                frozen_parameters, unfrozen_parameters = [], []

                if trainer.current_epoch == 0:
                    frozen = True

                    for module_name, module in module_map.items():
                        if module_name == finetuning_module:
                            # After the finetuning module, all parameters are unfrozen
                            frozen = False

                            frozen_parameters.extend(
                                [
                                    parameter.requires_grad
                                    for parameter in module[:-training_depth].parameters()
                                ]
                            )
                            unfrozen_parameters.extend(
                                [
                                    parameter.requires_grad
                                    for parameter in module[-training_depth:].parameters()
                                ]
                            )
                            continue

                        if frozen:
                            frozen_parameters.extend(
                                [parameter.requires_grad for parameter in module.parameters()]
                            )
                        else:
                            unfrozen_parameters.extend(
                                [parameter.requires_grad for parameter in module.parameters()]
                            )

                    assert not True in frozen_parameters
                    assert not False in unfrozen_parameters

                if trainer.current_epoch == 1:
                    # All parameter are unfrozen starting from epoch_unfreeze_all
                    unfrozen_parameters = [
                        parameter.requires_grad for parameter in pl_module.model.parameters()
                    ]

                    assert not False in unfrozen_parameters

        trainer = load_trainer(cfg, accelerator_type)

        finetuning_training_kwargs = cfg["finetuning"]["training_kwargs"]
        trainer.callbacks.append(GraphFinetuning(**finetuning_training_kwargs))

        # Add test callback to trainer
        trainer.callbacks.append(TestCallback(cfg))

        predictor.set_max_nodes_edges_per_graph(datamodule, stages=["train", "val"])

        # Run the model training
        trainer.fit(model=predictor, datamodule=datamodule)


if __name__ == "__main__":
    ut.main()
