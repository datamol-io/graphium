"""
Unit tests for the file goli/trainer/refactor_predictor.py
"""
import os
from os.path import dirname, abspath
import yaml

import torch
from torch.nn import BCELoss, MSELoss
import unittest as ut

from goli.trainer.refactor_predictor import PredictorModule
# Current project imports
import goli
from goli.config._loader import load_datamodule, load_metrics, load_metrics_mtl, load_architecture, load_predictor, load_trainer



# class test_Refactor_Predictor_EpochSummaries(ut.TestCase):
#     def test_create_epoch_summaries(self):
#         # Set up the working directory
#         MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
#         # CONFIG_FILE = "expts/config_molPCBA.yaml"
#         # CONFIG_FILE = "expts/config_molHIV.yaml"
#         # CONFIG_FILE = "expts/config_molPCQM4M.yaml"
#         # CONFIG_FILE = "expts/config_molPCQM4Mv2.yaml"
#         CONFIG_FILE = "expts/config_micro_ZINC.yaml"
#         # CONFIG_FILE = "expts/config_micro-PCBA.yaml"
#         # CONFIG_FILE = "expts/config_ZINC_bench_gnn.yaml"
#         os.chdir(MAIN_DIR)

#         with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
#             cfg = yaml.safe_load(f)

#         # Load and initialize the dataset
#         datamodule = load_datamodule(cfg)
#         print("\ndatamodule:\n", datamodule, "\n")

#         # Initialize the network
#         model_class, model_kwargs = load_architecture(
#             cfg,
#             in_dim_nodes=datamodule.num_node_feats_with_positional_encoding,
#             in_dim_edges=datamodule.num_edge_feats,
#         )

#         metrics = load_metrics(cfg)
#         print(metrics)

#         # Currently using the REFACTORED predictor
#         predictor = load_predictor(cfg, model_class, model_kwargs, metrics)

#         print(predictor.model)
#         print(predictor.summarize(max_depth=4))

#         trainer = load_trainer(cfg)

#         # Run the model training
#         print("\n------------ TRAINING STARTED ------------")
#         try:
#             trainer.fit(model=predictor, datamodule=datamodule)
#             print("\n------------ TRAINING COMPLETED ------------\n\n")

#         except Exception as e:
#             if not cfg["constants"]["raise_train_error"]:
#                 print("\n------------ TRAINING ERROR: ------------\n\n", e)
#             else:
#                 raise e

#         print("\n------------ TESTING STARTED ------------")
#         try:
#             ckpt_path = trainer.checkpoint_callbacks[0].best_model_path
#             trainer.test(model=predictor, datamodule=datamodule, ckpt_path=ckpt_path)
#             print("\n------------ TESTING COMPLETED ------------\n\n")

#         except Exception as e:
#             if not cfg["constants"]["raise_train_error"]:
#                 print("\n------------ TESTING ERROR: ------------\n\n", e)
#             else:
#                 raise e

#         print("EPOCH SUMMARY", predictor.epoch_summary.summaries)


class test_Refactor_Multitask_Predictor_EpochSummaries(ut.TestCase):
    def test_create_epoch_summaries(self):
        # Set up the working directory
        MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
        CONFIG_FILE = "tests/mtl/config_micro_ZINC_mtl_test.yaml"
        os.chdir(MAIN_DIR)

        with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
            cfg = yaml.safe_load(f)

        # Load and initialize the dataset
        datamodule = load_datamodule(cfg)
        #print("\ndatamodule:\n", datamodule, "\n")

        # Initialize the network
        model_class, model_kwargs = load_architecture(
            cfg,
            in_dim_nodes=datamodule.num_node_feats_with_positional_encoding,
            in_dim_edges=datamodule.num_edge_feats,
        )

        metrics = load_metrics_mtl(cfg)
        print(metrics)

        # Currently using the REFACTORED predictor
        predictor = load_predictor(cfg, model_class, model_kwargs, metrics)

        print(predictor.model)
        print(predictor.summarize(max_depth=4))

        trainer = load_trainer(cfg)

        # Run the model training
        print("\n------------ TRAINING STARTED ------------")
        try:
            trainer.fit(model=predictor, datamodule=datamodule)
            print("\n------------ TRAINING COMPLETED ------------\n\n")

        except Exception as e:
            if not cfg["constants"]["raise_train_error"]:
                print("\n------------ TRAINING ERROR: ------------\n\n", e)
            else:
                raise e

        print("\n------------ TESTING STARTED ------------")
        try:
            ckpt_path = trainer.checkpoint_callbacks[0].best_model_path
            trainer.test(model=predictor, datamodule=datamodule, ckpt_path=ckpt_path)
            print("\n------------ TESTING COMPLETED ------------\n\n")

        except Exception as e:
            if not cfg["constants"]["raise_train_error"]:
                print("\n------------ TESTING ERROR: ------------\n\n", e)
            else:
                raise e

if __name__ == "__main__":
    ut.main()