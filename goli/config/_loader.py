from typing import Dict, Mapping, Union, Any

import omegaconf
from copy import deepcopy
import torch
from loguru import logger

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from goli.ipu.ipu_dataloader import IPUDataloaderOptions

from goli.trainer.metrics import MetricWrapper
from goli.nn.architectures import FullGraphNetwork, FullGraphSiameseNetwork, FullGraphMultiTaskNetwork
from goli.trainer.predictor import PredictorModule
from goli.utils.spaces import DATAMODULE_DICT
from goli.ipu.ipu_wrapper import PredictorModuleIPU, IPUPluginGoli
from goli.ipu.ipu_utils import import_poptorch, load_ipu_options
from goli.trainer.loggers import WandbLoggerGoli

# Weights and Biases
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

def get_accelerator(
    config: Union[omegaconf.DictConfig, Dict[str, Any]],
) -> str:

    # Get the accelerator type
    accelerator = config["constants"].get("accelerator")
    acc_type = None
    if isinstance(accelerator, Mapping):
        acc_type = accelerator.get("type", None)
    if acc_type is not None:
        acc_type = acc_type.lower()

    # Get the GPU info
    gpus = config["trainer"]["trainer"].get("gpus", 0)
    if gpus > 0:
        assert (acc_type is None) or (acc_type == "gpu"), "Accelerator mismatch"
        acc_type = "gpu"

    if (acc_type == "gpu") and (not torch.cuda.is_available()):
        logger.warning(
            f"GPUs selected, but will be ignored since no GPU are available on this device"
        )
        acc_type = "cpu"

    # Get the IPU info
    ipus = config["trainer"]["trainer"].get("ipus", 0)
    if ipus > 0:
        assert (acc_type is None) or (acc_type == "ipu"), "Accelerator mismatch"
        acc_type = "ipu"
    if acc_type == "ipu":
        poptorch = import_poptorch()
        if not poptorch.ipuHardwareIsAvailable():
            logger.warning(
                f"IPUs selected, but will be ignored since no IPU are available on this device"
            )
            acc_type = "cpu"

    # Fall on cpu at the end
    if acc_type is None:
        acc_type = "cpu"
    return acc_type


def load_datamodule(
    config: Union[omegaconf.DictConfig, Dict[str, Any]]
):
    cfg_data = config["datamodule"]["args"]
    ipu_inference_opts, ipu_training_opts = None, None
    ipu_file = "tests/mtl/ipu.config"
    ipu_dataloader_training_opts = cfg_data.pop("ipu_dataloader_training_opts", None)
    ipu_dataloader_inference_opts = cfg_data.pop("ipu_dataloader_inference_opts", None)

    if get_accelerator(config) == "ipu":
        ipu_inference_opts, ipu_training_opts = load_ipu_options(ipu_file=ipu_file, seed=config["constants"]["seed"])

        bz_train = cfg_data["batch_size_train_val"]
        ipu_dataloader_training_opts = IPUDataloaderOptions(batch_size=bz_train, **ipu_dataloader_training_opts)
        ipu_dataloader_training_opts.set_kwargs()

        bz_test = cfg_data["batch_size_test"]
        ipu_dataloader_inference_opts = IPUDataloaderOptions(batch_size=bz_test, **ipu_dataloader_inference_opts)
        ipu_dataloader_inference_opts.set_kwargs()


    module_class = DATAMODULE_DICT[config["datamodule"]["module_type"]]
    datamodule = module_class(
                ipu_inference_opts = ipu_inference_opts,
                ipu_training_opts = ipu_training_opts,
                ipu_dataloader_training_opts=ipu_dataloader_training_opts,
                ipu_dataloader_inference_opts=ipu_dataloader_inference_opts,
                **config["datamodule"]["args"])

    return datamodule


def load_metrics(config: Union[omegaconf.DictConfig, Dict[str, Any]]):

    task_metrics = {}
    cfg_metrics = deepcopy(config["metrics"])
    if cfg_metrics is None:
        return task_metrics

    for task in cfg_metrics:
        task_metrics[task] = {}
        if cfg_metrics[task] is None:
            cfg_metrics[task] = []
        for this_metric in cfg_metrics[task]:
            name = this_metric.pop("name")
            task_metrics[task][name] = MetricWrapper(**this_metric)

    return task_metrics


def load_architecture(
    config: Union[omegaconf.DictConfig, Dict[str, Any]],
    in_dim_nodes: int,
    in_dim_edges: int,
):

    if isinstance(config, dict):
        config = omegaconf.OmegaConf.create(config)
    cfg_arch = config["architecture"]

    kwargs = {}

    # Select the architecture
    model_type = cfg_arch["model_type"].lower()
    if model_type == "fulldglnetwork":
        model_class = FullGraphNetwork
    elif model_type == "fulldglsiamesenetwork":
        model_class = FullGraphSiameseNetwork
        kwargs["dist_method"] = cfg_arch["dist_method"]
    elif model_type == "fullgraphmultitasknetwork":
        model_class = FullGraphMultiTaskNetwork
    else:
        raise ValueError(f"Unsupported model_type=`{model_type}`")

    # Prepare the various kwargs
    pre_nn_kwargs = dict(cfg_arch["pre_nn"]) if cfg_arch["pre_nn"] is not None else None
    pre_nn_edges_kwargs = dict(cfg_arch["pre_nn_edges"]) if cfg_arch["pre_nn_edges"] is not None else None
    gnn_kwargs = dict(cfg_arch["gnn"])
    post_nn_kwargs = dict(cfg_arch["post_nn"]) if cfg_arch["post_nn"] is not None else None
    task_heads_kwargs = cfg_arch["task_heads"] if cfg_arch["task_heads"] is not None else None     # This is of type ListConfig containing TaskHeadParams

    # Set the input dimensions
    if pre_nn_kwargs is not None:
        pre_nn_kwargs = dict(pre_nn_kwargs)
        pre_nn_kwargs.setdefault("in_dim", in_dim_nodes)
    else:
        gnn_kwargs.setdefault("in_dim", in_dim_nodes)

    if pre_nn_edges_kwargs is not None:
        pre_nn_edges_kwargs = dict(pre_nn_edges_kwargs)
        pre_nn_edges_kwargs.setdefault("in_dim", in_dim_edges)
    else:
        gnn_kwargs.setdefault("in_dim_edges", in_dim_edges)

    # Set the parameters for the full network
    if task_heads_kwargs is None:
        model_kwargs = dict(
            gnn_kwargs=gnn_kwargs,
            pre_nn_kwargs=pre_nn_kwargs,
            pre_nn_edges_kwargs=pre_nn_edges_kwargs,
            post_nn_kwargs=post_nn_kwargs,
        )
    else:
        task_head_params_list = []
        for params in omegaconf.OmegaConf.to_object(task_heads_kwargs): # This turns the ListConfig into List[TaskHeadParams]
            params_dict = dict(params)
            task_head_params_list.append(params_dict)

        model_kwargs = dict(
            gnn_kwargs=gnn_kwargs,
            pre_nn_kwargs=pre_nn_kwargs,
            pre_nn_edges_kwargs=pre_nn_edges_kwargs,
            post_nn_kwargs=post_nn_kwargs,
            task_heads_kwargs_list=task_head_params_list,
        )

    return model_class, model_kwargs


def load_predictor(config, model_class, model_kwargs, metrics):
    # Defining the predictor

    if get_accelerator(config) == "ipu":
        predictor_class = PredictorModuleIPU
    else:
        predictor_class = PredictorModule

    cfg_pred = dict(deepcopy(config["predictor"]))
    predictor = predictor_class(
        model_class=model_class,
        model_kwargs=model_kwargs,
        metrics=metrics,
        **cfg_pred,
    )

    return predictor


def load_trainer(config, run_name):
    cfg_trainer = deepcopy(config["trainer"])

    # Define the IPU plugin if required
    plugins = []
    accelerator = get_accelerator(config)
    ipu_file = "tests/mtl/ipu.config"
    if accelerator == "ipu":
        training_opts, inference_opts = load_ipu_options(ipu_file=ipu_file, seed=config["constants"]["seed"])
        plugins = IPUPluginGoli(training_opts=training_opts, inference_opts=inference_opts)

    # Set the number of gpus to 0 if no GPU is available
    _ = cfg_trainer["trainer"].pop("accelerator", None)
    gpus = cfg_trainer["trainer"].pop("gpus", None)
    ipus = cfg_trainer["trainer"].pop("ipus", None)
    if (accelerator == "gpu") and (gpus is None):
        gpus = 1
    if (accelerator == "ipu") and (ipus is None):
        ipus = 1
    if accelerator != "gpu":
        gpus = 0
    if accelerator != "ipu":
        ipus = 0

    trainer_kwargs = {}
    callbacks = []
    if "early_stopping" in cfg_trainer.keys():
        callbacks.append(EarlyStopping(**cfg_trainer["early_stopping"]))

    if "model_checkpoint" in cfg_trainer.keys():
        callbacks.append(ModelCheckpoint(**cfg_trainer["model_checkpoint"]))

    if "logger" in cfg_trainer.keys():
        # WandB logger (decomment to log runs)
        wandb_logger = WandbLoggerGoli(name=run_name, project="multitask-gnn")
        trainer_kwargs["logger"] = wandb_logger

    trainer_kwargs["callbacks"] = callbacks

    trainer = Trainer(
        terminate_on_nan=True,
        plugins=plugins,
        accelerator=accelerator,
        ipus = ipus,
        gpus = gpus,
        **cfg_trainer["trainer"],
        **trainer_kwargs,
    )

    return trainer

