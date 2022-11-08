from typing import Callable, Dict, Mapping, Type, Union, Any, Optional, List, Tuple

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
from goli.ipu.ipu_wrapper import PredictorModuleIPU, DictIPUStrategy
from goli.ipu.ipu_utils import import_poptorch, load_ipu_options
from goli.trainer.loggers import WandbLoggerGoli
from goli.data.datamodule import BaseDataModule

# Weights and Biases
from pytorch_lightning import Trainer


def get_accelerator(
    config: Union[omegaconf.DictConfig, Dict[str, Any]],
) -> str:
    """
    Get the accelerator from the config file, and ensure that they are
    consistant. For example, specifying `cpu` as the accelerators, but
    `gpus>0` as a Trainer option will yield an error.
    """

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
        logger.warning(f"GPUs selected, but will be ignored since no GPU are available on this device")
        acc_type = "cpu"

    # Get the IPU info
    ipus = config["trainer"]["trainer"].get("ipus", 0)
    if ipus > 0:
        assert (acc_type is None) or (acc_type == "ipu"), "Accelerator mismatch"
        acc_type = "ipu"
    if acc_type == "ipu":
        poptorch = import_poptorch()
        if not poptorch.ipuHardwareIsAvailable():
            logger.warning(f"IPUs selected, but will be ignored since no IPU are available on this device")
            acc_type = "cpu"

    # Fall on cpu at the end
    if acc_type is None:
        acc_type = "cpu"
    return acc_type


def get_max_num_nodes_edges_datamodule(datamodule: BaseDataModule, stages: Optional[List[str]] = None) -> Tuple[int, int]:
    """
    Get the maximum number of nodes and edges across all datasets from the datamodule

    Parameters:
        datamodule: The datamodule from which to extract the maximum number of nodes
        stages: The stages from which to extract the max num nodes.
            Possible values are ["train", "val", "test", "predict"].
            If None, all stages are considered.

    Returns:
        max_num_nodes: The maximum number of nodes across all datasets from the datamodule
        max_num_edges: The maximum number of edges across all datasets from the datamodule
    """

    allowed_stages = ["train", "val", "test", "predict"]
    if stages is None:
        stages = allowed_stages
    for stage in stages:
        assert stage in allowed_stages, f"stage value `{stage}` not allowed."

    max_nodes, max_edges = [], []
    # Max number of nodes/edges in the training dataset
    if (datamodule.train_ds is not None) and ("train" in stages):
        max_nodes.append(datamodule.train_ds.max_num_nodes_per_graph)
        max_edges.append(datamodule.train_ds.max_num_edges_per_graph)

    # Max number of nodes/edges in the validation dataset
    if (datamodule.val_ds is not None) and ("train" in stages):
        max_nodes.append(datamodule.val_ds.max_num_nodes_per_graph)
        max_edges.append(datamodule.val_ds.max_num_edges_per_graph)

    # Max number of nodes/edges in the test dataset
    if (datamodule.test_ds is not None) and ("train" in stages):
        max_nodes.append(datamodule.test_ds.max_num_nodes_per_graph)
        max_edges.append(datamodule.test_ds.max_num_edges_per_graph)

    # Max number of nodes/edges in the predict dataset
    if (datamodule.predict_ds is not None) and ("train" in stages):
        max_nodes.append(datamodule.predict_ds.max_num_nodes_per_graph)
        max_edges.append(datamodule.predict_ds.max_num_edges_per_graph)

    max_num_nodes = max(max_nodes)
    max_num_edges = max(max_edges)
    return max_num_nodes, max_num_edges


def load_datamodule(config: Union[omegaconf.DictConfig, Dict[str, Any]]) -> "goli.datamodule.BaseDataModule":
    """
    Load the datamodule from the specified configurations at the key
    `datamodule: args`.
    If the accelerator is IPU, load the IPU options as well.

    Parameters:
        config: The config file, with key `datamodule: args`
    Returns:
        datamodule: The datamodule used to process and load the data
    """

    cfg_data = config["datamodule"]["args"]

    # Default empty values for the IPU configurations
    ipu_training_opts, ipu_inference_opts = None, None
    ipu_file = "expts/configs/ipu.config"
    ipu_dataloader_training_opts = cfg_data.pop("ipu_dataloader_training_opts", {})
    ipu_dataloader_inference_opts = cfg_data.pop("ipu_dataloader_inference_opts", {})

    if get_accelerator(config) == "ipu":
        ipu_training_opts, ipu_inference_opts = load_ipu_options(
            ipu_file=ipu_file,
            seed=config["constants"]["seed"],
            model_name=config["constants"]["name"],
            gradient_accumulation=config["trainer"]["trainer"].get("accumulate_grad_batches", None),
        )

        # Define the Dataloader options for the IPU on the training sets
        bz_train = cfg_data["batch_size_training"]
        ipu_dataloader_training_opts = IPUDataloaderOptions(
            batch_size=bz_train, **ipu_dataloader_training_opts
        )
        ipu_dataloader_training_opts.set_kwargs()

        # Define the Dataloader options for the IPU on the inference sets
        bz_test = cfg_data["batch_size_inference"]
        ipu_dataloader_inference_opts = IPUDataloaderOptions(
            batch_size=bz_test, **ipu_dataloader_inference_opts
        )
        ipu_dataloader_inference_opts.set_kwargs()

    # Instanciate the datamodule
    module_class = DATAMODULE_DICT[config["datamodule"]["module_type"]]
    datamodule = module_class(
        ipu_training_opts=ipu_training_opts,
        ipu_inference_opts=ipu_inference_opts,
        ipu_dataloader_training_opts=ipu_dataloader_training_opts,
        ipu_dataloader_inference_opts=ipu_dataloader_inference_opts,
        **config["datamodule"]["args"],
    )

    return datamodule


def load_metrics(config: Union[omegaconf.DictConfig, Dict[str, Any]]) -> Dict[str, MetricWrapper]:
    """
    Loading the metrics to be tracked.
    Parameters:
        config: The config file, with key `metrics`
    Returns:
        metrics: A dictionary of all the metrics
    """

    task_metrics = {}
    cfg_metrics = deepcopy(config["metrics"])
    if cfg_metrics is None:
        return task_metrics

    # Wrap every metric in the class `MetricWrapper` to standardize them
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
    in_dims: Dict[str, int],
) -> Union[FullGraphNetwork, torch.nn.Module]:
    """
    Loading the architecture used for training.
    Parameters:
        config: The config file, with key `architecture`
        in_dims: Dictionary of the input dimensions for various
    Returns:
        architecture: The datamodule used to process and load the data
    """

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
    pe_encoders_kwargs = (
        dict(cfg_arch["pe_encoders"]) if cfg_arch.get("pe_encoders", None) is not None else None
    )

    pre_nn_kwargs = dict(cfg_arch["pre_nn"]) if cfg_arch["pre_nn"] is not None else None
    pre_nn_edges_kwargs = dict(cfg_arch["pre_nn_edges"]) if cfg_arch["pre_nn_edges"] is not None else None
    gnn_kwargs = dict(cfg_arch["gnn"])
    post_nn_kwargs = dict(cfg_arch["post_nn"]) if cfg_arch["post_nn"] is not None else None
    task_heads_kwargs = (
        cfg_arch["task_heads"] if cfg_arch["task_heads"] is not None else None
    )  # This is of type ListConfig containing TaskHeadParams

    # Initialize the input dimension for the positional encoders
    if pe_encoders_kwargs is not None:
        pe_encoders_kwargs = dict(pe_encoders_kwargs)
        for encoder in pe_encoders_kwargs["encoders"]:
            pe_encoders_kwargs["encoders"][encoder] = dict(pe_encoders_kwargs["encoders"][encoder])
        pe_encoders_kwargs.setdefault(
            "in_dims", in_dims
        )  # set the input dimensions of all pe with info from the data-module
    pe_out_dim = 0 if pe_encoders_kwargs is None else pe_encoders_kwargs["out_dim"]

    # Set the default `node` input dimension for the pre-processing neural net and graph neural net
    if pre_nn_kwargs is not None:
        pre_nn_kwargs = dict(pre_nn_kwargs)
        pre_nn_kwargs.setdefault("in_dim", in_dims["feat"] + pe_out_dim)
    else:
        gnn_kwargs.setdefault("in_dim", in_dims["feat"] + pe_out_dim)

    # Set the default `edge` input dimension for the pre-processing neural net and graph neural net
    if pre_nn_edges_kwargs is not None:
        pre_nn_edges_kwargs = dict(pre_nn_edges_kwargs)
        pre_nn_edges_kwargs.setdefault("in_dim", in_dims["edge_feat"])
    else:
        gnn_kwargs.setdefault("in_dim_edges", in_dims["edge_feat"])

    # Set the parameters for the full network
    task_head_params_list = []
    for params in omegaconf.OmegaConf.to_object(
        task_heads_kwargs
    ):  # This turns the ListConfig into List[TaskHeadParams]
        params_dict = dict(params)
        task_head_params_list.append(params_dict)

    # Set all the input arguments for the model
    model_kwargs = dict(
        gnn_kwargs=gnn_kwargs,
        pre_nn_kwargs=pre_nn_kwargs,
        pre_nn_edges_kwargs=pre_nn_edges_kwargs,
        pe_encoders_kwargs=pe_encoders_kwargs,
        post_nn_kwargs=post_nn_kwargs,
        task_heads_kwargs_list=task_head_params_list,
    )

    return model_class, model_kwargs


def load_predictor(
    config: Union[omegaconf.DictConfig, Dict[str, Any]],
    model_class: Type[torch.nn.Module],
    model_kwargs: Dict[str, Any],
    metrics: Dict[str, MetricWrapper],
) -> PredictorModule:
    """
    Defining the predictor module, which handles the training logic from `pytorch_lightning.LighningModule`
    Parameters:
        model_class: The torch Module containing the main forward function
    Returns:
        predictor: The predictor module
    """

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


def load_trainer(config: Union[omegaconf.DictConfig, Dict[str, Any]], run_name: str) -> Trainer:
    """
    Defining the pytorch-lightning Trainer module.
    Parameters:
        config: The config file, with key `trainer`
        run_name: The name of the current run. To be used for logging.
    Returns:
        trainer: the trainer module
    """
    cfg_trainer = deepcopy(config["trainer"])

    # Define the IPU plugin if required
    strategy = None
    accelerator = get_accelerator(config)
    ipu_file = "expts/configs/ipu.config"
    if accelerator == "ipu":
        training_opts, inference_opts = load_ipu_options(
            ipu_file=ipu_file,
            seed=config["constants"]["seed"],
            model_name=config["constants"]["name"],
            gradient_accumulation=config["trainer"]["trainer"].get("accumulate_grad_batches", None),
        )
        strategy = DictIPUStrategy(training_opts=training_opts, inference_opts=inference_opts)

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

    # Remove the gradient accumulation from IPUs, since it's handled by the device
    if accelerator == "ipu":
        cfg_trainer["trainer"].pop("accumulate_grad_batches", None)

    # Define the early stopping parameters
    trainer_kwargs = {}
    callbacks = []
    if "early_stopping" in cfg_trainer.keys():
        callbacks.append(EarlyStopping(**cfg_trainer["early_stopping"]))

    # Define the early model checkpoing parameters
    if "model_checkpoint" in cfg_trainer.keys():
        callbacks.append(ModelCheckpoint(**cfg_trainer["model_checkpoint"]))

    # Define the logger parameters
    if "logger" in cfg_trainer.keys():
        wandb_logger = WandbLoggerGoli(name=run_name, project="multitask-gnn", full_configs=config)
        trainer_kwargs["logger"] = wandb_logger

    trainer_kwargs["callbacks"] = callbacks

    trainer = Trainer(
        detect_anomaly=True,
        strategy=strategy,
        accelerator=accelerator,
        ipus=ipus,
        gpus=gpus,
        **cfg_trainer["trainer"],
        **trainer_kwargs,
    )

    return trainer
