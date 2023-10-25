# Misc
import os
from copy import deepcopy
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, Union

import joblib
import mup
import omegaconf

# Torch
import torch
import yaml

# Lightning
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import Logger, WandbLogger
from loguru import logger

from graphium.data.datamodule import BaseDataModule, MultitaskFromSmilesDataModule
from graphium.finetuning.finetuning_architecture import FullGraphFinetuningNetwork
from graphium.ipu.ipu_dataloader import IPUDataloaderOptions
from graphium.ipu.ipu_utils import import_poptorch, load_ipu_options
from graphium.nn.architectures import FullGraphMultiTaskNetwork
from graphium.nn.utils import MupMixin
from graphium.trainer.metrics import MetricWrapper
from graphium.trainer.predictor import PredictorModule
from graphium.utils.command_line_utils import get_anchors_and_aliases, update_config

# Graphium
from graphium.utils.mup import set_base_shapes
from graphium.utils.spaces import DATAMODULE_DICT, GRAPHIUM_PRETRAINED_MODELS_DICT
from graphium.utils import fs


def get_accelerator(
    config_acc: Union[omegaconf.DictConfig, Dict[str, Any]],
) -> str:
    """
    Get the accelerator from the config file, and ensure that they are
    consistant.
    """

    # Get the accelerator type
    accelerator_type = config_acc["type"]

    # Get the GPU info
    if (accelerator_type == "gpu") and (not torch.cuda.is_available()):
        raise ValueError(f"GPUs selected, but GPUs are not available on this device")

    # Get the IPU info
    if accelerator_type == "ipu":
        poptorch = import_poptorch()
        if poptorch is None:
            raise ValueError("IPUs selected, but PopTorch is not available")
        if not poptorch.ipuHardwareIsAvailable():
            raise ValueError(
                "IPUs selected, but no IPU is available/visible on this device. "
                "If you do have IPUs, please check that the IPUOF_VIPU_API_PARTITION_ID and "
                "IPUOF_VIPU_API_HOST environment variables are set."
            )

    # Fall on cpu at the end
    if accelerator_type is None:
        accelerator_type = "cpu"
    return accelerator_type


def _get_ipu_opts(config: Union[omegaconf.DictConfig, Dict[str, Any]]) -> Tuple[str, str]:
    r"""
    Get the paths of the IPU-specific config files from the main YAML config
    """

    accelerator_options = config["accelerator"]
    accelerator_type = accelerator_options["type"]

    if accelerator_type != "ipu":
        return None, None
    ipu_opts = accelerator_options["ipu_config"]
    ipu_inference_opts = accelerator_options.get("ipu_inference_config", None)

    return ipu_opts, ipu_inference_opts


def load_datamodule(
    config: Union[omegaconf.DictConfig, Dict[str, Any]], accelerator_type: str
) -> BaseDataModule:
    """
    Load the datamodule from the specified configurations at the key
    `datamodule: args`.
    If the accelerator is IPU, load the IPU options as well.

    Parameters:
        config: The config file, with key `datamodule: args`
        accelerator_type: The accelerator type, e.g. "cpu", "gpu", "ipu"
    Returns:
        datamodule: The datamodule used to process and load the data
    """

    cfg_data = config["datamodule"]["args"]

    # Instanciate the datamodule
    module_class = DATAMODULE_DICT[config["datamodule"]["module_type"]]

    if accelerator_type != "ipu":
        datamodule = module_class(
            **config["datamodule"]["args"],
        )
        return datamodule

    # IPU specific adjustments
    else:
        ipu_opts, ipu_inference_opts = _get_ipu_opts(config)

        # Default empty values for the IPU configurations
        ipu_training_opts = None

        ipu_dataloader_training_opts = cfg_data.pop("ipu_dataloader_training_opts", {})
        ipu_dataloader_inference_opts = cfg_data.pop("ipu_dataloader_inference_opts", {})
        ipu_training_opts, ipu_inference_opts = load_ipu_options(
            ipu_opts=ipu_opts,
            seed=config["constants"]["seed"],
            model_name=config["constants"]["name"],
            gradient_accumulation=config["trainer"]["trainer"].get("accumulate_grad_batches", None),
            ipu_inference_opts=ipu_inference_opts,
            precision=config["trainer"]["trainer"].get("precision"),
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
    cfg_metrics = config.get("metrics", None)
    if cfg_metrics is None:
        return task_metrics
    cfg_metrics = {key: deepcopy(value) for key, value in config["metrics"].items()}
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
) -> Union[FullGraphMultiTaskNetwork, torch.nn.Module]:
    """
    Loading the architecture used for training.
    Parameters:
        config: The config file, with key `architecture`
        in_dims: Dictionary of the input dimensions for various
    Returns:
        architecture: The datamodule used to process and load the data
    """

    if isinstance(config, dict) and "finetuning" not in config:
        config = omegaconf.OmegaConf.create(config)
    cfg_arch = config["architecture"]

    # Select the architecture
    model_type = cfg_arch["model_type"].lower()
    if model_type == "fullgraphmultitasknetwork":
        model_class = FullGraphMultiTaskNetwork
    elif model_type == "fullgraphfinetuningnetwork":
        model_class = FullGraphFinetuningNetwork
    else:
        raise ValueError(f"Unsupported model_type=`{model_type}`")

    # Prepare the various kwargs
    pe_encoders_kwargs = (
        dict(cfg_arch["pe_encoders"]) if cfg_arch.get("pe_encoders", None) is not None else None
    )

    pre_nn_kwargs = dict(cfg_arch["pre_nn"]) if cfg_arch["pre_nn"] is not None else None
    pre_nn_edges_kwargs = dict(cfg_arch["pre_nn_edges"]) if cfg_arch["pre_nn_edges"] is not None else None
    gnn_kwargs = dict(cfg_arch["gnn"])
    graph_output_nn_kwargs = (
        dict(cfg_arch["graph_output_nn"]) if cfg_arch["graph_output_nn"] is not None else None
    )
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
    pe_out_dim = 0 if pe_encoders_kwargs is None else pe_encoders_kwargs.get("out_dim", None)
    edge_pe_out_dim = 0 if pe_encoders_kwargs is None else pe_encoders_kwargs.get("edge_out_dim", None)

    # Set the default `node` input dimension for the pre-processing neural net and graph neural net
    in_dim = in_dims["feat"]
    if pe_out_dim is not None:
        in_dim += pe_out_dim
    if pre_nn_kwargs is not None:
        pre_nn_kwargs = dict(pre_nn_kwargs)
        pre_nn_kwargs.setdefault("in_dim", in_dim)
    else:
        gnn_kwargs.setdefault("in_dim", in_dim)

    # Set the default `edge` input dimension for the pre-processing neural net and graph neural net
    edge_in_dim = in_dims["edge_feat"]
    if edge_pe_out_dim is not None:
        edge_in_dim += edge_pe_out_dim
    if pre_nn_edges_kwargs is not None:
        pre_nn_edges_kwargs = dict(pre_nn_edges_kwargs)
        pre_nn_edges_kwargs.setdefault("in_dim", edge_in_dim)
    else:
        gnn_kwargs.setdefault("in_dim", edge_in_dim)

    # Set the parameters for the full network
    if "finetuning" not in config:
        task_heads_kwargs = omegaconf.OmegaConf.to_object(task_heads_kwargs)

    # Set all the input arguments for the model
    model_kwargs = dict(
        gnn_kwargs=gnn_kwargs,
        pre_nn_kwargs=pre_nn_kwargs,
        pre_nn_edges_kwargs=pre_nn_edges_kwargs,
        pe_encoders_kwargs=pe_encoders_kwargs,
        graph_output_nn_kwargs=graph_output_nn_kwargs,
        task_heads_kwargs=task_heads_kwargs,
    )
    # Get accelerator_kwargs if they exist
    accelerator_kwargs = config["accelerator"].get("accelerator_kwargs", None)
    if accelerator_kwargs is not None:
        model_kwargs["accelerator_kwargs"] = accelerator_kwargs

    if model_class is FullGraphFinetuningNetwork:
        finetuning_head_kwargs = config["finetuning"].pop("finetuning_head", None)
        pretrained_overwriting_kwargs = config["finetuning"].pop("overwriting_kwargs")
        pretrained_model = pretrained_overwriting_kwargs.pop("pretrained_model")

        model_kwargs = {
            "pretrained_model_kwargs": deepcopy(model_kwargs),
            "pretrained_overwriting_kwargs": pretrained_overwriting_kwargs,
            "pretrained_model": pretrained_model,
            "finetuning_head_kwargs": finetuning_head_kwargs,
        }

    return model_class, model_kwargs


def load_predictor(
    config: Union[omegaconf.DictConfig, Dict[str, Any]],
    model_class: Type[torch.nn.Module],
    model_kwargs: Dict[str, Any],
    metrics: Dict[str, MetricWrapper],
    task_levels: Dict[str, str],
    accelerator_type: str,
    featurization: Dict[str, str] = None,
    task_norms: Optional[Dict[Callable, Any]] = None,
    replicas: int = 1,
    gradient_acc: int = 1,
    global_bs: int = 1,
) -> PredictorModule:
    """
    Defining the predictor module, which handles the training logic from `lightning.LighningModule`
    Parameters:
        model_class: The torch Module containing the main forward function
        accelerator_type: The accelerator type, e.g. "cpu", "gpu", "ipu"
    Returns:
        predictor: The predictor module
    """

    if accelerator_type == "ipu":
        from graphium.ipu.ipu_wrapper import PredictorModuleIPU

        predictor_class = PredictorModuleIPU
    else:
        predictor_class = PredictorModule

    cfg_pred = dict(deepcopy(config["predictor"]))
    predictor = predictor_class(
        model_class=model_class,
        model_kwargs=model_kwargs,
        metrics=metrics,
        task_levels=task_levels,
        featurization=featurization,
        task_norms=task_norms,
        replicas=replicas,
        gradient_acc=gradient_acc,
        global_bs=global_bs,
        **cfg_pred,
    )

    mup_scale_factor = config["architecture"].pop("mup_scale_factor", None)

    if mup_scale_factor is not None and mup_scale_factor != 1:
        unscaled_model = predictor.model
        scaled_model_kwargs = unscaled_model.scale_kwargs(scale_factor=mup_scale_factor)
        del predictor
        predictor = predictor_class(
            model_class=model_class,
            model_kwargs=scaled_model_kwargs,
            metrics=metrics,
            task_levels=task_levels,
            featurization=featurization,
            task_norms=task_norms,
            replicas=replicas,
            gradient_acc=gradient_acc,
            global_bs=global_bs,
            **cfg_pred,
        )

    # mup base shapes
    mup_base_path = config["architecture"].pop("mup_base_path", None)
    predictor = load_mup(mup_base_path, predictor)

    return predictor


def load_mup(mup_base_path: str, predictor: PredictorModule) -> PredictorModule:
    """
    Load the base shapes for the mup, based either on a `.ckpt` or `.yaml` file.
    If `.yaml`, it should be generated by `mup.save_base_shapes`
    """
    model = predictor.model

    if not isinstance(model, MupMixin):
        raise TypeError("load_mup can only be applied to models that use the MupMixin")

    if mup_base_path is None:
        base = model.__class__(**model.make_mup_base_kwargs(divide_factor=2))
    elif mup_base_path.endswith(".ckpt"):
        base = predictor.__class__.load_from_checkpoint(mup_base_path, map_location="cpu")
    elif mup_base_path.endswith(".yaml"):
        base = mup_base_path
    else:
        raise ValueError(f"Unrecognized file type {mup_base_path}")
    predictor.model = set_base_shapes(predictor.model, base, rescale_params=False)
    return predictor


def load_trainer(
    config: Union[omegaconf.DictConfig, Dict[str, Any]],
    accelerator_type: str,
    date_time_suffix: str = "",
) -> Trainer:
    """
    Defining the pytorch-lightning Trainer module.
    Parameters:
        config: The config file, with key `trainer`
        accelerator_type: The accelerator type, e.g. "cpu", "gpu", "ipu"
        date_time_suffix: The date and time of the current run. To be used for logging.
    Returns:
        trainer: the trainer module
    """
    cfg_trainer = deepcopy(config["trainer"])

    # Define the IPU plugin if required
    strategy = cfg_trainer["trainer"].pop("strategy", "auto")
    if accelerator_type == "ipu":
        ipu_opts, ipu_inference_opts = _get_ipu_opts(config)

        training_opts, inference_opts = load_ipu_options(
            ipu_opts=ipu_opts,
            ipu_inference_opts=ipu_inference_opts,
            seed=config["constants"]["seed"],
            model_name=config["constants"]["name"],
            gradient_accumulation=config["trainer"]["trainer"].get("accumulate_grad_batches", None),
            precision=config["trainer"]["trainer"].get("precision"),
        )

        if strategy != "auto":
            raise ValueError("IPUs selected, but strategy is not set to 'auto'")

        from lightning_graphcore import IPUStrategy

        strategy = IPUStrategy(training_opts=training_opts, inference_opts=inference_opts)

    # Get devices
    devices = cfg_trainer["trainer"].pop("devices", 1)
    if accelerator_type == "ipu":
        devices = 1  # number of IPUs used is defined in the ipu options files

    # Remove the gradient accumulation from IPUs, since it's handled by the device
    if accelerator_type == "ipu":
        cfg_trainer["trainer"].pop("accumulate_grad_batches", None)

    # Define the early stopping parameters
    trainer_kwargs = {}
    callbacks = []
    if "early_stopping" in cfg_trainer.keys():
        callbacks.append(EarlyStopping(**cfg_trainer["early_stopping"]))

    # Define the early model checkpoing parameters
    if "model_checkpoint" in cfg_trainer.keys():
        callbacks.append(ModelCheckpoint(**cfg_trainer["model_checkpoint"]))

    if "learning_rate_monitor" in cfg_trainer.keys():
        callbacks.append(LearningRateMonitor(**cfg_trainer["learning_rate_monitor"]))
    else:
        callbacks.append(LearningRateMonitor())

    # Define the logger parameters
    wandb_cfg = config["constants"].get("wandb")
    if wandb_cfg is not None:
        name = wandb_cfg.pop("name", "main")
        if len(date_time_suffix) > 0:
            name += f"_{date_time_suffix}"
        trainer_kwargs["logger"] = WandbLogger(name=name, log_model=True, **wandb_cfg)

    trainer_kwargs["callbacks"] = callbacks
    trainer = Trainer(
        detect_anomaly=True,
        strategy=strategy,
        accelerator=accelerator_type,
        devices=devices,
        **cfg_trainer["trainer"],
        **trainer_kwargs,
    )
    return trainer


def save_params_to_wandb(
    logger: Logger,
    config: Union[omegaconf.DictConfig, Dict[str, Any]],
    predictor: PredictorModule,
    datamodule: MultitaskFromSmilesDataModule,
    unresolved_config: Optional[Union[omegaconf.DictConfig, Dict[str, Any]]] = None,
):
    """
    Save a few stuff to weights-and-biases WandB
    Parameters:
        logger: The object used to log the training. Usually WandbLogger
        config: The config file, with key `trainer`
        predictor: The predictor used to handle the train/val/test steps logic
        datamodule: The datamodule used to load the data into training
        unresolved_config: The unresolved config file
    """

    # Get the wandb runner and directory
    wandb_run = logger.experiment

    if wandb_run is None:
        wandb_dir = ""
    else:
        wandb_dir = wandb_run.dir

    # Save the mup base model to WandB as a yaml file
    mup.save_base_shapes(predictor.model, os.path.join(wandb_dir, "mup_base_params.yaml"))

    # Save the full configs as a YAML file
    with open(os.path.join(wandb_dir, "full_configs.yaml"), "w") as file:
        yaml.dump(config, file)

    if unresolved_config is not None:
        with open(os.path.join(wandb_dir, "unresolved_config.yaml"), "w") as file:
            yaml.dump(unresolved_config, file)

    # Save the featurizer into wandb
    featurizer_path = os.path.join(wandb_dir, "featurizer.pickle")
    joblib.dump(datamodule.smiles_transformer, featurizer_path)

    # Save the featurizer and configs into wandb
    if wandb_run is not None:
        wandb_run.save(os.path.join(wandb_dir, "*.yaml"), wandb_dir)
        wandb_run.save(os.path.join(wandb_dir, "*.pickle"), wandb_dir)


def load_accelerator(config: Union[omegaconf.DictConfig, Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    config = deepcopy(config)
    config_acc = config.get("accelerator", {})

    # Merge the accelerator config with the main config
    config_override = config_acc.get("config_override", {})
    merge_dicts(config, config_override)
    accelerator_type = get_accelerator(config_acc)

    if accelerator_type == "gpu":
        precision = config_acc.get("float32_matmul_precision", None)
        if precision is not None:
            torch.set_float32_matmul_precision(precision)

    return config, accelerator_type


def load_config_override(
    config: Union[omegaconf.DictConfig, Dict[str, Any]], main_dir: Optional[Union[str, os.PathLike]] = None
) -> Dict[str, Any]:
    config = deepcopy(config)
    config_override_path = config["constants"].get("config_override", None)
    if config_override_path is not None:
        if main_dir is not None:
            config_override_path = os.path.join(main_dir, config_override_path)
        with open(config_override_path, "r") as f:
            cfg_override = yaml.safe_load(f)
        config = merge_dicts(cfg_override, config, on_exist="overwrite")
    return config


def load_yaml_config(
    config_path: Union[str, os.PathLike],
    main_dir: Optional[Union[str, os.PathLike]] = None,
    unknown_args=None,
) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a dictionary.
    Also returns the anchors `&` and aliases `*` of the YAML file.
    Then, update the config with the unknown arguments.
    Finally, update the config with the config override file specified in `constants.config_override`.

    Parameters:
        config_path: The path to the YAML config file
        main_dir: The main directory of the project. If specified, the config override file will be loaded from this directory
        unknown_args: The unknown arguments to update the config with, taken from `argparse.parse_known_args`

    Returns:
        config: The config dictionary

    """
    if main_dir is not None:
        config_path = os.path.join(main_dir, config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        refs = get_anchors_and_aliases(config_path)
        if unknown_args is not None:
            config = update_config(config, unknown_args, refs)
    config = load_config_override(config, main_dir)  # This goes here to avoid overriding the hparam search

    return config


def merge_dicts(
    dict_a: Dict[str, Any], dict_b: Dict[str, Any], previous_dict_path: str = "", on_exist: str = "raise"
) -> None:
    """
    Recursively merges dict_b into dict_a. If a key is missing from dict_a,
    it is added from dict_b. If a key exists in both, an error is raised.
    `dict_a` is modified in-place.

    Parameters:
        dict_a: The dictionary to merge into. Modified in-place.
        dict_b: The dictionary to merge from.
        previous_dict_path: The key path of the parent dictionary,
        used to track the recursive calls.
        on_exist: What to do if a key already exists in dict_a. Options are "raise", "overwrite", "ignore".

    Raises:
        ValueError: If a key path already exists in dict_a.

    """
    assert on_exist in [
        "raise",
        "overwrite",
        "ignore",
    ], f"on_exist must be one of ['raise', 'overwrite', 'ignore'], got {on_exist}"

    for key, value_b in dict_b.items():
        if key not in dict_a:
            dict_a[key] = value_b
        else:
            value_a = dict_a[key]
            if previous_dict_path == "":
                previous_dict_path = key
            else:
                previous_dict_path = f"{previous_dict_path}/{key}"
            if isinstance(value_a, dict) and isinstance(value_b, dict):
                merge_dicts(value_a, value_b, previous_dict_path=previous_dict_path, on_exist=on_exist)
            else:
                if value_a != value_b:
                    if on_exist == "raise":
                        raise ValueError(f"Dict path already exists: {previous_dict_path}")
                    elif on_exist == "overwrite":
                        dict_a[key] = value_b
                    elif on_exist == "ignore":
                        pass
    return dict_a


def get_checkpoint_path(config: Union[omegaconf.DictConfig, Dict[str, Any]]) -> str:
    """
    Get the checkpoint path from a config file.
    If the path is a valid name or a valid path, return it.
    Otherwise, assume it refers to a file in the checkpointing dir.
    """

    cfg_trainer = config["trainer"]

    path = config.get("ckpt_name_for_testing", "last.ckpt")
    if path in GRAPHIUM_PRETRAINED_MODELS_DICT or fs.exists(path):
        return path

    if "model_checkpoint" in cfg_trainer.keys():
        dirpath = cfg_trainer["model_checkpoint"]["dirpath"]
        path = fs.join(dirpath, path)

    if not fs.exists(path):
        raise ValueError(f"Checkpoint path `{path}` does not exist")

    return path
