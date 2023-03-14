from typing import Dict, Mapping, Type, Union, Any

# Misc
import os
import omegaconf
from copy import deepcopy
from loguru import logger
import yaml
import joblib
import pathlib
import warnings

# Torch
import torch
import mup

# Lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, Logger

# Goli
from goli.utils.mup import set_base_shapes
from goli.ipu.ipu_dataloader import IPUDataloaderOptions
from goli.trainer.metrics import MetricWrapper
from goli.nn.architectures import FullGraphNetwork, FullGraphMultiTaskNetwork
from goli.trainer.predictor import PredictorModule
from goli.utils.spaces import DATAMODULE_DICT
from goli.ipu.ipu_utils import import_poptorch, load_ipu_options
from goli.data.datamodule import MultitaskFromSmilesDataModule
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


def load_datamodule(config: Union[omegaconf.DictConfig, Dict[str, Any]]) -> BaseDataModule:
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

    # Instanciate the datamodule
    module_class = DATAMODULE_DICT[config["datamodule"]["module_type"]]

    if get_accelerator(config) != "ipu":
        datamodule = module_class(
            **config["datamodule"]["args"],
        )
        return datamodule

    # IPU specific adjustments
    else:
        # Default empty values for the IPU configurations
        ipu_training_opts, ipu_inference_opts = None, None
        ipu_training_config_path = "expts/configs/ipu.config"
        if pathlib.Path(ipu_training_config_path).is_file():
            ipu_training_config_file = ipu_training_config_path
        else:
            raise ValueError(
                f"IPU configuration path must be specified "
                "and must be a file, instead got "
                '"{ipu_training_config_path}"'
            )

        ipu_inference_config_overrides_path = "expts/configs/ipu_inference.config"
        if pathlib.Path(ipu_inference_config_overrides_path).is_file():
            ipu_inference_config_overrides_file = ipu_inference_config_overrides_path
        else:
            warnings.warn(
                "IPU inference overrides configuration either not specified "
                "or not a file, using same options for training and inference"
            )
            ipu_inference_config_overrides_file = None

        ipu_dataloader_training_opts = cfg_data.pop("ipu_dataloader_training_opts", {})
        ipu_dataloader_inference_opts = cfg_data.pop("ipu_dataloader_inference_opts", {})
        ipu_training_opts, ipu_inference_opts = load_ipu_options(
            ipu_file=ipu_training_config_file,
            ipu_inference_overrides=ipu_inference_config_overrides_file,
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
    task_heads_kwargs = omegaconf.OmegaConf.to_object(task_heads_kwargs)

    accelerator_kwargs = (
        dict(cfg_arch["accelerator_options"])
        if cfg_arch.get("accelerator_options", None) is not None
        else None
    )

    if accelerator_kwargs is not None:
        accelerator_kwargs["_accelerator"] = get_accelerator(config)

    # Set all the input arguments for the model
    model_kwargs = dict(
        gnn_kwargs=gnn_kwargs,
        pre_nn_kwargs=pre_nn_kwargs,
        pre_nn_edges_kwargs=pre_nn_edges_kwargs,
        pe_encoders_kwargs=pe_encoders_kwargs,
        post_nn_kwargs=post_nn_kwargs,
        task_heads_kwargs=task_heads_kwargs,
        accelerator_kwargs=accelerator_kwargs,
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
        from goli.ipu.ipu_wrapper import PredictorModuleIPU

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

    # mup base shapes
    mup_base_path = config["architecture"].pop("mup_base_path", None)
    predictor = load_mup(mup_base_path, predictor)

    return predictor


def load_mup(mup_base_path: str, predictor: PredictorModule) -> PredictorModule:
    """
    Load the base shapes for the mup, based either on a `.ckpt` or `.yaml` file.
    If `.yaml`, it should be generated by `mup.save_base_shapes`
    """
    if mup_base_path is None:
        model = predictor.model
        base = model.__class__(**model.make_mup_base_kwargs(divide_factor=2))
    elif mup_base_path.endswith(".ckpt"):
        base = predictor.__class__.load_from_checkpoint(mup_base_path, map_location="cpu")
    elif mup_base_path.endswith(".yaml"):
        base = mup_base_path
    else:
        raise ValueError(f"Unrecognized file type {mup_base_path}")
    predictor.model = set_base_shapes(predictor.model, base, rescale_params=False)
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
            precision=cfg_trainer["trainer"].get("precision", None),
        )
        from goli.ipu.ipu_wrapper import DictIPUStrategy

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
        wandb_logger = WandbLogger(name=run_name, project="multitask-gnn")
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


def save_params_to_wandb(
    logger: Logger,
    config: Union[omegaconf.DictConfig, Dict[str, Any]],
    predictor: PredictorModule,
    datamodule: MultitaskFromSmilesDataModule,
):
    """
    Save a few stuff to weights-and-biases WandB
    Parameters:
        logger: The object used to log the training. Usually WandbLogger
        config: The config file, with key `trainer`
        predictor: The predictor used to handle the train/val/test steps logic
        datamodule: The datamodule used to load the data into training
    """

    # Save the mup base model to WandB as a yaml file
    mup.save_base_shapes(predictor.model, "mup_base_params.yaml")

    # Save the full configs as a YAML file
    with open(os.path.join(logger.experiment.dir, "full_configs.yaml"), "w") as file:
        yaml.dump(config, file)

    # Save the featurizer into wandb
    featurizer_path = os.path.join(logger.experiment.dir, "featurizer.pickle")
    joblib.dump(datamodule.smiles_transformer, featurizer_path)

    wandb_run = logger.experiment
    if wandb_run is not None:
        wandb_run.save("*.yaml")
        wandb_run.save("*.pickle")
