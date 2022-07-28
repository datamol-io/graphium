from collections import namedtuple
from turtle import forward
import poptorch

from typing import List, Dict, NamedTuple, Union, Any

import omegaconf
from copy import deepcopy
import torch
from loguru import logger

from pytorch_lightning.plugins import IPUPlugin
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from goli.nn.architectures import TaskHeadParams

from goli.trainer.metrics import MetricWrapper
from goli.nn.architectures import FullGraphNetwork, FullGraphSiameseNetwork, FullGraphMultiTaskNetwork, FeedForwardNN
# from goli.trainer.predictor import PredictorModule
from goli.trainer.refactor_predictor_mtl import PredictorModule
from goli.utils.spaces import DATAMODULE_DICT

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


def load_datamodule(
    config: Union[omegaconf.DictConfig, Dict[str, Any]],
    ipu_options=None
):
    module_class = DATAMODULE_DICT[config["datamodule"]["module_type"]]
    datamodule = module_class(**config["datamodule"]["args"])

    return datamodule


def load_metrics(config: Union[omegaconf.DictConfig, Dict[str, Any]]):

    metrics = {}
    cfg_metrics = deepcopy(config["metrics"])
    if cfg_metrics is None:
        return metrics

    for this_metric in cfg_metrics:
        name = this_metric.pop("name")
        metrics[name] = MetricWrapper(**this_metric)

    return metrics

def load_metrics_mtl(config: Union[omegaconf.DictConfig, Dict[str, Any]]):

    task_metrics = {}
    cfg_metrics = deepcopy(config["metrics"])
    if cfg_metrics is None:
        return task_metrics

    for task in cfg_metrics:
        task_metrics[task] = {}
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
    #if "task_heads" in cfg_arch: print("THE TASK HEADS: ", cfg_arch["task_heads"])
    #else: print("NO TASK HEADS")
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
            task_head_params_list.append(TaskHeadParams(**params_dict))

        model_kwargs = dict(
            gnn_kwargs=gnn_kwargs,
            pre_nn_kwargs=pre_nn_kwargs,
            pre_nn_edges_kwargs=pre_nn_edges_kwargs,
            post_nn_kwargs=post_nn_kwargs,
            task_heads_kwargs=task_head_params_list,
        )

    return model_class, model_kwargs


def load_predictor(config, model_class, model_kwargs, metrics):
    # Defining the predictor

    cfg_pred = dict(deepcopy(config["predictor"]))
    predictor = PredictorModuleIPU(
        model_class=model_class,
        model_kwargs=model_kwargs,
        metrics=metrics,
        **cfg_pred,
    )

    return predictor


def load_trainer(config, ipu_options=None):
    cfg_trainer = deepcopy(config["trainer"])

    # Set the number of gpus to 0 if no GPU is available
    gpus = cfg_trainer["trainer"].pop("gpus", 0)
    num_gpus = 0
    if isinstance(gpus, int):
        num_gpus = gpus
    elif isinstance(gpus, (list, tuple)):
        num_gpus = len(gpus)
    if (num_gpus > 0) and (not torch.cuda.is_available()):
        logger.warning(
            f"Number of GPUs selected is `{num_gpus}`, but will be ignored since no GPU are available on this device"
        )
        gpus = 0

    trainer_kwargs = {}
    callbacks = []
    if "early_stopping" in cfg_trainer.keys():
        callbacks.append(EarlyStopping(**cfg_trainer["early_stopping"]))

    if "model_checkpoint" in cfg_trainer.keys():
        callbacks.append(ModelCheckpoint(**cfg_trainer["model_checkpoint"]))

    if "logger" in cfg_trainer.keys():
        trainer_kwargs["logger"] = TensorBoardLogger(**cfg_trainer["logger"], default_hp_metric=False)

    trainer_kwargs["callbacks"] = callbacks


    #! need to add IPU options here
    '''
    trainer = pl.Trainer(
        max_epochs=1,
        progress_bar_refresh_rate=1,
        log_every_n_steps=1,
        plugins=IPUPlugin(inference_opts=options, training_opts=options)
    )
    '''
    trainer = Trainer(
        terminate_on_nan=True,
        # num_sanity_val_steps=0,
        **cfg_trainer["trainer"],
        **trainer_kwargs,
        plugins=IPUPluginGoli(inference_opts=ipu_options, training_opts=ipu_options)
    )



    # trainer = Trainer(
    #     terminate_on_nan=True,
    #     **cfg_trainer["trainer"],
    #     **trainer_kwargs,
    # )

    return trainer



from pytorch_lightning.trainer.states import RunningStage
from torch_geometric.data import Batch
from poptorch._args_parser import ArgsParser
import inspect
from poptorch import _impl
from torch import Tensor
from collections import namedtuple
from inspect import _ParameterKind
from pytorch_lightning.strategies.parallel import ParallelStrategy

def named_tuple_from_dict_batch(dict_or_batch, name):
    keys, vals = zip(*dict_or_batch.items())
    return namedtuple(name, keys)(*vals)


class IPUPluginGoli(IPUPlugin):

    def _step(self, stage: RunningStage, *args: Any, **kwargs: Any):

        # TODO: Change to named_tuple?? Easier for some stuff, but harder if a dictionary contains some tensors and some non-tensors.

        # Arguments for the loop
        new_args, all_keys = [], []
        keys_tensor_dict, keys_batch, keys_tensor, keys_others = {}, {}, {}, {}

        # Loop every argument. If a dict or pyg graph is found, split into tensors
        for data_key, data_val in args[0].items():
            if isinstance(data_val, (Dict, Batch)):
                for sub_key, sub_val in data_val.items():
                    if isinstance(sub_val, Tensor):
                        new_args.append(sub_val)
                        all_keys.append(sub_key)

                        # Append the keys for the tensors
                        if isinstance(data_val, Dict):
                            if data_key not in keys_tensor_dict:
                                keys_tensor_dict[data_key] = {}
                            keys_tensor_dict[data_key][sub_key] = len(all_keys) - 1

                        # Append the keys for the pyg Batch
                        elif isinstance(data_val, Batch):
                            if data_key not in keys_batch:
                                keys_batch[data_key] = {}
                            keys_batch[data_key][sub_key] = len(all_keys) - 1
                    else:
                        if data_key not in keys_others:
                            keys_others[data_key] = {}
                        keys_others[data_key][sub_key] = sub_val

            elif isinstance(data_val, Tensor):
                new_args.append(data_val)
                all_keys.append(data_key)
                keys_tensor[data_key] = len(all_keys) - 1
            else:
                keys_others[data_key] = data_val

        # Tell the module what are the labels associated to the labels and batches
        self.model.module._keys_tensor_dict = keys_tensor_dict
        self.model.module._keys_tensor = keys_tensor
        self.model.module._keys_batch = keys_batch
        self.model.module._keys_others = keys_others
        self.poptorch_models[stage]._args_parser._varnames = all_keys
        self.poptorch_models[stage]._args_parser._var_kinds = [_ParameterKind.VAR_POSITIONAL] * len(all_keys)

        # Run the step using only tuple of tensors
        out = super()._step(stage, *new_args, **kwargs)

        # Remove the keys from the module after the step is executed
        self.model.module._keys_tensor_dict = None
        self.model.module._keys_tensor = None
        self.model.module._keys_batch = None
        self.model.module._keys_others = None

        return out


class PredictorModuleIPU(PredictorModule):

    def forward(self, *inputs):
        # Not sure if I should keep it??
        if isinstance(inputs[0], dict):
            out_batch = super().forward(inputs[0])
            out_batch = torch.stack(tuple(out_batch["preds"].values()))
        else:
            batch = self._build_batch(*inputs)
            out_batch = super().forward(batch)
            out_batch = self._clean_output_batch(out_batch)
        return out_batch

    # def _general_step(self, batch: Dict[str, Tensor], batch_idx: int, step_name: str, to_cpu: bool) -> Dict[str, Any]:
    #     preds = self.forward(batch) # ["preds"]                    # The dictionary of predictions
    #     #targets = batch.pop("labels").to(dtype=preds.dtype)
    #     preds = {k: preds[ii] for ii, k in enumerate(targets_dict.keys())}

    #     return super()._general_step(batch, batch_idx, step_name, to_cpu)


    def training_step(self, *inputs) -> Dict[str, Any]:
        batch = self._build_batch(*inputs)
        out_batch = super().training_step(batch, batch_idx=0, to_cpu=False)
        out_batch = self._clean_output_batch(out_batch)
        return out_batch

    def validation_step(self, *inputs) -> Dict[str, Any]:
        batch = self._build_batch(*inputs)
        out_batch = super().validation_step(batch, batch_idx=0, to_cpu=False)
        out_batch = self._clean_output_batch(out_batch)
        return out_batch

    def test_step(self, *inputs) -> Dict[str, Any]:
        batch = self._build_batch(*inputs)
        out_batch = super().test_step(batch, batch_idx=0, to_cpu=False)
        out_batch = self._clean_output_batch(out_batch)
        return out_batch

    def predict_step(self, *inputs) -> Any:
        batch = self._build_batch(*inputs)
        out_batch = super().predict_step(batch, batch_idx=0)
        out_batch = self._clean_output_batch(out_batch)
        return out_batch

    def validation_epoch_end(self, outputs: Dict[str, Any]):
        outputs = self._retrieve_output_batch(outputs)
        return super().validation_epoch_end(outputs)

    def _retrieve_output_batch(self, outputs):
        new_outputs = []
        for batch in outputs:
            new_outputs.append({})
            for ii, struct in enumerate(self._output_step_structure):
                new_outputs[-1][struct[0]] = {}
                for jj, key in enumerate(struct[1:]):
                    new_outputs[-1][struct[0]][key] = batch[ii][jj]
        return new_outputs

    def _clean_output_batch(self, out_batch):

        # Transform Dict[Tensor] into Dict[Dict[Tensor]] by grouping them
        cleaned_batch, others = {}, {}
        for key, val in out_batch.items():
            if isinstance(val, dict):
                cleaned_batch[key] = val
            elif isinstance(val, Tensor):
                others[key] = val
        if len(others) > 0:
            cleaned_batch["others_"] = others

        # Save the structure of the dict somewhere
        output_step_structure = []
        for key, val in cleaned_batch.items():
            output_step_structure.append([key])
            for sub_key, sub_val in val.items():
                output_step_structure[-1].append(sub_key)
        self._output_step_structure = output_step_structure

        # # Convert Dict[Dict[Tensor]] into NamedTuple[NamedTuple[Tensor]]
        # new_dict = {}
        # for key, val in cleaned_batch.items():
        #     no_slash = {k.replace("/", "_slash_"): v for k, v in val.items()}
        #     new_dict[key] = named_tuple_from_dict_batch(no_slash, key)
        # cleaned_batch = named_tuple_from_dict_batch(new_dict, "cleaned_batch")

        # Convert Dict[Dict[Tensor]] into Tuple[Tuple[Tensor]]
        new_dict = {}
        for key, val in cleaned_batch.items():
            new_dict[key] = tuple(val.values())
        cleaned_batch = tuple(new_dict.values())

        return cleaned_batch

    def _build_batch(self, *inputs):

        batch = {}

        # Initialize the batch of pyg objects
        for key, pyg_elems in self._keys_batch.items():
            pyg_batch = Batch()
            for pyg_key, idx in pyg_elems.items():
                pyg_batch[pyg_key] = inputs[idx]
            batch[key] = pyg_batch

        # Initialize the dictionaries of tensors, such as the multitask labels
        for key, this_dict in self._keys_tensor_dict.items():
            tensor_dict = {}
            for tensor_key, idx in this_dict.items():
                tensor_dict[tensor_key] = inputs[idx]
            batch[key] = tensor_dict

        # Initialize the tensors
        for key, idx in self._keys_tensor.items():
            batch[key] = inputs[idx]

        # Initialize the other elements
        for key, val in self._keys_others.items():
            if isinstance(val, dict):
                if key not in batch.keys():
                    batch[key] = {}
                # Update the dict or pyg-Batch
                for sub_key, sub_val in val.items():
                    batch[sub_key] = sub_val
            else:
                batch[key] = val

        return batch
