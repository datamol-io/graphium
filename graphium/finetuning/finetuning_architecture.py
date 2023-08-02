from typing import Iterable, List, Dict, Tuple, Union, Callable, Any, Optional, Type

from copy import deepcopy

from loguru import logger

import torch
import torch.nn as nn

from torch import Tensor
from torch_geometric.data import Batch

from graphium.data.utils import get_keys
from graphium.nn.base_graph_layer import BaseGraphStructure
from graphium.nn.architectures.encoder_manager import EncoderManager
from graphium.nn.architectures import FullGraphMultiTaskNetwork, FeedForwardNN, FeedForwardPyg, TaskHeads
from graphium.nn.architectures.global_architectures import FeedForwardGraph
from graphium.trainer.predictor_options import ModelOptions
from graphium.nn.utils import MupMixin

from graphium.trainer import PredictorModule
from graphium.utils.spaces import GRAPHIUM_PRETRAINED_MODELS_DICT, FINETUNING_HEADS_DICT


class FullGraphFinetuningNetwork(nn.Module, MupMixin):
    def __init__(
        self,
        pretrained_model_kwargs: Dict[str, Any],
        pretrained_overwriting_kwargs: Dict[str, Any],
        finetuning_head_kwargs: Optional[Dict[str, Any]] = None,
        # accelerator_kwargs: Optional[Dict[str, Any]] = None,
        num_inference_to_average: int = 1,
        last_layer_is_readout: bool = False,
        name: str = "FullFinetuningGNN",
    ):
        r"""
        Class that allows to implement a full graph finetuning network architecture,
        including the pre-processing MLP and the post processing MLP.

        Parameters:

            pretrained_model_kwargs

            finetuning_head_kwargs:
                key-word arguments to use for the finetuning head.
                It must respect the following criteria:

                - [last_used_module]_kwarg["out_level"] must be equal to finetuning_head_kwargs["in_level"]
                - [last_used_module]_kwarg["out_dim"] must be equal to finetuning_head_kwargs["in_dim"]

                Here, [last_used_module] represents the module that is finetuned from,
                e.g., gnn, graph_output or (one of the) task_heads

            accelerator_kwargs:
                key-word arguments specific to the accelerator being used,
                e.g. pipeline split points

            num_inference_to_average:
                Number of inferences to average at val/test time. This is used to avoid the noise introduced
                by positional encodings with sign-flips. In case no such encoding is given,
                this parameter is ignored.
                NOTE: The inference time will be slowed-down proportionaly to this parameter.

            last_layer_is_readout: Whether the last layer should be treated as a readout layer.
                Allows to use the `mup.MuReadout` from the muTransfer method https://github.com/microsoft/mup

            name:
                Name attributed to the current network, for display and printing
                purposes.
        """

        super().__init__()

        self.name = name
        self.num_inference_to_average = num_inference_to_average
        self.last_layer_is_readout = last_layer_is_readout
        self._concat_last_layers = None
        self.finetuning_head_kwargs = finetuning_head_kwargs
        self.max_num_nodes_per_graph = None
        self.max_num_edges_per_graph = None

        self.pretrained_model = PretrainedModel(pretrained_model_kwargs, pretrained_overwriting_kwargs)

        if finetuning_head_kwargs is not None:
            self.finetuning_head = FinetuningHead(finetuning_head_kwargs)

    def forward(self, g: Batch) -> Tensor:
        r"""
        Apply the pre-processing neural network, the graph neural network,
        and the post-processing neural network on the graph features.

        Parameters:

            g:
                pyg Batch graph on which the convolution is done.
                Must contain the following elements:

                - Node key `"feat"`: `torch.Tensor[..., N, Din]`.
                  Input node feature tensor, before the network.
                  `N` is the number of nodes, `Din` is the input features dimension ``self.pre_nn.in_dim``

                - Edge key `"edge_feat"`: `torch.Tensor[..., N, Ein]` **Optional**.
                  The edge features to use. It will be ignored if the
                  model doesn't supporte edge features or if
                  `self.in_dim_edges==0`.

                - Other keys related to positional encodings `"pos_enc_feats_sign_flip"`,
                  `"pos_enc_feats_no_flip"`.

        Returns:

            `torch.Tensor[..., M, Dout]` or `torch.Tensor[..., N, Dout]`:
                Node or graph feature tensor, after the network.
                `N` is the number of nodes, `M` is the number of graphs,
                `Dout` is the output dimension ``self.graph_output_nn.out_dim``
                If the `self.gnn.pooling` is [`None`], then it returns node features and the output dimension is `N`,
                otherwise it returns graph features and the output dimension is `M`

        """

        g = self.pretrained_model.forward(g)

        if self.finetuning_head is not None:
            g = self.finetuning_head.forward(g)

        return g

    def make_mup_base_kwargs(self, divide_factor: float = 2.0) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.

        Returns:
            Dictionary with the kwargs to create the base model.
        """
        kwargs = dict(
            pretrained_model_kwargs=None,
            finetuning_head_kwargs=None,
            num_inference_to_average=self.num_inference_to_average,
            last_layer_is_readout=self.last_layer_is_readout,
            name=self.name,
        )

        kwargs["pretrained_model_kwargs"] = self.pretrained_model.make_mup_base_kwargs(
            divide_factor=divide_factor, factor_in_dim=True
        )

        if self.finetuning_head is not None:
            kwargs["finetuning_head_kwargs"] = self.finetuning_head.make_mup_base_kwargs(
                divide_factor=divide_factor, factor_in_dim=True
            )

        return kwargs

    def set_max_num_nodes_edges_per_graph(self, max_nodes: Optional[int], max_edges: Optional[int]) -> None:
        """
        Set the maximum number of nodes and edges for all gnn layers and encoder layers

        Parameters:
            max_nodes: Maximum number of nodes in the dataset.
                This will be useful for certain architecture, but ignored by others.

            max_edges: Maximum number of edges in the dataset.
                This will be useful for certain architecture, but ignored by others.
        """
        self.max_num_nodes_per_graph = max_nodes
        self.max_num_edges_per_graph = max_edges
        if (self.encoder_manager is not None) and (self.encoder_manager.pe_encoders is not None):
            for encoder in self.encoder_manager.pe_encoders.values():
                encoder.max_num_nodes_per_graph = max_nodes
                encoder.max_num_edges_per_graph = max_edges
        if self.gnn is not None:
            for layer in self.gnn.layers:
                if isinstance(layer, BaseGraphStructure):
                    layer.max_num_nodes_per_graph = max_nodes
                    layer.max_num_edges_per_graph = max_edges

        self.task_heads.set_max_num_nodes_edges_per_graph(max_nodes, max_edges)

    # @property
    # def in_dim(self) -> int:
    #     r"""
    #     Returns the input dimension of the network
    #     """
    #     if self.pre_nn is not None:
    #         return self.pre_nn.in_dim
    #     else:
    #         return self.gnn.in_dim

    # @property
    # def out_dim(self) -> int:
    #     r"""
    #     Returns the output dimension of the network
    #     """
    #     return self.gnn.out_dim

    # @property
    # def out_dim_edges(self) -> int:
    #     r"""
    #     Returns the output dimension of the edges
    #     of the network.
    #     """
    #     if self.gnn.full_dims_edges is not None:
    #         return self.gnn.full_dims_edges[-1]
    #     return self.gnn.in_dim_edges

    # @property
    # def in_dim_edges(self) -> int:
    #     r"""
    #     Returns the input edge dimension of the network
    #     """
    #     return self.gnn.in_dim_edges


class PretrainedModel(nn.Module, MupMixin):
    def __init__(
        self,
        pretrained_model_kwargs: Dict[str, Any],
        pretrained_overwriting_kwargs: Dict[str, Any],
        pretrained_model: str = 'dummy-pretrained-model-cpu'
    ):
        r"""
        A flexible neural network architecture, with variable hidden dimensions,
        support for multiple layer types, and support for different residual
        connections.

        This class is meant to work with different graph neural networks
        layers. Any layer must inherit from `graphium.nn.base_graph_layer.BaseGraphStructure`
        or `graphium.nn.base_graph_layer.BaseGraphLayer`.

        Parameters:

            pretrained_model_kwargs:

            pretrained_overwriting_kwargs:

            pretrained_model:

        """

        super().__init__()

        # Load pretrained model
        pretrained_model = PredictorModule.load_from_checkpoint(
            GRAPHIUM_PRETRAINED_MODELS_DICT[pretrained_model]
        ).model

        # Initialize new model with architecture after
        net = type(pretrained_model)
        self.net = net(**pretrained_model_kwargs)

        # Overwrite shared parameters with pretrained model
        self.overwrite_with_pretrained(pretrained_model, **pretrained_overwriting_kwargs)


    def forward(self, g: Union[torch.Tensor, Batch]):
        g = self.net.forward(g)

        return g

    def overwrite_with_pretrained(
        self,
        pretrained_model: str,
        task: str,
        finetuning_module: str,
        added_depth: int,
        task_head_from_pretrained: str = None,
    ):
        self.net._create_module_map()
        module_map = self.net._module_map

        # Below list should also come from pretrained model (like module_map)
        for module in ["pre_nn", "pre_nn_edges", "gnn", "graph_output_nn", "task_heads"]:
            if module == finetuning_module:
                break

            self.overwrite_complete_module(pretrained_model, module, module_map)

        self.overwrite_partial_module(pretrained_model, module, module_map, task, added_depth, task_head_from_pretrained)

    def overwrite_partial_module(
        self, pretrained_model, module, module_map, task, added_depth, task_head_from_pretrained
    ):
        """
        Completely overwrite the specified module
        """
        if module == "gnn":
            shared_depth = len(module_map[module].layers) - added_depth
            assert shared_depth >= 0
            if shared_depth > 0:
                module_map[module].layers[:shared_depth] = pretrained_model.gnn.layers[:shared_depth]

        elif module == "graph_output_nn":
            for task_level in module_map[module].keys():
                shared_depth = (
                    len(module_map[module][task_level].graph_output_nn.layers) - added_depth
                )
                assert shared_depth >= 0
                if shared_depth > 0:
                    module_map[module][task_level].graph_output_nn.layers = (
                        pretrained_model.task_heads.graph_output_nn[task_level].graph_output_nn.layers[
                            :shared_depth
                        ]
                        + module_map[module][task_level].graph_output_nn.layers[shared_depth:]
                    )

        elif module == "task_heads":
            shared_depth = len(module_map[module][task].layers) - added_depth
            assert shared_depth >= 0
            if shared_depth > 0:
                module_map[module][task].layers = (
                    pretrained_model.task_heads.task_heads[task_head_from_pretrained].layers[:shared_depth]
                    + module_map[module][task].layers[shared_depth:]
                )

        elif module in ["pre_nn", "pre_nn_edges"]:
            raise NotImplementedError(f"Finetune from (edge) pre-NNs is not supported")

        else:
            raise NotImplementedError(f"This is an unknown module type")

    def overwrite_complete_module(self, pretrained_model, module, module_map):
        """
        Completely overwrite the specified module
         """
        if module == "pre_nn":
            try:
                module_map[module] = pretrained_model.pre_nn.layers
            except:
                pass
                # logger.warning(
                #     f"Pretrained ({pretrained_model.pre_nn}) and/or finetune model ({self.pre_nn}) do not use a pre-NN."
                # )

        elif module == "pre_nn_edges":
            try:
                module_map[module] = pretrained_model.pre_nn_edges.layers
            except:
                pass
                # logger.warning(
                #     f"Pretrained ({pretrained_model.pre_nn_edges}) and/or finetune model ({self.pre_nn_edges}) do not use a pre-NN-edges."
                # )

        elif module == "gnn":
            module_map[module] = pretrained_model.gnn.layers

        elif module == "graph_output_nn":
            for task_level in module_map[module].keys():
                module_map[module][task_level] = pretrained_model.task_heads.graph_output_nn[
                    task_level
                ]

        else:
            raise NotImplementedError(f"This is an unknown module type")

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension

        Returns:
            Dictionary with the kwargs to create the base model.
        """
        # For the post-nn network, all the dimension are divided

        return self.net.make_mup_base_kwargs(
            divide_factor=divide_factor, factor_in_dim=factor_in_dim
        )


class FinetuningHead(nn.Module, MupMixin):
    def __init__(self, finetuning_head_kwargs: Dict[str, Any]):
        r"""
        A flexible neural network architecture, with variable hidden dimensions,
        support for multiple layer types, and support for different residual
        connections.

        This class is meant to work with different graph neural networks
        layers. Any layer must inherit from `graphium.nn.base_graph_layer.BaseGraphStructure`
        or `graphium.nn.base_graph_layer.BaseGraphLayer`.

        Parameters:

            ...

        """

        super().__init__()
        self.task = finetuning_head_kwargs.pop("task", None)
        self.previous_module = finetuning_head_kwargs.pop("previous_module", "task_heads")
        self.incoming_level = finetuning_head_kwargs.pop("incoming_level", "graph")

        model_type = finetuning_head_kwargs.pop("model_type", "mlp")
        net = FINETUNING_HEADS_DICT[model_type]
        self.net = net(**finetuning_head_kwargs)

    def forward(self, g: Union[torch.Tensor, Batch]):
        if self.previous_module == "task_heads":
            g = list(g.values())[0]

        g = self.net.forward(g)

        return {self.task: g}

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension

        Returns:
            Dictionary with the kwargs to create the base model.
        """
        # For the post-nn network, all the dimension are divided

        return self.net.make_mup_base_kwargs(
            divide_factor=divide_factor, factor_in_dim=factor_in_dim
        )
