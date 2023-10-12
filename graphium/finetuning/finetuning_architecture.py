from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch

from graphium.nn.utils import MupMixin
from graphium.trainer.predictor import PredictorModule
from graphium.utils.spaces import FINETUNING_HEADS_DICT


class FullGraphFinetuningNetwork(nn.Module, MupMixin):
    def __init__(
        self,
        pretrained_model: Union[str, "PretrainedModel"],
        pretrained_model_kwargs: Dict[str, Any] = {},
        pretrained_overwriting_kwargs: Dict[str, Any] = {},
        finetuning_head_kwargs: Optional[Dict[str, Any]] = None,
        num_inference_to_average: int = 1,
        last_layer_is_readout: bool = False,
        name: str = "FullFinetuningGNN",
    ):
        r"""
        Flexible class that allows to implement an end-to-end graph finetuning network architecture, supporting flexible pretrained models and finetuning heads.
        The network decomposes into two parts of class PretrainedModel and FinetuningHead. The PretrainedModel class allows basic finetuning such as
        finetuning from a specified module of the pretrained model and dropping/adding layers in this module. The (optional) FinetuningHead class allows more
        flexible finetuning with a custom network applied after the pretrained model. If not specified, we fall back to basic finetuning integrated in PretrainedModel.

        Parameters:

            pretrained_model:
                A PretrainedModel or an identifier of pretrained model within GRAPHIUM_PRETRAINED_MODELS_DICT or a valid .ckpt checkpoint path

            pretrained_model_kwargs:
                Key-word arguments to instantiate a model of the same class as the pretrained model (e.g., FullGraphMultitaskNetwork))

            pretrained_overwriting_kwargs:
                Key-word arguments indicating which parameters of loaded model are shared with the pretrained part of FullGraphFinetuningNetwork

            finetuning_head_kwargs:
                Key-word arguments to use for the finetuning head.
                It must respect the following criteria:
                - pretrained_model_kwargs[last_used_module]["out_level"] must be equal to finetuning_head_kwargs["in_level"]
                - pretrained_model_kwargs[last_used_module]["out_dim"] must be equal to finetuning_head_kwargs["in_dim"]

                Here, [last_used_module] represents the module that is finetuned from,
                e.g., gnn, graph_output or (one of the) task_heads

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
        self.pretrained_model = pretrained_model
        self.pretrained_overwriting_kwargs = pretrained_overwriting_kwargs
        self.finetuning_head_kwargs = finetuning_head_kwargs
        self.max_num_nodes_per_graph = None
        self.max_num_edges_per_graph = None
        self.finetuning_head = None

        if not isinstance(self.pretrained_model, PretrainedModel):
            self.pretrained_model = PretrainedModel(
                self.pretrained_model, pretrained_model_kwargs, pretrained_overwriting_kwargs
            )

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
            pretrained_model=self.pretrained_model,
            pretrained_model_kwargs=None,
            finetuning_head_kwargs=None,
            num_inference_to_average=self.num_inference_to_average,
            last_layer_is_readout=self.last_layer_is_readout,
            name=self.name,
        )

        kwargs["pretrained_model_kwargs"] = self.pretrained_model.make_mup_base_kwargs(
            divide_factor=divide_factor
        )

        if self.finetuning_head is not None:
            kwargs["finetuning_head_kwargs"] = self.finetuning_head.make_mup_base_kwargs(
                divide_factor=divide_factor, factor_in_dim=True
            )

        kwargs["pretrained_overwriting_kwargs"] = self.pretrained_overwriting_kwargs

        return kwargs

    def set_max_num_nodes_edges_per_graph(self, max_nodes: Optional[int], max_edges: Optional[int]) -> None:
        r"""
        Set the maximum number of nodes and edges for all gnn layers and encoder layers

        Parameters:
            max_nodes: Maximum number of nodes in the dataset.
                This will be useful for certain architecture, but ignored by others.

            max_edges: Maximum number of edges in the dataset.
                This will be useful for certain architecture, but ignored by others.
        """

        self.pretrained_model.net.set_max_num_nodes_edges_per_graph(max_nodes, max_edges)


class PretrainedModel(nn.Module, MupMixin):
    def __init__(
        self,
        pretrained_model: str,
        pretrained_model_kwargs: Dict[str, Any],
        pretrained_overwriting_kwargs: Dict[str, Any],
    ):
        r"""
        Flexible class allowing to finetune pretrained models from GRAPHIUM_PRETRAINED_MODELS_DICT or from a ckeckpoint path.
        Can be any model that inherits from nn.Module, MupMixin and comes with a module map (e.g., FullGraphMultitaskNetwork)

        Parameters:

            pretrained_model:
                Identifier of pretrained model within GRAPHIUM_PRETRAINED_MODELS_DICT or from a checkpoint path

            pretrained_model_kwargs:
                Key-word arguments to instantiate a model of the same class as the pretrained model (e.g., FullGraphMultitaskNetwork))

            pretrained_overwriting_kwargs:
                Key-word arguments indicating which parameters of loaded model are shared with the pretrained part of FullGraphFinetuningNetwork

        """

        super().__init__()

        # Load pretrained model
        pretrained_model = PredictorModule.load_pretrained_model(pretrained_model, device='cpu').model
        pretrained_model.create_module_map()

        # Initialize new model with architecture after desired modifications to architecture.
        net = type(pretrained_model)
        self.net = net(**pretrained_model_kwargs)
        self.net.create_module_map()

        # Overwrite parameters shared between loaded and modified pretrained model
        self.overwrite_with_pretrained(pretrained_model, **pretrained_overwriting_kwargs)

    def forward(self, g: Union[torch.Tensor, Batch]):
        g = self.net.forward(g)

        return g

    def overwrite_with_pretrained(
        self,
        pretrained_model,
        finetuning_module: str,
        added_depth: int = 0,
        sub_module_from_pretrained: str = None,
    ):
        """
        Overwrite parameters shared between loaded and modified pretrained model

        Parameters:
            pretrained_model: Model from GRAPHIUM_PRETRAINED_MODELS_DICT
            finetuning_module: Module to finetune from
            added_depth: Number of modified layers at the end of finetuning module
            sub_module_from_pretrained: Optional submodule to finetune from
        """
        module_map = self.net._module_map
        module_map_from_pretrained = pretrained_model._module_map

        module_names_from_pretrained = module_map_from_pretrained.keys()
        super_module_names_from_pretrained = set(
            [module_name.split("-")[0] for module_name in module_names_from_pretrained]
        )

        for module_name in module_map.keys():
            # Below exception handles some modules (e.g., pe_encoders in FullGraphMultitaskNetwork) that do not support len());
            # They can always be replaced entirely
            try:
                shared_depth = len(module_map[module_name])
            except:
                module_map[module_name] = module_map_from_pretrained[module_name]
                continue

            if module_name.startswith(finetuning_module):
                shared_depth -= added_depth

            if module_name in module_map_from_pretrained.keys():
                for idx in range(shared_depth):
                    module_map[module_name][idx] = module_map_from_pretrained[module_name][idx]
            elif module_name.split("-")[0] in super_module_names_from_pretrained:
                for idx in range(shared_depth):
                    module_map[module_name][idx] = module_map_from_pretrained[
                        "".join([module_name.split("-")[0], "-", sub_module_from_pretrained])
                    ][idx]
            else:
                raise RuntimeError("Mismatch between loaded pretrained model and model to be overwritten.")

            if module_name.startswith(finetuning_module):
                break

    def make_mup_base_kwargs(self, divide_factor: float = 2.0) -> Dict[str, Any]:
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

        return self.net.make_mup_base_kwargs(divide_factor=divide_factor)


class FinetuningHead(nn.Module, MupMixin):
    def __init__(self, finetuning_head_kwargs: Dict[str, Any]):
        r"""
        Flexible class allowing to use a custom finetuning head on top of the pretrained model.
        Can be any model that inherits from nn.Module, MupMixin.

        Parameters:

            finetuning_head_kwargs: Key-word arguments needed to instantiate a custom (or existing) finetuning head from FINETUNING_HEADS_DICT

        """

        super().__init__()
        self.task = finetuning_head_kwargs.pop("task", None)
        self.previous_module = finetuning_head_kwargs.pop("previous_module", "task_heads")
        self.incoming_level = finetuning_head_kwargs.pop("incoming_level", "graph")

        model_type = finetuning_head_kwargs.pop("model_type", "mlp")
        net = FINETUNING_HEADS_DICT[model_type]
        self.net = net(**finetuning_head_kwargs)

    def forward(self, g: Union[Dict[str, Union[torch.Tensor, Batch]], torch.Tensor, Batch]):
        if isinstance(g, (torch.Tensor, Batch)):
            pass
        elif isinstance(g, Dict) and len(g) == 1:
            g = list(g.values())[0]
        else:
            raise TypeError("Output type from pretrained model not appropriate for finetuning head")

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

        return self.net.make_mup_base_kwargs(divide_factor=divide_factor, factor_in_dim=factor_in_dim)
