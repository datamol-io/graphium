import re
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
from typing import Dict, List, Tuple

from goli.dgl.pna_operations import PNA_AGGREGATORS, PNA_SCALERS
from goli.dgl.base_layers import MLP, FCLayer, get_activation
from goli.dgl.dgl_layers.base_dgl_layer import BaseDGLLayer

"""
    PNA: Principal Neighbourhood Aggregation 
    Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
"""


class BasePNALayer(BaseDGLLayer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregators: List[str],
        scalers: List[str],
        activation="relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        avg_d: float = 1.0,
        last_activation="none",
    ):
        r"""
        Abstract class used to standardize the implementation of PNA layers
        in the current library.

        PNA: Principal Neighbourhood Aggregation
        Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
        https://arxiv.org/abs/2004.05718

        Method ``layer_inputs_edges()`` needs to be implemented in children classes

        Parameters
        ------------

        in_dim: int
            Input feature dimensions of the layer

        out_dim: int
            Output feature dimensions of the layer

        aggregators: list(str)
            Set of aggregation function identifiers,
            e.g. "mean", "max", "min", "std", "sum", "var", "moment3".
            The results from all aggregators will be concatenated.

        scalers: list(str)
            Set of scaling functions identifiers
            e.g. "identidy", "amplification", "attenuation"
            The results from all scalers will be concatenated

        activation: str, Callable, Default="relu"
            activation function to use in the layer

        dropout: float, Default=0.
            The ratio of units to dropout. Must be between 0 and 1

        batch_norm: bool, Default=False
            Whether to use batch normalization

        avg_d: float, Default=1.
            Average degree of nodes in the training set, used by scalers to normalize

        last_activation: str, Callable, Default="none"
            activation function to use in the last layer of the internal MLP

        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        # Initializing basic attributes
        self.avg_d = avg_d
        self.last_activation = get_activation(last_activation)

        # Initializing aggregators and scalers
        self.aggregators = [PNA_AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [PNA_SCALERS[scale] for scale in scalers]

    def message_func(self, edges) -> Dict:
        r"""
        The message function to generate messages along the edges.
        """
        return {"e": edges.data["e"]}

    def reduce_func(self, nodes) -> Dict:
        r"""
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data["h"]
        h = nodes.mailbox["e"]
        D = h.shape[-2]
        h_to_cat = [aggr(h=h, h_in=h_in) for aggr in self.aggregators]
        h = torch.cat(h_to_cat, dim=1)

        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)

        return {"h": h}


    def layer_outputs_edges(self) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns
        ---------

        uses_edges: bool
            Always ``False`` for the current class
        """
        return False

    def get_out_dim_factor(self) -> int:
        r"""
        Get the factor by which the output dimension is multiplied for
        the next layer.

        For standard layers, this will return ``1``.

        But for others, such as ``GatLayer``, the output is the concatenation
        of the outputs from each head, so the out_dim gets multiplied by
        the number of heads, and this function should return the number
        of heads.

        Returns
        ---------

        dim_factor: int
            Always ``1`` for the current class
        """
        return 1


class PNAConvolutionalLayer(BasePNALayer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregators: List[str],
        scalers: List[str],
        activation="relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        avg_d: Dict[str, float] = {"log" :1.0},
        last_activation="none",
        posttrans_layers: int = 1,
    ):
        r"""
        Implementation of the convolutional architecture of the PNA layer,
        previously known as `PNASimpleLayer`. This layer aggregates the
        neighbouring messages using multiple aggregators and scalers,
        concatenates their results, then applies an MLP on the concatenated
        features.

        PNA: Principal Neighbourhood Aggregation
        Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
        https://arxiv.org/abs/2004.05718

        Parameters
        ------------

        in_dim: int
            Input feature dimensions of the layer

        out_dim: int
            Output feature dimensions of the layer

        aggregators: list(str)
            Set of aggregation function identifiers,
            e.g. "mean", "max", "min", "std", "sum", "var", "moment3".
            The results from all aggregators will be concatenated.

        scalers: list(str)
            Set of scaling functions identifiers
            e.g. "identidy", "amplification", "attenuation"
            The results from all scalers will be concatenated

        activation: str, Callable, Default="relu"
            activation function to use in the layer

        dropout: float, Default=0.
            The ratio of units to dropout. Must be between 0 and 1

        batch_norm: bool, Default=False
            Whether to use batch normalization

        avg_d: Dict(str, float), Default={"log": 1.}
            Average degree of nodes in the training set, used by scalers to normalize

        last_activation: str, Callable, Default="none"
            activation function to use in the last layer of the internal MLP

        posttrans_layers: int, Default=1
            number of layers in the MLP transformation after the aggregation

        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            aggregators=aggregators,
            scalers=scalers,
            avg_d=avg_d,
            activation=activation,
            dropout=0,
            batch_norm=False,
            last_activation=last_activation,
        )

        # MLP used on the aggregated messages of the neighbours
        self.posttrans = MLP(
            in_dim=(len(aggregators) * len(scalers)) * self.in_dim,
            hidden_dim=self.out_dim,
            out_dim=self.out_dim,
            layers=posttrans_layers,
            mid_activation=self.activation,
            last_activation=self.last_activation,
            dropout=dropout,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
        )

    def pretrans_edges(self, edges) -> Dict:
        r"""
        Return a mapping to the features of the source nodes.
        """
        return {"e": edges.src["h"]}

    def forward(self, g: DGLGraph, h: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the PNA convolutional layer, with the specified post transformation

        Parameters
        ------------

        g: dgl.DGLGraph
            graph on which the convolution is done

        h: torch.Tensor(..., N, Din)
            Node feature tensor, before convolution.
            N is the number of nodes, Din is the input dimension ``self.in_dim``

        Returns
        ---------

        h: torch.Tensor(..., N, Dout)
            Node feature tensor, after convolution.
            N is the number of nodes, Dout is the output dimension ``self.out_dim``

        """

        g.ndata["h"] = h
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]

        # post-transformation
        h = self.posttrans(h)

        return h

    def layer_inputs_edges(self) -> bool:
        r"""
        Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns
        ---------

        uses_edges: bool
            Always ``False`` for the current class
        """
        return False

    @staticmethod
    def layer_supports_edges() -> bool:
        r"""
        Return a boolean specifying if the layer type supports edges or not.

        Returns
        ---------

        supports_edges: bool
            Always ``False`` for the current class
        """
        return False


class PNAMessagePassingLayer(BasePNALayer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregators: List[str],
        scalers: List[str],
        activation="relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        avg_d: Dict[str, float] = {"log" :1.0},
        last_activation="none",
        posttrans_layers: int = 1,
        pretrans_layers: int = 1,
        in_dim_edges: int = 0,
    ):
        r"""
        Implementation of the convolutional architecture of the PNA layer,
        previously known as `PNASimpleLayer`. This layer aggregates the
        neighbouring messages using multiple aggregators and scalers,
        concatenates their results, then applies an MLP on the concatenated
        features.

        PNA: Principal Neighbourhood Aggregation
        Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
        https://arxiv.org/abs/2004.05718

        Parameters
        ------------

        in_dim: int
            Input feature dimensions of the layer

        out_dim: int
            Output feature dimensions of the layer

        aggregators: list(str)
            Set of aggregation function identifiers,
            e.g. "mean", "max", "min", "std", "sum", "var", "moment3".
            The results from all aggregators will be concatenated.

        scalers: list(str)
            Set of scaling functions identifiers
            e.g. "identidy", "amplification", "attenuation"
            The results from all scalers will be concatenated

        activation: str, Callable, Default="relu"
            activation function to use in the layer

        dropout: float, Default=0.
            The ratio of units to dropout. Must be between 0 and 1

        batch_norm: bool, Default=False
            Whether to use batch normalization

        avg_d: Dict(str, float), Default={"log": 1.}
            Average degree of nodes in the training set, used by scalers to normalize

        last_activation: str, Callable, Default="none"
            activation function to use in the last layer of the internal MLP

        posttrans_layers: int, Default=1
            number of layers in the MLP transformation after the aggregation

        pretrans_layers: int, Default=1
            number of layers in the transformation before the aggregation

        in_dim_edges: int, Default=0
            size of the edge features. If 0, edges are ignored

        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            aggregators=aggregators,
            scalers=scalers,
            avg_d=avg_d,
            activation=activation,
            dropout=0,
            batch_norm=False,
            last_activation=last_activation,
        )

        self.in_dim_edges = in_dim_edges
        self.edge_features = self.in_dim_edges > 0

        # MLP used on each pair of nodes with their edge MLP(h_u, h_v, e_uv)
        self.pretrans = MLP(
            in_dim=2 * in_dim + in_dim_edges,
            hidden_dim=in_dim,
            out_dim=in_dim,
            layers=pretrans_layers,
            mid_activation=self.activation,
            last_activation=self.last_activation,
            dropout=dropout,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
        )

        # MLP used on the aggregated messages MLP(h'_u)
        self.posttrans = MLP(
            in_dim=(len(self.aggregators) * len(self.scalers) + 1) * in_dim,
            hidden_dim=out_dim,
            out_dim=out_dim,
            layers=posttrans_layers,
            mid_activation=self.activation,
            last_activation=self.last_activation,
            dropout=dropout,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
        )

    def pretrans_edges(self, edges) -> Dict:
        r"""
        Return a mapping to the concatenation of the features from
        the source node, the destination node, and the edge between them (if applicable).
        """
        if self.edge_features:
            z2 = torch.cat([edges.src["h"], edges.dst["h"], edges.data["ef"]], dim=1)
        else:
            z2 = torch.cat([edges.src["h"], edges.dst["h"]], dim=1)
        return {"e": self.pretrans(z2)}

    def forward(
        self, g: DGLGraph, h: torch.Tensor, e: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply the PNA Message passing layer, with the specified pre/post transformations

        Parameters
        ------------

        g: dgl.DGLGraph
            graph on which the convolution is done

        h: torch.Tensor(..., N, Din)
            Node feature tensor, before convolution.
            N is the number of nodes, Din is the input dimension ``self.in_dim``

        e: torch.Tensor(..., N, Din_edges) or None, Default=None
            Edge feature tensor, before convolution.
            N is the number of nodes, Din is the input edge dimension

            Can be set to None if the layer does not use edge features
            i.e. ``self.layer_inputs_edges() -> False``

        Returns
        ---------

        h: torch.Tensor(..., N, Dout)
            Node feature tensor, after convolution.
            N is the number of nodes, Dout is the output dimension ``self.out_dim``

        """

        g.ndata["h"] = h
        if self.edge_features:  # add the edges information only if edge_features = True
            g.edata["ef"] = e

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata["h"]], dim=1)

        # post-transformation
        h = self.posttrans(h)

        return h

    def layer_inputs_edges(self) -> bool:
        r"""
        Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns
        ---------

        uses_edges: bool
            Returns ``self.edge_features``
        """
        return self.edge_features

    @staticmethod
    def layer_supports_edges() -> bool:
        r"""
        Return a boolean specifying if the layer type supports edges or not.

        Returns
        ---------

        supports_edges: bool
            Always ``True`` for the current class
        """
        return True