import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
from typing import Dict, List, Tuple, Union, Callable

from goli.nn.dgn_operations import parse_dgn_aggregator
from goli.nn.base_layers import MLP, FCLayer, get_activation
from goli.nn.dgl_layers.pna_layer import PNAConvolutionalLayer, PNAMessagePassingLayer
from goli.utils.decorators import classproperty


class BaseDGNLayer:
    def _parse_aggregators(self, aggregators: List[str]) -> List[Callable]:
        r"""
        Parse the aggregators from a list of strings into a list of callables
        """
        return parse_dgn_aggregator(aggregators_name=aggregators)

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        The message function to generate messages along the edges.
        """
        return {
            "e": edges.data["e"],
            "source_pos": edges.data["source_pos"],
            "dest_pos": edges.data["dest_pos"],
        }

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        r"""
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data["h"]
        h = nodes.mailbox["e"]
        source_pos = nodes.mailbox["source_pos"]
        dest_pos = nodes.mailbox["dest_pos"]
        D = h.shape[-2]

        # aggregators and scalers
        h_to_cat = [
            aggr(h=h, h_in=h_in, source_pos=source_pos, dest_pos=dest_pos) for aggr in self.aggregators
        ]
        h = torch.cat(h_to_cat, dim=-1)
        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)

        return {"h": h}


class DGNConvolutionalLayer(BaseDGNLayer, PNAConvolutionalLayer):
    def _parse_aggregators(self, aggregators: List[str]) -> List[Callable]:
        return super(BaseDGNLayer, self)._parse_aggregators(aggregators)

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        return super(BaseDGNLayer, self).message_func(edges)

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        return super(BaseDGNLayer, self).reduce_func(nodes)

    def pretrans_edges(self, edges):
        return {"e": edges.src["h"], "source_pos": edges.src["pos"], "dest_pos": edges.dst["pos"]}


class DGNMessagePassingLayer(BaseDGNLayer, PNAMessagePassingLayer):
    def _parse_aggregators(self, aggregators: List[str]) -> List[Callable]:
        return super(BaseDGNLayer, self)._parse_aggregators(aggregators)

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        return super(BaseDGNLayer, self).message_func(edges)

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        return super(BaseDGNLayer, self).reduce_func(nodes)

    def pretrans_edges(self, edges):
        pretrans = super().pretrans_edges(edges)
        pretrans.update({"source_pos": edges.src["pos"], "dest_pos": edges.dst["pos"]})
        return pretrans
