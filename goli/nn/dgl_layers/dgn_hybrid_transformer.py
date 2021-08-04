from typing import List, Union, Callable, Dict
from dgl import DGLGraph
import torch


from goli.nn.dgl_layers import DGNMessagePassingLayer


class DGNHybridTransformerLayer(DGNMessagePassingLayer):

    def __init__(self, in_dim: int, out_dim: int, num_heads:int,aggregators: List[str], scalers: List[str], activation: Union[Callable, str], dropout: float, normalization: Union[str, Callable], avg_d: Dict[str, float], last_activation: Union[Callable, str], posttrans_layers: int, pretrans_layers: int, in_dim_edges: int):
        super().__init__(in_dim, out_dim, aggregators, scalers, activation=activation, dropout=dropout, normalization=normalization, avg_d=avg_d, last_activation=last_activation, posttrans_layers=posttrans_layers, pretrans_layers=pretrans_layers, in_dim_edges=in_dim_edges)

        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model = out_dim,nhead=num_heads, dim_feedforward=out_dim, dropout=dropout,activation=activation)

    def forward(self, g: DGLGraph, h: torch.Tensor, e: torch.Tensor = None) -> torch.Tensor:
        h = super().forward(g=g, h=h, e=e)
        h = self.transformer_layer(h)
        return h


