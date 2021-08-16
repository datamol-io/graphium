from typing import List, Union, Callable, Dict
from dgl import DGLGraph
import torch


from goli.nn.dgl_layers import DGNMessagePassingLayer


class DGNHybridTransformerLayer(DGNMessagePassingLayer):

    def __init__(
        self, 
        in_dim: int, 
        out_dim: int,
        num_heads: int,
        aggregators: List[str], 
        scalers: List[str], 
        activation: Union[Callable, str], 
        dropout: float, 
        normalization: Union[str, Callable],
        in_dim_edges: int = 0,
    ):  
        self.num_heads = num_heads
        super().__init__(in_dim, out_dim, aggregators, scalers, activation=activation, dropout=dropout, normalization=normalization, in_dim_edges=in_dim_edges)

        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model = out_dim,nhead=num_heads, dim_feedforward=out_dim, dropout=dropout)

    def forward(self, g: DGLGraph, h: torch.Tensor, e: torch.Tensor = None) -> torch.Tensor:
        # h: # [num_nodes_total, in_dim]
        h = super().forward(g=g, h=h, e=e) # [num_nodes_total, out_dim]
        m,n = h.size()
    
        
        # Take the tensor h, and transform it into a tensor of shape [sequence_lenght, batch_size, out_dim]
        seq_len = g.batch_num_nodes().max()
        h_mask = torch.arange(seq_len, device=h.device).unsqueeze(1)
        h_mask = h_mask < g.batch_num_nodes().unsqueeze(0)
        h_mask = h_mask.unsqueeze(-1).repeat([1, 1, h.shape[-1]]) # [sequence_lenght, batch_size, out_dim]

        #src: [sequence_lenght, batch_size, out_dim]
        src = torch.zeros([g.batch_num_nodes().max(), g.batch_size, h.shape[-1]], dtype=h.dtype, device=h.device)
        src = src.masked_scatter(h_mask, h.flatten())
        
        # src_mask = ~h_mask[:,:,-1].t()

        h = self.transformer_layer(src=src) #src_key_padding_mask = src_mask) # [sequence_lenght, batch_size, num_dim]
        h = h[h_mask]
        h = h.reshape([m,n]) # [num_nodes_total, out_dim]

        # idea1: del src, src_mask
        # idea2: line 42 -> h = src.masked_scatter
        # idea3: ???

        return h # [num_nodes_total, out_dim]


