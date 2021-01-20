import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv

from goli.dgl.dgl_layers.base_dgl_layer import BaseDGLLayer

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""

class GATLayer(BaseDGLLayer):
    """
    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        
    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """    
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=False, activation=F.elu):
        super().__init__(in_dim=in_dim, out_dim=out_dim, residual=residual, 
                        activation=activation, dropout=dropout, batch_norm=batch_norm)

        self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, batch_norm)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

    def forward(self, g, h):
        h_in = h # for residual connection

        h = self.gatconv(g, h).flatten(1)
            
        if self.activation:
            h = self.activation(h)
            
        if self.residual:
            h = h_in + h # residual connection

        return h

    @staticmethod
    def _parse_layer_args(in_dims, out_dims, **kwargs):

        kwargs_of_lists = {}
        kwargs_keys_to_remove = ['num_heads']

        num_heads = kwargs.pop('num_heads')
        num_heads_list = [num_heads for ii in range(len(in_dims) - 1)] + [1]
        in_dims[1:] = [this_dim * num_heads for this_dim in in_dims[1:]]
        kwargs_of_lists['num_heads'] = num_heads_list
        true_out_dims = in_dims[1:] + out_dims[-1:]

        return in_dims, out_dims, true_out_dims, kwargs_of_lists, kwargs_keys_to_remove
    
