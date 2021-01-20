import torch.nn as nn


class BaseDGLLayer(nn.Module):
    def __init__(self, in_dim, out_dim, residual, activation, dropout, batch_norm):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

    @staticmethod
    def _parse_layer_args(in_dims, out_dims, **kwargs):

        kwargs_of_lists = {}
        kwargs_keys_to_remove = []
        true_out_dims = out_dims
        return in_dims, out_dims, true_out_dims, kwargs_of_lists, kwargs_keys_to_remove
