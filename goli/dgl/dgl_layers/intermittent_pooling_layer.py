import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from goli.dgl.base_layers import parse_pooling_layer, FCLayer, MLP

"""
    
"""


class IntermittentPoolingLayer(nn.Module):
    """


    Parameters
    ----------


    """

    def __init__(
        self,
        in_dim,
        num_layers=2,
        pooling="sum",
        activation="relu",
        last_activation=None,
        dropout=0.0,
        batch_norm=False,
        bias=True,
        init_fn=None,
        device="cpu",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.pooling = pooling
        self.pool_layer, self.pool_out_dim = parse_pooling_layer(in_dim=in_dim, pooling=pooling)

        # Create the feedforward network that follows the pooling
        hidden_dims = num_layers * [in_dim]

        in_dim = self.pool_out_dim + in_dim if (len(self.pool_layer) > 0) else in_dim
        self.mlp = MLP(
            in_dim=in_dim,
            hidden_dim=in_dim,
            out_dim=in_dim,
            layers=num_layers,
            mid_activation=activation,
            last_activation=last_activation,
            dropout=dropout,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            device=device,
        )

    def forward(self, g, h):

        if len(self.pool_layer) > 0:
            # Compute pooling
            pooled = self.pool_layer(g, h)

            # Concatenate the pooling values to each of the node features
            prev_nodes = 0
            new_h = []
            for ii, num_nodes in enumerate(g.batch_num_nodes):
                this_h = h[prev_nodes : prev_nodes + num_nodes]
                pooled_expanded = pooled[ii : ii + 1].expand(
                    this_h.shape[:-1] + pooled[ii : ii + 1].shape[-1:]
                )
                new_h.append(pooled_expanded)
                prev_nodes += num_nodes
            h = torch.cat([h, torch.cat(new_h, dim=0)], dim=-1)

        # Apply an MLP
        h = self.mlp(h)

        return h
