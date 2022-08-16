'''
#!Andy
adapated from https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gps_layer.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from typing import Callable, Union, Optional


from goli.nn.base_graph_layer import BaseGraphModule
from goli.nn.pyg_layers import GatedGCNPyg, GINConvPyg, GINEConvPyg, PNAMessagePassingPyg
from goli.utils.decorators import classproperty



#from graphgps.layer.gatedgcn_layer import GatedGCNLayer
#from graphgps.layer.gine_conv_layer import GINEConvESLapPE

'''
#! Andy to do
1. inherent from basegraphlayer instead of nn.Module
2. 
'''

PYG_LAYERS_DICT = {
    "pyg:gin": GINConvPyg,
    "pyg:gine": GINEConvPyg,
    "pyg:gated-gcn": GatedGCNPyg,
    "pyg:pna-msgpass": PNAMessagePassingPyg,
}

ATTENTION_LAYERS_DICT = {
    "full-attention": torch.nn.MultiheadAttention
}

class GPSLayerPyg(BaseGraphModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: Optional[int] = None,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        mpnn_type: str = "pyg:gine",
        mpnn_kwargs = None,
        attn_type: str = "full-attention",
        attn_kwargs = None,
    ):
        r"""
        GINE: Graph Isomorphism Networks with Edges
        Strategies for Pre-training Graph Neural Networks
        Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Jure Leskovec
        https://arxiv.org/abs/1905.12265

        [!] code uses the pytorch-geometric implementation of GINEConv

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            activation:
                activation function to use in the layer

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            init_eps :
                Initial :math:`\epsilon` value, default: ``0``.

            learn_eps :
                If True, :math:`\epsilon` will be a learnable parameter.

        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
        )

        self.dropout_local = self.dropout_layer
        self.dropout_attn = self.dropout_layer
        self.ff_dropout1 = self.dropout_layer
        self.ff_dropout2 = self.dropout_layer
        self.ff_linear1 = nn.Linear(in_dim, in_dim * 2)
        self.ff_linear2 = nn.Linear(in_dim * 2, in_dim)
        self.ff_out = nn.Linear(in_dim, out_dim)

        self.normalization = self.norm_layer

        self.activation = self.activation_layer

        # TODO: The rest
        mpnn_default_values = {
            "in_dim": in_dim,
            "out_dim": in_dim,
            "in_dim_edges": in_dim_edges
        }

        #mpnn_kwargs.setdefault(default_values)
        if (mpnn_kwargs is None):
            mpnn_kwargs = mpnn_default_values

        mpnn_class = PYG_LAYERS_DICT[mpnn_type]
        self.mpnn = mpnn_class(**mpnn_kwargs)

        attn_default_values = {
            "embed_dim": in_dim,  
            "num_heads": 1, 
            "dropout": dropout,
            "batch_first": True,
        }

        if attn_kwargs is None:
            attn_kwargs = attn_default_values
        #attn_kwargs.setdefault(default_values)
        attn_class = ATTENTION_LAYERS_DICT[attn_type]
        self.attn_layer = attn_class(**attn_kwargs)

    #! Andy, check for reshaping on the fake graph
    # TODO: Use self.mpnn and self.attn_layer



    def forward(self, batch):
        # pe, h, edge_index, edge_attr = batch.pos_enc_feats_sign_flip, batch.h, batch.edge_index, batch.edge_attr
        h = batch.h

        h_in1 = h  # for first residual connection
        h_out_list = []
        # Local MPNN with edge attributes.
        h_local = (self.mpnn(batch.clone())).h
        h_local = self.dropout_local(h_local)
        h_local = h_in1 + h_local  # Residual connection.
        if (self.normalization):
            h_local = self.normalization(h_local)
            h_out_list.append(h_local)

        '''
        check where positional encoding should be added
        check how to use torch.nn.MultiheadAttention
        '''

        # Multi-head attention.
        #* batch.batch is the indicator vector for nodes of which graph it belongs to
        #* h_dense 
        if self.attn_layer is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.normalization:
                h_attn = self.normalization(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        h = self.ff_out(h)
        if self.normalization:
            h = self.normalization(h)
        
        batch.h = h
        return batch

    def _ff_block(self, x):
        """Feed Forward block.
        """
        if (self.activation is None):
            x = self.ff_dropout1(self.ff_linear1(x))
        else:
            x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.attn_layer(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x


    # # forward function that doesn't do anything
    # def forward(self, batch):
    #     #x, edge_index, edge_attr = batch.h, batch.edge_index, batch.edge_attr

    #     batch = self.mpnn(batch)
    #     #batch.h = self.apply_norm_activation_dropout(batch.h)
    #     return batch

    @classproperty
    def layer_supports_edges(cls) -> bool:
        r"""
        Return a boolean specifying if the layer type supports edges or not.

        Returns:

            supports_edges: bool
                Always ``True`` for the current class
        """
        return True

    @property
    def layer_inputs_edges(self) -> bool:
        r"""
        Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Always ``True`` for the current class
        """
        return True

    @property
    def layer_outputs_edges(self) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Always ``False`` for the current class
        """
        return self.mpnn.layer_outputs_edges

    @property
    def out_dim_factor(self) -> int:
        r"""
        Get the factor by which the output dimension is multiplied for
        the next layer.

        For standard layers, this will return ``1``.

        But for others, such as ``GatLayer``, the output is the concatenation
        of the outputs from each head, so the out_dim gets multiplied by
        the number of heads, and this function should return the number
        of heads.

        Returns:

            int:
                Always ``1`` for the current class
        """
        return 1


# class GPSLayerPygOLD(nn.Module):
#     """Local MPNN + full graph attention x-former layer.
#     """

#     def __init__(self, dim_h,
#                  local_gnn_type, global_model_type, num_heads,
#                  pna_degrees=None, equivstable_pe=False, dropout=0.0,
#                  attn_dropout=0.0, layer_norm=False, batch_norm=True):
#         super().__init__()

#         self.dim_h = dim_h
#         self.num_heads = num_heads
#         self.attn_dropout = attn_dropout
#         self.layer_norm = layer_norm
#         self.batch_norm = batch_norm
#         self.equivstable_pe = equivstable_pe

#         # Local message-passing model.
#         if local_gnn_type == 'None':
#             self.local_model = None
#         elif local_gnn_type == 'GENConv':
#             self.local_model = pygnn.GENConv(dim_h, dim_h)
#         elif local_gnn_type == 'GINE':
#             # gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
#             #                        nn.ReLU(),
#             #                        Linear_pyg(dim_h, dim_h))
#             self.local_model = GINEConvPyg(in_dim=dim_h,
#                                             out_dim=dim_h,
#                                             activation="relu",
#                                             dropout=attn_dropout)
#         elif local_gnn_type == 'GAT':
#             self.local_model = pygnn.GATConv(in_channels=dim_h,
#                                              out_channels=dim_h // num_heads,
#                                              heads=num_heads,
#                                              edge_dim=dim_h)
#         elif local_gnn_type == 'PNA':
#             # Defaults from the paper.
#             # aggregators = ['mean', 'min', 'max', 'std']
#             # scalers = ['identity', 'amplification', 'attenuation']
#             aggregators = ['mean', 'max', 'sum']
#             scalers = ['identity']
#             deg = torch.from_numpy(np.array(pna_degrees))
#             self.local_model = pygnn.PNAConv(dim_h, dim_h,
#                                              aggregators=aggregators,
#                                              scalers=scalers,
#                                              deg=deg,
#                                              edge_dim=16, # dim_h,
#                                              towers=1,
#                                              pre_layers=1,
#                                              post_layers=1,
#                                              divide_input=False)
#         #! andy: check equivstable_pe=equivstable_pe
#         elif local_gnn_type == 'CustomGatedGCN':
#             self.local_model =  GatedGCNPyg(dim_h, dim_h,
#                                              dropout=dropout)
#         else:
#             raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
#         self.local_gnn_type = local_gnn_type

#         # Global attention transformer-style model.
#         if global_model_type == 'None':
#             self.self_attn = None
#         elif global_model_type == 'Transformer':
#             self.self_attn = torch.nn.MultiheadAttention(
#                 dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
#         else:
#             raise ValueError(f"Unsupported global x-former model: "
#                              f"{global_model_type}")

#         '''
#         #!Andy, don't implement Performer and BigBird for now
#         '''
#         # elif global_model_type == 'Performer':
#         #     self.self_attn = SelfAttention(
#         #         dim=dim_h, heads=num_heads,
#         #         dropout=self.attn_dropout, causal=False)
#         # elif global_model_type == "BigBird":
#         #     bigbird_cfg.dim_hidden = dim_h
#         #     bigbird_cfg.n_heads = num_heads
#         #     bigbird_cfg.dropout = dropout
#         #     self.self_attn = SingleBigBirdLayer(bigbird_cfg)
#         self.global_model_type = global_model_type

#         if self.layer_norm and self.batch_norm:
#             raise ValueError("Cannot apply two types of normalization together")

#         # Normalization for MPNN and Self-Attention representations.
#         if self.layer_norm:
#             self.norm1_local = pygnn.norm.GraphNorm(dim_h)
#             self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
#         if self.batch_norm:
#             self.norm1_local = nn.BatchNorm1d(dim_h)
#             self.norm1_attn = nn.BatchNorm1d(dim_h)
#         self.dropout_local = nn.Dropout(dropout)
#         self.dropout_attn = nn.Dropout(dropout)

#         # Feed Forward block.
#         self.activation = F.relu
#         self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
#         self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
#         if self.layer_norm:
#             # self.norm2 = pygnn.norm.LayerNorm(dim_h)
#             self.norm2 = pygnn.norm.GraphNorm(dim_h)
#             # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
#         if self.batch_norm:
#             self.norm2 = nn.BatchNorm1d(dim_h)
#         self.ff_dropout1 = nn.Dropout(dropout)
#         self.ff_dropout2 = nn.Dropout(dropout)


#     '''
#     #? Andy's notes 
#     gps layer forward function work as follows:
#     1. specify a local MPNN model which can handle edge attributes like GINE
#     2. specify a self attention scheme
    
#     '''

    # def forward(self, batch):
    #     h = batch.x
    #     h_in1 = h  # for first residual connection

    #     h_out_list = []
    #     # Local MPNN with edge attributes.
    #     if self.local_model is not None:
    #         self.local_model: pygnn.conv.MessagePassing  # Typing hint.
    #         if self.local_gnn_type == 'CustomGatedGCN':
    #             es_data = None
    #             if self.equivstable_pe:
    #                 es_data = batch.pe_EquivStableLapPE
    #             local_out = self.local_model(Batch(batch=batch,
    #                                                x=h,
    #                                                edge_index=batch.edge_index,
    #                                                edge_attr=batch.edge_attr,
    #                                                pe_EquivStableLapPE=es_data))
    #             # GatedGCN does residual connection and dropout internally.
    #             h_local = local_out.x
    #             batch.edge_attr = local_out.edge_attr
    #         else:
    #             if self.equivstable_pe:
    #                 h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
    #                                            batch.pe_EquivStableLapPE)
    #             else:
    #                 h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
    #             h_local = self.dropout_local(h_local)
    #             h_local = h_in1 + h_local  # Residual connection.

    #         if self.layer_norm:
    #             h_local = self.norm1_local(h_local, batch.batch)
    #         if self.batch_norm:
    #             h_local = self.norm1_local(h_local)
    #         h_out_list.append(h_local)

    #     # Multi-head attention.
    #     if self.self_attn is not None:
    #         h_dense, mask = to_dense_batch(h, batch.batch)
    #         if self.global_model_type == 'Transformer':
    #             h_attn = self._sa_block(h_dense, None, ~mask)[mask]
    #         elif self.global_model_type == 'Performer':
    #             h_attn = self.self_attn(h_dense, mask=mask)[mask]
    #         elif self.global_model_type == 'BigBird':
    #             h_attn = self.self_attn(h_dense, attention_mask=mask)
    #         else:
    #             raise RuntimeError(f"Unexpected {self.global_model_type}")

    #         h_attn = self.dropout_attn(h_attn)
    #         h_attn = h_in1 + h_attn  # Residual connection.
    #         if self.layer_norm:
    #             h_attn = self.norm1_attn(h_attn, batch.batch)
    #         if self.batch_norm:
    #             h_attn = self.norm1_attn(h_attn)
    #         h_out_list.append(h_attn)

    #     # Combine local and global outputs.
    #     # h = torch.cat(h_out_list, dim=-1)
    #     h = sum(h_out_list)

    #     # Feed Forward block.
    #     h = h + self._ff_block(h)
    #     if self.layer_norm:
    #         h = self.norm2(h, batch.batch)
    #     if self.batch_norm:
    #         h = self.norm2(h)
    #     batch.x = h
    #     return batch

#     def _sa_block(self, x, attn_mask, key_padding_mask):
#         """Self-attention block.
#         """
#         x = self.self_attn(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return x

#     def _ff_block(self, x):
#         """Feed Forward block.
#         """
#         x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
#         return self.ff_dropout2(self.ff_linear2(x))

#     def extra_repr(self):
#         s = f'summary: dim_h={self.dim_h}, ' \
#             f'local_gnn_type={self.local_gnn_type}, ' \
#             f'global_model_type={self.global_model_type}, ' \
#             f'heads={self.num_heads}'
#         return s
