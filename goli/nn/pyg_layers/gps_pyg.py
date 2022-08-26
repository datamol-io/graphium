'''
#!Andy
adapated from https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gps_layer.py
'''

import torch
import torch.nn as nn
from torch_scatter import scatter
from typing import Callable, Union, Optional


from goli.nn.base_graph_layer import BaseGraphModule
from goli.nn.pyg_layers import GatedGCNPyg, GINConvPyg, GINEConvPyg, PNAMessagePassingPyg
from goli.utils.decorators import classproperty
from goli.ipu.to_dense_batch import to_dense_batch


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

        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
        )

        # Dropout layers
        self.dropout_local = self.dropout_layer
        self.dropout_attn = self._parse_dropout(dropout=self.dropout)
        self.ff_dropout1 = self._parse_dropout(dropout=self.dropout)
        self.ff_dropout2 = self._parse_dropout(dropout=self.dropout)

        # Linear layers
        self.ff_linear1 = nn.Linear(in_dim, in_dim * 2)
        self.ff_linear2 = nn.Linear(in_dim * 2, in_dim)
        self.ff_out = nn.Linear(in_dim, out_dim)

        # Normalization layers
        self.norm_layer_local = self._parse_norm(normalization=self.normalization, dim=in_dim)
        self.norm_layer_attn = self._parse_norm(normalization=self.normalization, dim=in_dim)
        self.norm_layer_ff = self.norm_layer

        # Set the default values for the MPNN layer
        if (mpnn_kwargs is None):
            mpnn_kwargs = {}
        mpnn_kwargs.setdefault("in_dim", in_dim)
        mpnn_kwargs.setdefault("out_dim", in_dim)
        mpnn_kwargs.setdefault("in_dim_edges", in_dim_edges)
        # TODO: The rest of default values

        # Initialize the MPNN layer
        mpnn_class = PYG_LAYERS_DICT[mpnn_type]
        self.mpnn = mpnn_class(**mpnn_kwargs)

        # Set the default values for the Attention layer
        if attn_kwargs is None:
            attn_kwargs = {}
        attn_kwargs.setdefault("embed_dim", in_dim)
        attn_kwargs.setdefault("num_heads", 1)
        attn_kwargs.setdefault("dropout", dropout)
        attn_kwargs.setdefault("batch_first", True)

        # Initialize the Attention layer
        attn_class = ATTENTION_LAYERS_DICT[attn_type]
        self.attn_layer = attn_class(**attn_kwargs)


    def forward(self, batch):
        # pe, h, edge_index, edge_attr = batch.pos_enc_feats_sign_flip, batch.h, batch.edge_index, batch.edge_attr
        h = batch.h

        h_in = h  # for first residual connection

        # Local MPNN with edge attributes.
        batch_out = (self.mpnn(batch.clone()))
        h_local = batch_out.h
        h_local = self.dropout_local(h_local)
        h_local = h_in + h_local  # Residual connection.
        if (self.norm_layer_local):
            h_local = self.norm_layer_local(h_local)

        # Multi-head attention.
        #* batch.batch is the indicator vector for nodes of which graph it belongs to
        #* h_dense
        if self.attn_layer is not None:

            # TODO: Better way to determine if on IPU? Here we're checking for padding
            on_ipu = ("graph_is_true" in batch.keys) and (not batch.graph_is_true.all())

            if on_ipu:
                max_num_nodes_per_graph = batch.dataset_max_nodes_per_graph[0].item()
            else:
                max_num_nodes_per_graph = None
            h_dense, mask = to_dense_batch(h, batch.batch, max_num_nodes_per_graph=max_num_nodes_per_graph, drop_nodes_last_graph=on_ipu)
            h_attn = self._sa_block(h_dense, None, ~mask) #[mask]
            h_attn = self._to_sparse_batch(h_dense=h_dense, mask=mask, sparse_shape=h.shape, on_ipu=on_ipu)

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in + h_attn  # Residual connection.
            if self.norm_layer_attn:
                h_attn = self.norm_layer_attn(h_attn)

        # Combine local and global outputs.
        h = h_local + h_attn

        # Feed Forward block.
        h = self._ff_block(h)

        batch_out.h = h

        return batch_out

    @staticmethod
    def _to_sparse_batch(h_dense, mask, sparse_shape, on_ipu):
        # TODO: Unit-Test this function
        if on_ipu:
            # Indexing not available on IPU. The 'hack' belows allows to index using the `scatter` function
            # by scattering all true nodes individually, and fake nodes into the same element.
            mask_expand = mask.unsqueeze(-1).expand(h_dense.shape).flatten()
            mask_idx = torch.cumsum(mask_expand, dim=0)
            mask_idx[~mask_expand] = 0
            h_sparse = scatter(h_dense.flatten(), mask_idx, reduce="sum", dim_size=torch.prod(torch.as_tensor(sparse_shape))+1)
            h_sparse = h_sparse[1:].reshape(sparse_shape)
        else:
            # Simply index the tensor.
            h_sparse = h_dense[mask]

        return h_sparse

    def _ff_block(self, h):
        """Feed Forward block.
        """
        h_in = h
        # First linear layer + activation + dropout
        if (self.activation_layer is None):
            h = self.ff_dropout1(self.ff_linear1(h))
        else:
            h = self.ff_dropout1(self.activation_layer(self.ff_linear1(h)))

        # Second linear layer + dropout
        h = self.ff_dropout2(self.ff_linear2(h))

        # Residual
        h = h + h_in

        # Third linear layer + norm
        h = self.ff_out(h)
        if self.norm_layer_ff:
            h = self.norm_layer_ff(h)
        return h

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

