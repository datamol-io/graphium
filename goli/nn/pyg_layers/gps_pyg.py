"""
adapated from https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gps_layer.py
"""

from copy import deepcopy
from typing import Callable, Union, Optional


from goli.nn.base_graph_layer import BaseGraphModule
from goli.nn.base_layers import FCLayer, MultiheadAttentionMup
from goli.nn.pyg_layers import GatedGCNPyg, GINConvPyg, GINEConvPyg, PNAMessagePassingPyg, MPNNPlusPyg
from goli.utils.decorators import classproperty
from goli.ipu.to_dense_batch import to_dense_batch, to_sparse_batch
from goli.ipu.ipu_utils import import_poptorch

PYG_LAYERS_DICT = {
    "pyg:gin": GINConvPyg,
    "pyg:gine": GINEConvPyg,
    "pyg:gated-gcn": GatedGCNPyg,
    "pyg:pna-msgpass": PNAMessagePassingPyg,
    "pyg:mpnnplus": MPNNPlusPyg,
}

ATTENTION_LAYERS_DICT = {
    "full-attention": MultiheadAttentionMup,
    "none": None,
}


class GPSLayerPyg(BaseGraphModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: Optional[int] = None,
        out_dim_edges: Optional[int] = None,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        node_residual: Optional[bool] = True,
        normalization: Union[str, Callable] = "none",
        mpnn_type: str = "pyg:gine",
        mpnn_kwargs=None,
        attn_type: str = "full-attention",
        attn_kwargs=None,
        droppath_rate_attn: float = 0.0,
        droppath_rate_ffn: float = 0.0,
        **kwargs,
    ):
        r"""
        GPS: Recipe for a General, Powerful, Scalable Graph Transformer
        Ladislav Rampášek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, Dominique Beaini
        https://arxiv.org/abs/2205.12454

        GPS++: An Optimised Hybrid MPNN/Transformer for Molecular Property Prediction
        Dominic Masters, Josef Dean, Kerstin Klaser, Zhiyi Li, Sam Maddrell-Mander, Adam Sanders, Hatem Helal, Deniz Beker, Ladislav Rampášek, Dominique Beaini
        https://arxiv.org/abs/2212.02229

        Parameters:

            in_dim:
                Input node feature dimensions of the layer

            out_dim:
                Output node feature dimensions of the layer

            in_dim:
                Input edge feature dimensions of the layer

            out_dim:
                Output edge feature dimensions of the layer

            activation:
                activation function to use in the layer

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            node_residual:
                If node residual is used after on the gnn layer output

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            mpnn_type:
                type of mpnn used, choose from "pyg:gin", "pyg:gine", "pyg:gated-gcn", "pyg:pna-msgpass" and "pyg:mpnnplus"

            mpnn_kwargs:
                kwargs for mpnn layer

            attn_type:
                type of attention used, choose from "full-attention" and "none"

            attn_kwargs:
                kwargs for attention layer

            droppath_rate_attn:
                stochastic depth drop rate for attention layer

            droppath_rate_ffn:
                stochastic depth drop rate for ffn layer

        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            droppath_rate=droppath_rate_attn,
            **kwargs,
        )

        # Dropout layers
        self.dropout_local = self.dropout_layer
        self.dropout_attn = self._parse_dropout(dropout=self.dropout)
        self.ff_dropout1 = self._parse_dropout(dropout=self.dropout)
        self.ff_dropout2 = self._parse_dropout(dropout=self.dropout)

        # Residual connections
        self.node_residual = node_residual

        # linear layers
        self.ff_linear1 = FCLayer(in_dim, in_dim * 2, activation=None)
        self.ff_linear2 = FCLayer(in_dim * 2, in_dim, activation=None)
        self.ff_out = FCLayer(in_dim, out_dim, activation=None)

        # Normalization layers
        self.norm_layer_local = self._parse_norm(normalization=self.normalization, dim=in_dim)
        self.norm_layer_attn = self._parse_norm(normalization=self.normalization, dim=in_dim)
        self.norm_layer_ff = self._parse_norm(self.normalization)

        # Set the default values for the MPNN layer
        if mpnn_kwargs is None:
            mpnn_kwargs = {}
        mpnn_kwargs = deepcopy(mpnn_kwargs)
        mpnn_kwargs.setdefault("in_dim", in_dim)
        mpnn_kwargs.setdefault("out_dim", in_dim)
        mpnn_kwargs.setdefault("in_dim_edges", in_dim_edges)
        mpnn_kwargs.setdefault("out_dim_edges", out_dim_edges)
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

        # Drop path layers
        self.droppath_attn = self._parse_droppath(droppath_rate_attn)
        self.droppath_ffn = self.droppath_layer

        # Initialize the Attention layer
        self.attn_layer = self._parse_attn_layer(attn_type, **attn_kwargs)

    def forward(self, batch):
        # pe, h, edge_index, edge_attr = batch.pos_enc_feats_sign_flip, batch.h, batch.edge_index, batch.edge_attr
        h = batch.h

        h_in = h  # for first residual connection

        # Local MPNN with edge attributes.
        batch_out = self.mpnn(batch.clone())
        h_local = batch_out.h
        if self.dropout_local is not None:
            h_local = self.dropout_local(h_local)
        if self.node_residual:
            h_local = h_in + h_local  # Residual connection for nodes, not used in gps++.
        if self.norm_layer_local is not None:
            h_local = self.norm_layer_local(h_local)
        h = h_local

        # Multi-head attention.
        # * batch.batch is the indicator vector for nodes of which graph it belongs to
        # * h_dense
        # Check whether the model runs on IPU, if so define a maximal number of nodes per graph when reshaping
        poptorch = import_poptorch(raise_error=False)
        on_ipu = (poptorch is not None) and (poptorch.isRunningOnIpu())
        batch_size = None if h.device.type != "ipu" else batch.graph_is_true.shape[0]
        if self.attn_layer is not None:
            max_num_nodes_per_graph = None
            if on_ipu:
                max_num_nodes_per_graph = self.max_num_nodes_per_graph

            # Convert the tensor to a dense batch, then back to a sparse batch
            h_dense, mask, idx = to_dense_batch(
                h,
                batch=batch.batch,
                batch_size=batch_size,
                max_num_nodes_per_graph=max_num_nodes_per_graph,
                drop_nodes_last_graph=on_ipu,
            )
            h_attn = self._sa_block(h_dense, None, ~mask)
            h_attn = to_sparse_batch(h_attn, idx)
            if self.droppath_attn is not None:
                h_attn = self.droppath_attn(h_attn, batch.batch, batch_size)

            # Dropout, residual, norm
            if self.dropout_attn is not None:
                h_attn = self.dropout_attn(h_attn)
            h_attn = h_in + h_attn
            if self.norm_layer_attn is not None:
                h_attn = self.norm_layer_attn(h_attn)

            # Combine local and global outputs.
            h = h + h_attn

        # Feed Forward block.
        h = self._ff_block(h, batch.batch, batch_size)

        batch_out.h = h

        return batch_out

    def _parse_attn_layer(self, attn_type, **attn_kwargs):
        attn_layer, attn_class = None, None
        if attn_type is not None:
            attn_class = ATTENTION_LAYERS_DICT[attn_type]
        if attn_class is not None:
            attn_layer = attn_class(**attn_kwargs)
        return attn_layer

    def _ff_block(self, h, batch, batch_size):
        """Feed Forward block."""
        h_in = h
        # First linear layer + activation + dropout
        h = self.ff_linear1(h)
        if self.activation_layer is not None:
            h = self.activation_layer(h)
        if self.ff_dropout1 is not None:
            h = self.ff_dropout1(h)

        # Second linear layer + dropout
        h = self.ff_linear2(h)
        if self.ff_dropout2 is not None:
            h = self.ff_dropout2(h)

        if self.droppath_ffn is not None:
            h = self.droppath_ffn(h, batch, batch_size)

        # Residual
        h = h + h_in

        # Third linear layer + norm
        h = self.ff_out(h)
        if self.norm_layer_ff is not None:
            h = self.norm_layer_ff(h)
        return h

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        x = self.attn_layer(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        return x

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
