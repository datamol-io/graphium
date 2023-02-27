"""
adapated from https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gps_layer.py
"""

from copy import deepcopy
from typing import Callable, Union, Optional, Dict, Any
from torch.nn import Module
from torch import Tensor
from torch_geometric.data import Batch

from goli.nn.base_graph_layer import BaseGraphModule
from goli.nn.base_layers import FCLayer, MultiheadAttentionMup, MLP, get_activation_str
from goli.nn.pyg_layers import (
    GatedGCNPyg,
    GINConvPyg,
    GINEConvPyg,
    PNAMessagePassingPyg,
    MPNNPlusPyg,
)
from goli.utils.decorators import classproperty
from goli.ipu.to_dense_batch import to_dense_batch, to_sparse_batch
from goli.ipu.ipu_utils import is_running_on_ipu

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
        biased_attention_key: Optional[str] = None,
        attn_kwargs=None,
        droppath_rate_attn: float = 0.0,
        droppath_rate_ffn: float = 0.0,
        hidden_dim_scaling: float = 4.0,
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
                stochastic depth drop rate for attention layer https://arxiv.org/abs/1603.09382

            droppath_rate_ffn:
                stochastic depth drop rate for ffn layer https://arxiv.org/abs/1603.09382

            mpnn_type:
                Type of MPNN layer to use. Choices specified in PYG_LAYERS_DICT

            biased_attention_key:
                indicates if biased attention is used by specifying a key corresponding to the pyg attribute in the batch (processed by the gaussian kernel encoder)
                default: None means biased attention is not used

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
        # Set the other attributes
        self.in_dim_edges = in_dim_edges
        self.out_dim_edges = out_dim_edges

        # Dropout layers
        self.dropout_local = self.dropout_layer
        self.dropout_attn = self._parse_dropout(dropout=self.dropout)

        # DropPath layers
        self.droppath_ffn = self._parse_droppath(droppath_rate_ffn)

        # Residual connections
        self.node_residual = node_residual

        # MLP applied at the end of the GPS layer
        self.mlp = MLP(
            in_dim=in_dim,
            hidden_dims=int(hidden_dim_scaling * in_dim),
            out_dim=in_dim,
            depth=2,
            activation=activation,
            dropout=self.dropout,
        )
        self.f_out = FCLayer(in_dim, out_dim, normalization=normalization)

        # Normalization layers
        self.norm_layer_local = self._parse_norm(normalization=self.normalization, dim=in_dim)
        self.norm_layer_attn = self._parse_norm(normalization=self.normalization, dim=in_dim)
        self.norm_layer_ff = self._parse_norm(self.normalization)

        self.biased_attention_key = biased_attention_key

        # Initialize the MPNN and Attention layers
        self.mpnn = self._parse_mpnn_layer(mpnn_type, mpnn_kwargs)
        self.attn_layer = self._parse_attn_layer(attn_type, self.biased_attention_key, attn_kwargs)

    def forward(self, batch: Batch) -> Batch:
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

        # Multi-head attention.
        if self.attn_layer is not None:
            h_attn = self._self_attention_block(h, h_in, batch)
            # Combine local and global outputs.
            h = h_local + h_attn
        else:
            h = h_local

        # MLP block, with skip connection
        h = h + self.mlp(h)
        h = self.f_out(h)
        # Add the droppath to the output of the MLP
        batch_size = None if h.device.type != "ipu" else batch.graph_is_true.shape[0]
        if self.droppath_ffn is not None:
            h = self.droppath_ffn(h, batch.batch, batch_size)


        batch_out.h = h

        return batch_out

    def _parse_mpnn_layer(self, mpnn_type, mpnn_kwargs: Dict[str, Any]) -> Optional[Module]:
        """Parse the MPNN layer."""

        mpnn_kwargs = deepcopy(mpnn_kwargs)
        if mpnn_kwargs is None:
            mpnn_kwargs = {}

        # Set the default values
        mpnn_kwargs = deepcopy(mpnn_kwargs)
        mpnn_kwargs.setdefault("in_dim", self.in_dim)
        mpnn_kwargs.setdefault("out_dim", self.in_dim)
        mpnn_kwargs.setdefault("in_dim_edges", self.in_dim_edges)
        mpnn_kwargs.setdefault("out_dim_edges", self.out_dim_edges)
        # TODO: The rest of default values
        self.mpnn_kwargs = mpnn_kwargs

        # Initialize the MPNN layer
        mpnn_class = PYG_LAYERS_DICT[mpnn_type]
        mpnn_layer = mpnn_class(**mpnn_kwargs, layer_depth=self.layer_depth, layer_idx=self.layer_idx)

        return mpnn_layer

    def _parse_attn_layer(self, attn_type, biased_attention_key: str, attn_kwargs: Dict[str, Any]) -> Optional[Module]:
        """Parse the attention layer."""

        # Set the default values for the Attention layer
        if attn_kwargs is None:
            attn_kwargs = {}
        attn_kwargs.setdefault("embed_dim", self.in_dim)
        attn_kwargs.setdefault("num_heads", 1)
        attn_kwargs.setdefault("dropout", self.dropout)
        attn_kwargs.setdefault("batch_first", True)
        self.attn_kwargs = attn_kwargs

        # Initialize the Attention layer
        attn_layer, attn_class = None, None
        if attn_type is not None:
            attn_class = ATTENTION_LAYERS_DICT[attn_type]
        if attn_class is not None:
            attn_layer = attn_class(biased_attention_key, **attn_kwargs)
        return attn_layer



    def _self_attention_block(self, h: Tensor, h_in: Tensor, batch: Batch) -> Tensor:
        """
        Applying the multi-head self-attention to the batch of graphs.
        First the batch is converted from [num_nodes, hidden_dim] to [num_graphs, max_num_nodes, hidden_dim]
        Then the self-attention is applied on each graph
        Then the batch is converted again to [num_nodes, hidden_dim]
        """

        # Multi-head attention.
        on_ipu = is_running_on_ipu()
        max_num_nodes_per_graph = None
        if on_ipu:
            max_num_nodes_per_graph = self.max_num_nodes_per_graph

        # Convert the tensor to a dense batch, then back to a sparse batch
        batch_size = None if h.device.type != "ipu" else batch.graph_is_true.shape[0]

        # h[num_nodes, hidden_dim] -> h_dense[num_graphs, max_num_nodes, hidden_dim]
        h_dense, mask, idx = to_dense_batch(
            h,
            batch=batch.batch,  # The batch index as a vector that indicates for nodes of which graph it belongs to
            batch_size=batch_size,
            max_num_nodes_per_graph=max_num_nodes_per_graph,
            drop_nodes_last_graph=on_ipu,
        )

        attn_bias = None
        if self.biased_attention_key is not None:
            attn_bias = batch[self.biased_attention_key]

        # h_dense[num_graphs, max_num_nodes, hidden_dim] -> h_attn[num_graphs, max_num_nodes, hidden_dim]
        h_attn = self._sa_block(h_dense, attn_bias=attn_bias, attn_mask=None, key_padding_mask=~mask)

        # h_attn[num_graphs, max_num_nodes, hidden_dim] -> h_attn[num_nodes, hidden_dim]
        h_attn = to_sparse_batch(h_attn, idx)

        # Dropout, residual, norm
        if self.dropout_attn is not None:
            h_attn = self.dropout_attn(h_attn)
        h_attn = h_in + h_attn
        if self.norm_layer_attn is not None:
            h_attn = self.norm_layer_attn(h_attn)
        if self.droppath_layer is not None:
            self.droppath_layer(h_attn, batch.batch, batch_size)

        # Combine local and global outputs.
        return h + h_attn

    def _sa_block(self, x, attn_bias, attn_mask=None, key_padding_mask=None):
        """Self-attention block."""
        x = self.attn_layer(
            x,
            x,
            x,
            attn_bias=attn_bias,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
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
