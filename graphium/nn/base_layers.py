from typing import Union, Callable, Optional, Type, Tuple, Iterable
from copy import deepcopy
from loguru import logger

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, IntTensor
import mup.init as mupi
from mup import set_base_shapes, MuReadout
from torch.nn.functional import linear

from graphium.ipu.ipu_utils import is_running_on_ipu

SUPPORTED_ACTIVATION_MAP = {
    "ReLU",
    "Sigmoid",
    "Tanh",
    "ELU",
    "SELU",
    "GLU",
    "GELU",
    "LeakyReLU",
    "Softplus",
    "None",
}


def get_activation(activation: Union[type(None), str, Callable]) -> Optional[Callable]:
    r"""
    returns the activation function represented by the input string

    Parameters:
        activation: Callable, `None`, or string with value:
            "none", "ReLU", "Sigmoid", "Tanh", "ELU", "SELU", "GLU", "GELU", "LeakyReLU", "Softplus"

    Returns:
        Callable or None: The activation function
    """
    if (activation is not None) and callable(activation):
        # activation is already a function
        return activation

    if (activation is None) or (activation.lower() == "none"):
        return None

    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(
        activation[0], str
    ), f"Unhandled activation function {activation} of type {type(activation)}"
    activation = activation[0]

    return vars(torch.nn.modules.activation)[activation]()


def get_activation_str(activation: Union[type(None), str, Callable]) -> str:
    r"""
    returns the string related to the activation function

    Parameters:
        activation: Callable, `None`, or string with value:
            "none", "ReLU", "Sigmoid", "Tanh", "ELU", "SELU", "GLU", "LeakyReLU", "Softplus"

    Returns:
        The name of the activation function
    """

    if isinstance(activation, str):
        return activation

    if activation is None:
        return "None"

    if isinstance(activation, Callable):
        return activation.__class__._get_name(activation)
    else:
        raise ValueError(f"Unhandled activation function {activation} of type {type(activation)}")


def get_norm(normalization: Union[Type[None], str, Callable], dim: Optional[int] = None):
    r"""
    returns the normalization function represented by the input string

    Parameters:
        normalization: Callable, `None`, or string with value:
            "none", "batch_norm", "layer_norm"
        dim: Dimension where to apply the norm. Mandatory for 'batch_norm' and 'layer_norm'

    Returns:
        Callable or None: The normalization function
    """
    parsed_norm = None
    if (normalization is None) or (normalization in ["none", "NoneType"]):
        pass
    elif callable(normalization):
        parsed_norm = normalization
    elif normalization in ["batch_norm", "BatchNorm1d"]:
        parsed_norm = nn.BatchNorm1d(dim)
    elif normalization in ["layer_norm", "LayerNorm"]:
        parsed_norm = nn.LayerNorm(dim)
    else:
        raise ValueError(
            f"Undefined normalization `{normalization}`, must be `None`, `Callable`, 'batch_norm', 'layer_norm', 'none'"
        )
    return deepcopy(parsed_norm)


class MultiheadAttentionMup(nn.MultiheadAttention):
    """
    Modifying the MultiheadAttention to work with the muTransfer paradigm.
    The layers are initialized using the mup package.
    The `_scaled_dot_product_attention` normalizes the attention matrix with `1/d` instead of `1/sqrt(d)`
    The biased self-attention option is added to have 3D attention bias.
    """

    def __init__(self, biased_attention, **kwargs):
        super().__init__(**kwargs)
        self.biased_attention = biased_attention

    def _reset_parameters(self):
        set_base_shapes(self, None, rescale_params=False)  # Set the shapes of the tensors, useful for mup
        if self._qkv_same_embed_dim:
            mupi.xavier_uniform_(self.in_proj_weight)
        else:
            mupi.xavier_uniform_(self.q_proj_weight)
            mupi.xavier_uniform_(self.k_proj_weight)
            mupi.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            mupi.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            mupi.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        precision: Optional[str] = "32",
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # attn_bias [batch, num_heads, nodes, nodes]
        if not self.biased_attention or attn_bias is None:
            attn_bias = 0.0
        # assuming source and target have the same sequence length (homogeneous graph attention)
        batch, nodes, hidden = query.size()
        assert (
            hidden == self.embed_dim
        ), f"query hidden dimension {hidden} != embed_dim {self.embed_dim} in class"
        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        scaling_factor = 1 / head_dim  # use head_dim instead of (head_dim**0.5) for mup
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        q_proj_weight, k_proj_weight, v_proj_weight = self.in_proj_weight.chunk(3)
        # [batch, num_heads, nodes, head_size]
        q = linear(query, q_proj_weight, b_q).view(batch, nodes, self.num_heads, -1).transpose(1, 2)
        # [batch, num_heads, nodes, head_size]
        k = linear(key, k_proj_weight, b_k).view(batch, nodes, self.num_heads, -1).transpose(1, 2)
        # [batch, num_heads, nodes, head_size]
        v = linear(value, v_proj_weight, b_v).view(batch, nodes, self.num_heads, -1).transpose(1, 2)
        q = q * scaling_factor
        # [batch, num_heads, nodes, nodes]
        attn_weights = q @ k.transpose(-1, -2)
        # [batch, num_heads, nodes, nodes]
        attn_weights += attn_bias
        key_padding_mask_value = float("-inf") if precision == "32" else -10000
        # key_padding_mask: [batch, 1, 1, nodes]
        if key_padding_mask is not None:
            masked_attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                key_padding_mask_value,
            )
        else:
            masked_attn_weights = attn_weights
        masked_attn_weights = F.softmax(masked_attn_weights, dim=-1)
        attn_probs = F.dropout(masked_attn_weights, p=self.dropout, training=self.training)
        # [batch, num_heads, nodes, nodes] * [batch, num_heads, nodes, head_size] -> [batch, num_heads, nodes, head_size]
        attn = attn_probs @ v
        # [batch, nodes, embd_dim]
        attn = attn.transpose(1, 2).contiguous().view(batch, nodes, self.embed_dim)
        # [batch, nodes, embd_dim]
        out = (self.out_proj(attn), None)
        return out


class TransformerEncoderLayerMup(nn.TransformerEncoderLayer):
    r"""
    Modified version of ``torch.nn.TransformerEncoderLayer`` that uses :math:`1/n`-scaled attention
    for compatibility with muP (as opposed to the original :math:`1/\sqrt{n}` scaling factor)
    Arguments are the same as ``torch.nn.TransformerEncoderLayer``.
    """

    def __init__(self, biased_attention, *args, **kwargs) -> None:
        super(TransformerEncoderLayerMup, self).__init__(*args, **kwargs)

        # Extract arguments passed to __init__ as a dictionary
        signature = inspect.signature(nn.TransformerEncoderLayer.__init__)

        # `self` needs to passed, which makes things tricky, but using this object seems fine for now
        bound_signature = signature.bind(self, *args, **kwargs)
        bound_signature.apply_defaults()

        mha_names = ["embed_dim", "num_heads", "dropout", "batch_first", "device", "dtype"]
        transformer_names = ["d_model", "nhead", "dropout", "batch_first", "device", "dtype"]

        # Override self attention to use muP
        self.self_attn = MultiheadAttentionMup(
            biased_attention,
            **{
                mha_name: bound_signature.arguments[transformer_name]
                for mha_name, transformer_name in zip(mha_names, transformer_names)
            },
        )


class MuReadoutGraphium(MuReadout):
    """
    PopTorch-compatible replacement for `mup.MuReadout`

    Not quite a drop-in replacement for `mup.MuReadout` - you need to specify
    `base_width`.

    Set `base_width` to width of base model passed to `mup.set_base_shapes`
    to get same results on IPU and CPU. Should still "work" with any other
    value, but won't give the same results as CPU
    """

    def __init__(self, in_features, *args, **kwargs):
        super().__init__(in_features, *args, **kwargs)
        self.base_width = in_features

    @property
    def absolute_width(self):
        return float(self.in_features)

    @property
    def base_width(self):
        return self._base_width

    @base_width.setter
    def base_width(self, val):
        if val is None:
            return
        assert isinstance(
            val, (int, torch.int, torch.long)
        ), f"`base_width` must be None, int or long, provided {val} of type {type(val)}"
        self._base_width = val

    def width_mult(self):
        return self.absolute_width / self.base_width


class MuReadoutGraphium(MuReadout):
    """
    PopTorch-compatible replacement for `mup.MuReadout`

    Not quite a drop-in replacement for `mup.MuReadout` - you need to specify
    `base_width`.

    Set `base_width` to width of base model passed to `mup.set_base_shapes`
    to get same results on IPU and CPU. Should still "work" with any other
    value, but won't give the same results as CPU
    """

    def __init__(self, in_features, *args, **kwargs):
        super().__init__(in_features, *args, **kwargs)
        self.base_width = in_features

    @property
    def absolute_width(self):
        return float(self.in_features)

    @property
    def base_width(self):
        return self._base_width

    @base_width.setter
    def base_width(self, val):
        if val is None:
            return
        assert isinstance(
            val, (int, torch.int, torch.long)
        ), f"`base_width` must be None, int or long, provided {val} of type {type(val)}"
        self._base_width = val

    def width_mult(self):
        return self.absolute_width / self.base_width


class FCLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        bias: bool = True,
        init_fn: Optional[Callable] = None,
        is_readout_layer: bool = False,
        droppath_rate: float = 0.0,
    ):
        r"""
        A simple fully connected and customizable layer. This layer is centered around a `torch.nn.Linear` module.
        The order in which transformations are applied is:

        - Dense Layer
        - Activation
        - Dropout (if applicable)
        - Batch Normalization (if applicable)

        Parameters:
            in_dim:
                Input dimension of the layer (the `torch.nn.Linear`)
            out_dim:
                Output dimension of the layer.
            dropout:
                The ratio of units to dropout. No dropout by default.
            activation:
                Activation function to use.
            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function
            bias:
                Whether to enable bias in for the linear layer.
            init_fn:
                Initialization function to use for the weight of the layer. Default is
                $$\mathcal{U}(-\sqrt{k}, \sqrt{k})$$ with $$k=\frac{1}{ \text{in_dim}}$$
            is_readout_layer: Whether the layer should be treated as a readout layer by replacing of `torch.nn.Linear`
                by `mup.MuReadout` from the muTransfer method https://github.com/microsoft/mup

            droppath_rate:
                stochastic depth drop rate, between 0 and 1, see https://arxiv.org/abs/1603.09382
        Attributes:
            dropout (int):
                The ratio of units to dropout.
            normalization (None or Callable):
                Normalization layer
            linear (`torch.nn.Linear`):
                The linear layer
            activation (`torch.nn.Module`):
                The activation layer
            init_fn (Callable):
                Initialization function used for the weight of the layer
            in_dim (int):
                Input dimension of the linear layer
            out_dim (int):
                Output dimension of the linear layer
        """

        super().__init__()

        self.__params = locals()
        del self.__params["__class__"]
        del self.__params["self"]

        # Basic parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.dropout = None
        self.normalization = get_norm(normalization, dim=out_dim)

        # Dropout and activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        self.activation = get_activation(activation)

        self.drop_path = None
        if droppath_rate > 0:
            self.drop_path = DropPath(droppath_rate)

        # Linear layer, or MuReadout layer
        if not is_readout_layer:
            self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            self.linear = MuReadoutGraphium(in_dim, out_dim, bias=bias)

            # Warn user in case of weird parameters
            if self.normalization is not None:
                logger.warning(
                    f"Normalization is not `None` for the readout layer. Provided {self.normalization}"
                )
            if (self.dropout is not None) and (self.dropout.p > 0):
                logger.warning(f"Dropout is not `None` or `0` for the readout layer. Provided {self.dropout}")

        # Define the initialization function based on `muTransfer`, and reset the parameters
        self.init_fn = init_fn if init_fn is not None else mupi.xavier_uniform_
        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        """
        Reset the parameters of the linear layer using the `init_fn`.
        """
        set_base_shapes(self, None, rescale_params=False)  # Set the shapes of the tensors, useful for mup
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the FC layer on the input features.

        Parameters:

            h: `torch.Tensor[..., Din]`:
                Input feature tensor, before the FC.
                `Din` is the number of input features

        Returns:

            `torch.Tensor[..., Dout]`:
                Output feature tensor, after the FC.
                `Dout` is the number of output features

        """

        if torch.prod(torch.as_tensor(h.shape[:-1])) == 0:
            h = torch.zeros(
                list(h.shape[:-1]) + [self.linear.out_features],
                device=h.device,
                dtype=h.dtype,
            )
            return h

        h = self.linear(h)

        if self.normalization is not None:
            if h.shape[1] != self.out_dim:
                h = self.normalization(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.normalization(h)

        if self.dropout is not None:
            h = self.dropout(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.drop_path is not None:
            h = self.drop_path(h)

        return h

    @property
    def in_channels(self) -> int:
        r"""
        Get the input channel size. For compatibility with PyG.
        """
        return self.in_dim

    @property
    def out_channels(self) -> int:
        r"""
        Get the output channel size. For compatibility with PyG.
        """
        return self.out_dim

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_dim} -> {self.out_dim}, activation={self.activation})"


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Union[Iterable[int], int],
        out_dim: int,
        depth: int,
        activation: Union[str, Callable] = "relu",
        last_activation: Union[str, Callable] = "none",
        dropout: float = 0.0,
        last_dropout: float = 0.0,
        normalization: Union[Type[None], str, Callable] = "none",
        last_normalization: Union[Type[None], str, Callable] = "none",
        first_normalization: Union[Type[None], str, Callable] = "none",
        last_layer_is_readout: bool = False,
        droppath_rate: float = 0.0,
        constant_droppath_rate: bool = True,
    ):
        r"""
        Simple multi-layer perceptron, built of a series of FCLayers

        Parameters:
            in_dim:
                Input dimension of the MLP
            hidden_dims:
                Either an integer specifying all the hidden dimensions,
                or a list of dimensions in the hidden layers.
            out_dim:
                Output dimension of the MLP.
            depth:
                If `hidden_dims` is an integer, `depth` is 1 + the number of
                hidden layers to use.
                If `hidden_dims` is a list, then
                `depth` must be `None` or equal to `len(hidden_dims) + 1`
            activation:
                Activation function to use in all the layers except the last.
                if `layers==1`, this parameter is ignored
            last_activation:
                Activation function to use in the last layer.
            dropout:
                The ratio of units to dropout. Must be between 0 and 1
            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization in the hidden layers.
                - `Callable`: Any callable function

                if `layers==1`, this parameter is ignored
            last_normalization:
                Norrmalization to use **after the last layer**. Same options as `normalization`.
            first_normalization:
                Norrmalization to use in **before the first layer**. Same options as `normalization`.
            last_dropout:
                The ratio of units to dropout at the last layer.
            last_layer_is_readout: Whether the last layer should be treated as a readout layer.
                Allows to use the `mup.MuReadout` from the muTransfer method https://github.com/microsoft/mup
            droppath_rate:
                stochastic depth drop rate, between 0 and 1.
                See https://arxiv.org/abs/1603.09382
            constant_droppath_rate:
                If `True`, drop rates will remain constant accross layers.
                Otherwise, drop rates will vary stochastically.
                See `DropPath.get_stochastic_drop_rate`
        """

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Parse the hidden dimensions and depth
        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * (depth - 1)
        else:
            self.hidden_dims = list(hidden_dims)
            assert (depth is None) or (
                depth == len(self.hidden_dims) + 1
            ), "Mismatch between the provided network depth from `hidden_dims` and `depth`"
        self.depth = len(self.hidden_dims) + 1

        # Parse the normalization
        self.first_normalization = get_norm(first_normalization, dim=in_dim)

        all_dims = [in_dim] + self.hidden_dims + [out_dim]
        fully_connected = []
        if depth == 0:
            self.fully_connected = None
            return
        else:
            for ii in range(depth):
                if ii < (depth - 1):
                    # Define the parameters for all intermediate layers
                    this_activation = activation
                    this_normalization = normalization
                    this_dropout = dropout
                    is_readout_layer = False
                else:
                    # Define the parameters for the last layer
                    this_activation = last_activation
                    this_normalization = last_normalization
                    this_dropout = last_dropout
                    is_readout_layer = last_layer_is_readout

                if constant_droppath_rate:
                    this_drop_rate = droppath_rate
                else:
                    this_drop_rate = DropPath.get_stochastic_drop_rate(droppath_rate, ii, depth)

                # Add a fully-connected layer
                fully_connected.append(
                    FCLayer(
                        all_dims[ii],
                        all_dims[ii + 1],
                        activation=this_activation,
                        normalization=this_normalization,
                        dropout=this_dropout,
                        is_readout_layer=is_readout_layer,
                        droppath_rate=this_drop_rate,
                    )
                )

        self.fully_connected = nn.Sequential(*fully_connected)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the MLP on the input features.

        Parameters:

            h: `torch.Tensor[..., Din]`:
                Input feature tensor, before the MLP.
                `Din` is the number of input features

        Returns:

            `torch.Tensor[..., Dout]`:
                Output feature tensor, after the MLP.
                `Dout` is the number of output features

        """
        if self.first_normalization is not None:
            h = self.first_normalization(h)
        if self.fully_connected is not None:
            h = self.fully_connected(h)
        return h

    @property
    def in_features(self):
        return self.in_dim

    def __getitem__(self, idx: int) -> nn.Module:
        return self.fully_connected[idx]

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        return self.__class__.__name__ + " (" + str(self.in_dim) + " -> " + str(self.out_dim) + ")"


class GRU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        r"""
        Wrapper class for the GRU used by the GNN framework, nn.GRU is used for the Gated Recurrent Unit itself

        Parameters:
            in_dim:
                Input dimension of the GRU layer
            hidden_dim:
                Hidden dimension of the GRU layer.
        """

        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(in_dim=in_dim, hidden_dim=hidden_dim)

    def forward(self, x, y):
        r"""
        Parameters:
            x:  `torch.Tensor[B, N, Din]`
                where Din <= in_dim (difference is padded)
            y:  `torch.Tensor[B, N, Dh]`
                where Dh <= hidden_dim (difference is padded)

        Returns:
            torch.Tensor: `torch.Tensor[B, N, Dh]`

        """
        assert x.shape[-1] <= self.in_dim and y.shape[-1] <= self.hidden_dim

        (B, N, _) = x.shape
        x = x.reshape(1, B * N, -1).contiguous()
        y = y.reshape(1, B * N, -1).contiguous()

        # padding if necessary
        if x.shape[-1] < self.in_dim:
            x = F.pad(input=x, pad=[0, self.in_dim - x.shape[-1]], mode="constant", value=0)
        if y.shape[-1] < self.hidden_dim:
            y = F.pad(
                input=y,
                pad=[0, self.hidden_dim - y.shape[-1]],
                mode="constant",
                value=0,
            )

        x = self.gru(x, y)[1]
        x = x.reshape(B, N, -1)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_rate: float):
        r"""
        DropPath class for stochastic depth
        Deep Networks with Stochastic Depth
        Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra and Kilian Weinberger
        https://arxiv.org/abs/1603.09382

        Parameters:
            drop_rate:
                Drop out probability
        """

        super().__init__()
        self.drop_rate = drop_rate

    @staticmethod
    def get_stochastic_drop_rate(
        drop_rate: float, layer_idx: Optional[int] = None, layer_depth: Optional[int] = None
    ):
        """
        Get the stochastic drop rate from the nominal drop rate, the layer index, and the layer depth.

        `return drop_rate * (layer_idx / (layer_depth - 1))`

        Parameters:
            drop_rate:
                Drop out nominal probability

            layer_idx:
                The index of the current layer

            layer_depth:
                The total depth (number of layers) associated to this specific layer

        """
        if drop_rate == 0:
            return 0
        else:
            assert (layer_idx is not None) and (
                layer_depth is not None
            ), f"layer_idx={layer_idx} and layer_depth={layer_depth} should be integers when `droppath_rate>0`"
            return drop_rate * (layer_idx / (layer_depth - 1))

    def forward(
        self,
        input: Tensor,
        batch_idx: IntTensor,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""
        Parameters:
            input:  `torch.Tensor[total_num_nodes, hidden]`
            batch: batch attribute of the batch object, batch.batch
            batch_size: The batch size. Must be provided when working on IPU

        Returns:
            torch.Tensor: `torch.Tensor[total_num_nodes, hidde]`

        """
        on_ipu = is_running_on_ipu()

        if self.drop_rate > 0:
            keep_prob = 1 - self.drop_rate

            # Parse the batch size
            if batch_size is None:
                if on_ipu:
                    raise ValueError(
                        "When using the IPU the batch size must be "
                        "provided during compilation instead of determined at runtime"
                    )
                else:
                    batch_size = int(batch_idx.max()) + 1

            # mask shape: [num_graphs, 1]
            mask = input.new_empty(batch_size, 1).bernoulli_(keep_prob)
            # if on_ipu, the last graph is a padded fake graph
            if on_ipu:
                mask[-1] = 0
            # using gather to extend mask to [total_num_nodes, 1]
            node_mask = mask[batch_idx]
            if keep_prob == 0:
                # avoid dividing by 0
                input_scaled = input
            else:
                input_scaled = input / keep_prob
            out = input_scaled * node_mask
        else:
            out = input
        return out
