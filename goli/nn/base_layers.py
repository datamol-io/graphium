from typing import Union, Callable, Optional, Type, Tuple
from copy import deepcopy
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import mup.init as mupi
from mup import set_base_shapes, MuReadout, get_shapes

from goli.ipu.ipu_utils import import_poptorch

SUPPORTED_ACTIVATION_MAP = {"ReLU", "Sigmoid", "Tanh", "ELU", "SELU", "GLU", "LeakyReLU", "Softplus", "None"}


def get_activation(activation: Union[type(None), str, Callable]) -> Optional[Callable]:
    r"""
    returns the activation function represented by the input string

    Parameters:
        activation: Callable, `None`, or string with value:
            "none", "ReLU", "Sigmoid", "Tanh", "ELU", "SELU", "GLU", "LeakyReLU", "Softplus"

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
    assert len(activation) == 1 and isinstance(activation[0], str), "Unhandled activation function"
    activation = activation[0]

    return vars(torch.nn.modules.activation)[activation]()


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
    if normalization is None or normalization == "none":
        pass
    elif callable(normalization):
        parsed_norm = normalization
    elif normalization == "batch_norm":
        parsed_norm = nn.BatchNorm1d(dim)
    elif normalization == "layer_norm":
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
    """

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

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:

        # Patching the forward to use a different scaling for the dot-product
        prev_fn = F._scaled_dot_product_attention
        F._scaled_dot_product_attention = _mup_scaled_dot_product_attention
        out = super().forward(*args, **kwargs)
        F._scaled_dot_product_attention = prev_fn
        return out


def _mup_scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    This modifies the standard torch function by normalizing with `1/d` instead of `1/sqrt(d)`

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / E  # Instead of `q / math.sqrt(E)`
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


class MuReadoutGoli(MuReadout):
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
        base_in_dim: Optional[int] = None,
        base_out_dim: Optional[int] = None,
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
        self.base_in_dim = base_in_dim
        self.base_out_dim = base_out_dim

        # Dropout and activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        self.activation = get_activation(activation)

        # Linear layer, or MuReadout layer
        if not is_readout_layer:
            self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            self.linear = MuReadoutGoli(in_dim, out_dim, bias=bias)

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
        base_shapes = get_shapes(self)
        base_shapes["linear.weight"] = list(base_shapes["linear.weight"])
        if self.base_in_dim is not None:
            base_shapes["linear.weight"][0] = self.base_in_dim
        if self.base_out_dim is not None:
            base_shapes["linear.weight"][1] = self.base_out_dim
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
            h = torch.zeros(list(h.shape[:-1]) + [self.linear.out_features], device=h.device, dtype=h.dtype)
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
        hidden_dim: int,
        out_dim: int,
        layers: int,
        activation: Union[str, Callable] = "relu",
        last_activation: Union[str, Callable] = "none",
        dropout: float = 0.0,
        last_dropout: float = 0.0,
        normalization: Union[Type[None], str, Callable] = "none",
        last_normalization: Union[Type[None], str, Callable] = "none",
        first_normalization: Union[Type[None], str, Callable] = "none",
        last_layer_is_readout: bool = False,
    ):
        r"""
        Simple multi-layer perceptron, built of a series of FCLayers

        Parameters:
            in_dim:
                Input dimension of the MLP
            hidden_dim:
                Hidden dimension of the MLP. All hidden dimensions will have
                the same number of parameters
            out_dim:
                Output dimension of the MLP.
            layers:
                Number of hidden layers
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

        """

        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.first_normalization = get_norm(first_normalization, dim=in_dim)

        fully_connected = []
        if layers == 0:
            self.fully_connected = None
            return
        elif layers == 1:
            fully_connected.append(
                FCLayer(
                    in_dim,
                    out_dim,
                    activation=last_activation,
                    normalization=last_normalization,
                    dropout=last_dropout,
                    is_readout_layer=last_layer_is_readout,
                )
            )
        elif layers > 1:
            fully_connected.append(
                FCLayer(
                    in_dim,
                    hidden_dim,
                    activation=activation,
                    normalization=normalization,
                    dropout=dropout,
                )
            )
            for _ in range(layers - 2):
                fully_connected.append(
                    FCLayer(
                        hidden_dim,
                        hidden_dim,
                        activation=activation,
                        normalization=normalization,
                        dropout=dropout,
                    )
                )
            fully_connected.append(
                FCLayer(
                    hidden_dim,
                    out_dim,
                    activation=last_activation,
                    normalization=last_normalization,
                    dropout=last_dropout,
                    is_readout_layer=last_layer_is_readout,
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
            y = F.pad(input=y, pad=[0, self.hidden_dim - y.shape[-1]], mode="constant", value=0)

        x = self.gru(x, y)[1]
        x = x.reshape(B, N, -1)
        return x
