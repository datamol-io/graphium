import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_ACTIVATION_MAP = {"ReLU", "Sigmoid", "Tanh", "ELU", "SELU", "GLU", "LeakyReLU", "Softplus", "None"}
EPS = 1e-5


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation

    if (activation is None) or (activation.lower() == "none"):
        return None

    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), "Unhandled activation function"
    activation = activation[0]

    return vars(torch.nn.modules.activation)[activation]()


class FCLayer(nn.Module):
    """
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:
    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Parameters:
        in_dim:
            type: int
            Input dimension of the layer (the torch.nn.Linear)
        out_dim: int
            Output dimension of the layer.
        dropout:
            type: float
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable
            Activation function to use.
            (Default value = relu)
        batch_norm: bool
            Whether to use batch normalization
            (Default value = False)
        bias: bool
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_dim}}`
            (Default value = None)

    Attributes:
        dropout: int
            The ratio of units to dropout.
        batch_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_dim: int
            Input dimension of the linear layer
        out_dim: int
            Output dimension of the linear layer
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        bias=True,
        init_fn=None,
    ):
        super().__init__()

        self.__params = locals()
        del self.__params["__class__"]
        del self.__params["self"]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.dropout = None
        self.batch_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_dim)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_dim)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)

        if self.batch_norm is not None:
            if h.shape[1] != self.out_dim:
                h = self.batch_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.batch_norm(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)

        return h

    def __repr__(self):
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, activation={self.activation})"


class MLP(nn.Module):
    """
    Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        layers,
        mid_activation="relu",
        last_activation="none",
        dropout=0.0,
        mid_batch_norm=False,
        last_batch_norm=False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(
                FCLayer(
                    in_dim,
                    out_dim,
                    activation=last_activation,
                    batch_norm=last_batch_norm,
                    dropout=dropout,
                )
            )
        else:
            self.fully_connected.append(
                FCLayer(
                    in_dim,
                    hidden_dim,
                    activation=mid_activation,
                    batch_norm=mid_batch_norm,
                    dropout=dropout,
                )
            )
            for _ in range(layers - 2):
                self.fully_connected.append(
                    FCLayer(
                        hidden_dim,
                        hidden_dim,
                        activation=mid_activation,
                        batch_norm=mid_batch_norm,
                        dropout=dropout,
                    )
                )
            self.fully_connected.append(
                FCLayer(
                    hidden_dim,
                    out_dim,
                    activation=last_activation,
                    batch_norm=last_batch_norm,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_dim) + " -> " + str(self.out_dim) + ")"


class GRU(nn.Module):
    """
    Wrapper class for the GRU used by the GNN framework, nn.GRU is used for the Gated Recurrent Unit itself
    """

    def __init__(self, input_size, hidden_dim, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=input_size, hidden_dim=hidden_dim).to(device)

    def forward(self, x, y):
        """
        :param x:   shape: (B, N, Din) where Din <= input_size (difference is padded)
        :param y:   shape: (B, N, Dh) where Dh <= hidden_dim (difference is padded)
        :return:    shape: (B, N, Dh)
        """
        assert x.shape[-1] <= self.input_size and y.shape[-1] <= self.hidden_dim

        (B, N, _) = x.shape
        x = x.reshape(1, B * N, -1).contiguous()
        y = y.reshape(1, B * N, -1).contiguous()

        # padding if necessary
        if x.shape[-1] < self.input_size:
            x = F.pad(input=x, pad=[0, self.input_size - x.shape[-1]], mode="constant", value=0)
        if y.shape[-1] < self.hidden_dim:
            y = F.pad(input=y, pad=[0, self.hidden_dim - y.shape[-1]], mode="constant", value=0)

        x = self.gru(x, y)[1]
        x = x.reshape(B, N, -1)
        return x
