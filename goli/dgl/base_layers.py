import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, Set2Set, GlobalAttentionPooling

from goli.commons.utils import ModuleListConcat


SUPPORTED_ACTIVATION_MAP = {"ReLU", "Sigmoid", "Tanh", "ELU", "SELU", "GLU", "LeakyReLU", "Softplus", "None"}
EPS = 1e-5


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), "Unhandled activation function"
    activation = activation[0]
    if activation.lower() == "none":
        return None
    return vars(torch.nn.modules.activation)[activation]()


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:
    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)
    Arguments
    ----------
        in_dim: int
            Input dimension of the layer (the torch.nn.Linear)
        out_dim: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        batch_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_dim}}`
            (Default value = None)
    Attributes
    ----------
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
        in_dim,
        out_dim,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        bias=True,
        init_fn=None,
        device="cpu",
    ):
        super(FCLayer, self).__init__()

        self.__params = locals()
        del self.__params["__class__"]
        del self.__params["self"]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.linear = nn.Linear(in_dim, out_dim, bias=bias).to(device)
        self.dropout = None
        self.batch_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_dim).to(device)
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
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.batch_norm is not None:
            if h.shape[1] != self.out_dim:
                h = self.batch_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.batch_norm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_dim) + " -> " + str(self.out_dim) + ")"


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
        device="cpu",
    ):
        super(MLP, self).__init__()

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
                    device=device,
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
                    device=device,
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
                        device=device,
                        dropout=dropout,
                    )
                )
            self.fully_connected.append(
                FCLayer(
                    hidden_dim,
                    out_dim,
                    activation=last_activation,
                    batch_norm=last_batch_norm,
                    device=device,
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
        super(GRU, self).__init__()
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


class S2SReadout(nn.Module):
    """
    Performs a Set2Set aggregation of all the graph nodes' features followed by a series of fully connected layers
    """

    def __init__(self, in_dim, hidden_dim, out_dim, fc_layers=3, device="cpu", final_activation="relu"):
        super(S2SReadout, self).__init__()

        # set2set aggregation
        self.set2set = Set2Set(in_dim, device=device)

        # fully connected layers
        self.mlp = MLP(
            in_dim=2 * in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            layers=fc_layers,
            mid_activation="relu",
            last_activation=final_activation,
            mid_batch_norm=True,
            last_batch_norm=False,
            device=device,
        )

    def forward(self, x):
        x = self.set2set(x)
        return self.mlp(x)


class StdPooling(nn.Module):
    r"""Apply standard deviation pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \sigma_{k=1}^{N_i}\left( x^{(i)}_k \right)
    """

    def __init__(self):
        super(StdPooling, self).__init__()
        self.sum_pooler = SumPooling()
        self.relu = nn.ReLU()

    def forward(self, graph, feat):
        r"""Compute standard deviation pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """

        readout = torch.sqrt(
            self.relu((self.sum_pooler(graph, feat ** 2)) - (self.sum_pooler(graph, feat) ** 2)) + EPS
        )
        return readout


class MinPooling(MaxPooling):
    r"""Apply min pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \min_{k=1}^{N_i}\left( x^{(i)}_k \right)
    """

    def forward(self, graph, feat):
        r"""Compute max pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """

        readout = -super().forward(graph, -feat)
        return readout


def parse_pooling_layer(in_dim: int, pooling: List[str], n_iters: int = 2, n_layers: int = 2):
    r"""
    Select the pooling layers from a list of strings, and put them
    in a Module that concatenates their outputs.

    Parameters
    ------------

    in_dim: int
        The dimension at the input layer of the pooling

    pooling: list(str)
        The list of pooling layers to use. The accepted strings are:

        - "sum": SumPooling

        - "mean": MeanPooling

        - "max": MaxPooling

        - "min": MinPooling

        - "std": StdPooling

        - "s2s": Set2Set

    n_iters: int, Default=2
        IGNORED FOR ALL POOLING LAYERS, EXCEPT "s2s".
        The number of iterations.

    n_layers : int, Default=2
        IGNORED FOR ALL POOLING LAYERS, EXCEPT "s2s".
        The number of recurrent layers.
    """

    # TODO: Add configuration for the pooling layer kwargs

    # Create the pooling layer
    pooling = pooling.lower()
    pool_layer = ModuleListConcat()
    out_pool_dim = 0

    for this_pool in re.split("\s+|_", pooling):
        out_pool_dim += in_dim
        if this_pool == "sum":
            pool_layer.append(SumPooling())
        elif this_pool == "mean":
            pool_layer.append(AvgPooling())
        elif this_pool == "max":
            pool_layer.append(MaxPooling())
        elif this_pool == "min":
            pool_layer.append(MinPooling())
        elif this_pool == "std":
            pool_layer.append(StdPooling())
            pool_layer.append(Set2Set(input_dim=in_dim, n_iters=n_iters, n_layers=n_layers))
            out_pool_dim += in_dim
        elif (this_pool == "none") or (this_pool is None):
            pass
        else:
            raise NotImplementedError(f"Undefined pooling `{this_pool}`")

    return pool_layer, out_pool_dim


class VirtualNode(nn.Module):
    def __init__(self, dim, dropout, batch_norm=False, bias=True, residual=True, vn_type="sum"):
        super().__init__()
        if (vn_type is None) or (vn_type.lower() == "none"):
            self.vn_type = None
            self.fc_layer = None
            self.residual = None
            return

        self.vn_type = vn_type.lower()
        self.fc_layer = FCLayer(
            in_size=dim,
            out_size=dim,
            activation="relu",
            dropout=dropout,
            b_norm=batch_norm,
            bias=bias,
        )
        self.residual = residual

    def forward(self, g, h, vn_h):

        g.ndata["h"] = h

        # Pool the features
        if self.vn_type is None:
            return vn_h, h
        elif self.vn_type == "mean":
            pool = mean_nodes(g, "h")
        elif self.vn_type == "sum":
            pool = sum_nodes(g, "h")
        elif self.vn_type == "logsum":
            pool = mean_nodes(g, "h")
            lognum = torch.log(torch.tensor(g.batch_num_nodes, dtype=h.dtype, device=h.device))
            pool = pool * lognum.unsqueeze(-1)
        else:
            raise ValueError(
                f'Undefined input "{self.pooling}". Accepted values are "none", "sum", "mean", "logsum"'
            )

        # Compute the new virtual node features
        vn_h_temp = self.fc_layer.forward(vn_h + pool)
        if self.residual:
            vn_h = vn_h + vn_h_temp
        else:
            vn_h = vn_h_temp

        # Add the virtual node value to the graph features
        temp_h = torch.cat(
            [vn_h[ii : ii + 1].repeat(num_nodes, 1) for ii, num_nodes in enumerate(g.batch_num_nodes)],
            dim=0,
        )
        h = h + temp_h

        return vn_h, h
