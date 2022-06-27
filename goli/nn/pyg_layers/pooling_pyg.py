import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from typing import List, Union, Callable, Tuple, Optional

from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_scatter import scatter

from goli.nn.base_layers import MLP, FCLayer
from goli.utils.tensor import ModuleListConcat


EPS = 1e-6

def global_min_pool(x: Tensor, batch: LongTensor, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    minimum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

    Parameters:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='min')


def global_logsum_pool(x: Tensor, batch: LongTensor, size: Optional[int] = None):
    r"""
    Apply pooling over the nodes in the graph using a mean aggregation,
    but scaled by the log of the number of nodes. This gives the same
    expressive power as the sum, but helps deal with graphs that are
    significantly larger than others by using a logarithmic scale.

    $$r^{(i)} = \frac{\log N_i}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k$$

    Parameters:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    size = int(batch.max().item() + 1) if size is None else size
    mean_pool = scatter(x, batch, dim=0, dim_size=size, reduce='mean')
    _, num_nodes = torch.unique(batch, return_counts=True)
    lognum = torch.log(num_nodes)
    return mean_pool * lognum.unsqueeze(-1)


def global_std_pool(x: Tensor, batch: LongTensor, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    minimum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

    Parameters:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    size = int(batch.max().item() + 1) if size is None else size
    mean = scatter(x, batch, dim=0, out=None, dim_size=size, reduce='mean')
    mean_squares = scatter(x * x, batch, dim=0, out=None, dim_size=size, reduce='mean')
    out = mean_squares - mean * mean
    return torch.sqrt(torch.relu(out) + 1e-5)


def parse_pooling_layer_pyg(in_dim: int, pooling: Union[str, List[str]]):
    r"""
    Select the pooling layers from a list of strings, and put them
    in a Module that concatenates their outputs.

    Parameters:

        in_dim:
            The dimension at the input layer of the pooling

        pooling:
            The list of pooling layers to use. The accepted strings are:

            - "sum": `SumPooling`
            - "mean": `MeanPooling`
            - "max": `MaxPooling`
            - "min": `MinPooling`
            - "std": `StdPooling`

    """

    # TODO: Add configuration for the pooling layer kwargs

    # Create the pooling layer
    pool_layer = ModuleListConcat()
    out_pool_dim = 0
    if isinstance(pooling, str):
        pooling = [pooling]

    for this_pool in pooling:
        this_pool = None if this_pool is None else this_pool.lower()
        out_pool_dim += in_dim
        if this_pool == "sum":
            pool_layer.append(global_add_pool)
        elif this_pool == "mean":
            pool_layer.append(global_mean_pool)
        elif this_pool == "logsum":
            pool_layer.append(global_logsum_pool)
        elif this_pool == "max":
            pool_layer.append(global_max_pool)
        elif this_pool == "min":
            pool_layer.append(global_min_pool)
        elif this_pool == "std":
            pool_layer.append(global_std_pool)
        elif (this_pool == "none") or (this_pool is None):
            pass
        else:
            raise NotImplementedError(f"Undefined pooling `{this_pool}`")

    return pool_layer, out_pool_dim


class VirtualNodePyg(nn.Module):
    def __init__(
        self,
        dim: int,
        vn_type: Union[type(None), str] = "sum",
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        bias: bool = True,
        residual: bool = True,
    ):
        r"""
        The VirtualNode is a layer that pool the features of the graph,
        applies a neural network layer on the pooled features,
        then add the result back to the node features of every node.

        Parameters:

            in_dim:
                Input feature dimensions of the virtual node layer

            activation:
                activation function to use in the neural network layer.

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            bias:
                Whether to add a bias to the neural network

            residual:
                Whether all virtual nodes should be connected together
                via a residual connection

        """
        super().__init__()
        if (vn_type is None) or (vn_type.lower() == "none"):
            self.vn_type = None
            self.fc_layer = None
            self.residual = None
            return

        self.vn_type = vn_type.lower()
        self.residual = residual
        self.fc_layer = FCLayer(
            in_dim=dim,
            out_dim=dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            bias=bias,
        )

    def forward(
        self, x: Tensor, vn_x: Tensor, batch: LongTensor
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Apply the virtual node layer.

        Parameters:

            x (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the input features

            vn_x (torch.Tensor[..., M, Din]):
                Graph feature of the previous virtual node, or `None`
                `M` is the number of graphs, `Din` is the input features.
                It is added to the result after the MLP, as a residual connection

            batch

        Returns:

            `x = torch.Tensor[..., N, Dout]`:
                Node feature tensor, after convolution and residual.
                `N` is the number of nodes, `Dout` is the output features of the layer and residual

            `vn_x = torch.Tensor[..., M, Dout]`:
                Graph feature tensor to be used at the next virtual node, or `None`
                `M` is the number of graphs, `Dout` is the output features

        """

        # Pool the features
        if self.vn_type is None:
            return x, vn_x
        elif self.vn_type == "mean":
            pool = global_mean_pool(x, batch)
        elif self.vn_type == "max":
            pool = global_max_pool(x, batch)
        elif self.vn_type == "sum":
            pool = global_add_pool(x, batch)
        elif self.vn_type == "logsum":
            pool = global_logsum_pool(x, batch)
        else:
            raise ValueError(
                f'Undefined input "{self.pooling}". Accepted values are "none", "sum", "mean", "logsum"'
            )

        # Compute the new virtual node features
        vn_x_temp = self.fc_layer.forward(vn_x + pool)
        if self.residual:
            vn_x = vn_x + vn_x_temp
        else:
            vn_x = vn_x_temp

        # Add the virtual node value to the graph features
        x = x + vn_x[batch]

        return x, vn_x
