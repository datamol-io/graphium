"""
SignNet https://arxiv.org/abs/2202.13013
based on https://github.com/cptq/SignNet-BasisNet
"""
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_scatter import scatter
from torch_geometric.data import Batch

from graphium.nn.base_layers import MLP
from graphium.nn.encoders.base_encoder import BaseEncoder


class SimpleGIN(nn.Module):
    def __init__(
        self, in_dim, hidden_dim, out_dim, num_layers, normalization="none", dropout=0.5, activation="relu"
    ):
        r"""
        not supported yet
        """
        # TODO: Not sure this works? Needs updating.
        super().__init__()
        self.layers = nn.ModuleList()
        self.normalization = normalization
        # input layer
        update_net = MLP(
            in_dim,
            hidden_dim,
            hidden_dim,
            2,
            normalization=normalization,
            dropout=dropout,
            activation=activation,
            last_normalization=normalization,
            last_dropout=dropout,
        )
        self.layers.append(GINConv(update_net))
        # hidden layers
        for i in range(num_layers - 2):
            update_net = MLP(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                2,
                normalization=normalization,
                dropout=dropout,
                activation=activation,
                last_normalization=normalization,
                last_dropout=dropout,
            )
            self.layers.append(GINConv(update_net))
        # output layer
        update_net = MLP(
            hidden_dim,
            hidden_dim,
            out_dim,
            2,
            normalization=normalization,
            dropout=dropout,
            activation=activation,
            last_normalization=normalization,
            last_dropout=dropout,
        )
        self.layers.append(GINConv(update_net))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for ii, layer in enumerate(self.layers):
            x = layer(x, edge_index)
        return x


class GINDeepSigns(nn.Module):
    """Sign invariant neural network with MLP aggregation.
    f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        k,
        dim_pe,
        rho_num_layers,
        normalization="none",
        dropout=0.5,
        activation="relu",
    ):
        # TODO: Not sure this works? Needs updating.
        super().__init__()
        self.enc = SimpleGIN(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            normalization=normalization,
            dropout=dropout,
            activation=activation,
        )
        rho_dim = out_channels * k
        self.rho = MLP(
            rho_dim,
            hidden_channels,
            dim_pe,
            rho_num_layers,
            normalization=normalization,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        x = x.transpose(0, 1)  # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)
        x = x.transpose(0, 1).reshape(N, -1)  # K x N x Out -> N x (K * Out)
        x = self.rho(x)  # N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


class MaskedGINDeepSigns(nn.Module):
    """Sign invariant neural network with sum pooling and DeepSet.
    f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dim_pe,
        rho_num_layers,
        normalization="none",
        dropout=0.5,
        activation="relu",
    ):
        # TODO: Not sure this works? Needs updating.
        super().__init__()
        self.enc = SimpleGIN(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            normalization=normalization,
            dropout=dropout,
            activation=activation,
        )
        self.rho = MLP(
            out_channels,
            hidden_channels,
            dim_pe,
            rho_num_layers,
            normalization=normalization,
            dropout=dropout,
            activation=activation,
        )

    def batched_n_nodes(self, batch_index):
        batch_size = batch_index.max().item() + 1
        one = batch_index.new_ones(batch_index.size(0))
        n_nodes = scatter(
            one, batch_index, dim=0, dim_size=batch_size, reduce="add"
        )  # Number of nodes in each graph.
        n_nodes = n_nodes.unsqueeze(1)
        return torch.cat([size * n_nodes.new_ones(size) for size in n_nodes])

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        K = x.shape[1]  # Max. number of eigen vectors / frequencies.
        x = x.transpose(0, 1)  # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)  # K x N x Out
        x = x.transpose(0, 1)  # K x N x Out -> N x K x Out

        batched_num_nodes = self.batched_n_nodes(batch_index)
        mask = torch.cat([torch.arange(K).unsqueeze(0) for _ in range(N)])
        mask = (mask.to(batch_index.device) < batched_num_nodes.unsqueeze(1)).bool()
        x[~mask] = 0
        x = x.sum(dim=1)  # (sum over K) -> N x Out
        x = self.rho(x)  # N x Out -> N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


class SignNetNodeEncoder(BaseEncoder):
    r"""SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet

    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \rhi(-v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.

    SignNetPE of size dim_pe will get appended to each node feature vector.

    Args:
        dim_emb: Size of final node embedding
    """

    def __init__(
        self,
        input_keys: Dict,
        output_keys: List[str],
        in_dim,  # Size of PE embedding
        hidden_dim,
        out_dim,
        num_layers,  # Num. layers in \phi GNN part
        model_type,  # 'MLP' or 'DeepSet'
        max_freqs,  # Num. eigenvectors (frequencies)
        use_input_keys_prefix: bool = True,
        num_layers_post=0,  # Num. layers in \rho MLP/DeepSet
        activation="relu",
        dropout=0.0,
        normalization="none",
    ):
        # TODO: Not sure this works? Needs updating.
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            in_dim=in_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            activation=activation,
            use_input_keys_prefix=use_input_keys_prefix,
        )

        # Parse the `input_keys`.
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.max_freqs = max_freqs
        self.num_layers_post = num_layers_post
        self.dropout = dropout
        self.normalization = normalization

        if model_type not in ["MLP", "DeepSet"]:
            raise ValueError(f"Unexpected SignNet model {model_type}")
        self.model_type = model_type

        if num_layers_post < 1:
            raise ValueError(f"Num layers in rho model has to be positive.")

        if out_dim - in_dim < 1:
            raise ValueError(
                f"SignNet PE size {in_dim} is too large for " f"desired embedding size of {out_dim}."
            )

        # Sign invariant neural network.
        if self.model_type == "MLP":
            self.sign_inv_net = GINDeepSigns(
                in_channels=1,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=num_layers,
                k=max_freqs,
                dim_pe=in_dim,
                rho_num_layers=num_layers_post,
                normalization=normalization,
                dropout=dropout,
                activation=activation,
            )
        elif self.model_type == "DeepSet":
            self.sign_inv_net = MaskedGINDeepSigns(
                in_channels=1,
                hidden_channels=hidden_dim,
                out_channels=out_dim,
                num_layers=num_layers,
                dim_pe=in_dim,
                rho_num_layers=num_layers_post,
                normalization=normalization,
                dropout=dropout,
                activation=activation,
            )
        else:
            raise ValueError(f"Unexpected model {self.model_type}")

    def parse_input_keys(self, input_keys):
        if len(input_keys) != 1:
            raise ValueError(f"`{self.__class__}` only supports 1 key")
        for key in input_keys:
            assert not key.startswith(
                "edge_"
            ), f"Input keys must be node features, not edge features, for encoder {self.__class__}"
            assert not key.startswith(
                "graph_"
            ), f"Input keys must be node features, not graph features, for encoder {self.__class__}"
        return input_keys

    def parse_output_keys(self, output_keys):
        for key in output_keys:
            assert not key.startswith(
                "edge_"
            ), f"Edge encodings are not supported for encoder {self.__class__}"
            assert not key.startswith(
                "graph_"
            ), f"Graph encodings are not supported for encoder {self.__class__}"
        return output_keys

    def forward(self, batch: Batch, key_prefix: Optional[str] = None) -> Dict[str, torch.Tensor]:
        input_keys = self.parse_input_keys_with_prefix(key_prefix)
        eigvecs, edge_index, batch_index = batch[input_keys[0]], batch["edge_index"], batch["batch_index"]
        pos_enc = eigvecs.unsqueeze(-1)  # (Num nodes) x (Num Eigenvectors) x 1

        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 1

        # SignNet
        pos_enc = self.sign_inv_net(pos_enc, edge_index, batch_index)  # (Num nodes) x (pos_enc_dim)

        output = {key: pos_enc for key in self.output_keys}

        return output

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        # TODO: Update this. It is broken

        Parameters:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension
        """
        base_kwargs = super().make_mup_base_kwargs(divide_factor=divide_factor, factor_in_dim=factor_in_dim)
        base_kwargs.update(
            dict(
                hidden_dim=round(self.hidden_dim / divide_factor),
                model_type=self.model_type,
                max_freqs=self.max_freqs,
                num_layers_post=self.num_layers_post,
                dropout=self.dropout,
                normalization=type(self.normalization).__name__,
            )
        )
        return base_kwargs
