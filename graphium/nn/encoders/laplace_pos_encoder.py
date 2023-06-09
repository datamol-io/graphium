from typing import List, Dict, Any, Optional, Union, Callable
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from graphium.nn.base_layers import MLP, get_norm, FCLayer
from graphium.nn.encoders.base_encoder import BaseEncoder


class LapPENodeEncoder(BaseEncoder):
    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        in_dim: int,  # Size of Laplace PE embedding. Only used by the MLP model
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: Optional[Union[str, Callable]] = "relu",
        model_type: str = "DeepSet",  # 'Transformer' or 'DeepSet' or 'MLP'
        num_layers_post=1,  # Num. layers to apply after pooling
        dropout=0.0,
        first_normalization=None,
        use_input_keys_prefix: bool = True,
        **model_kwargs,
    ):
        r"""
        Laplace Positional Embedding node encoder.
        LapPE of size dim_pe will get appended to each node feature vector.

        Parameters:
            input_keys: List of input keys to use from the data object.
            output_keys: List of output keys to add to the data object.
            in_dim : Size of Laplace PE embedding. Only used by the MLP model
            hidden_dim: Size of hidden layer
            out_dim: Size of final node embedding
            num_layers: Number of layers in the MLP
            activation: Activation function to use.
            model_type: 'Transformer' or 'DeepSet' or 'MLP'
            num_layers_post: Number of layers to apply after pooling
            dropout: Dropout rate
            first_normalization: Normalization to apply to the first layer.
        """
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            in_dim=in_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            activation=activation,
            first_normalization=first_normalization,
            use_input_keys_prefix=use_input_keys_prefix,
        )

        # Parse the `input_keys`.
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        if num_layers_post == 0:
            assert hidden_dim == out_dim, "Hidden dim must be equal to out dim if num_layers_post == 0"
        self.num_layers_post = num_layers_post
        self.dropout = dropout
        self.model_kwargs = model_kwargs

        if out_dim - in_dim < 1:
            raise ValueError(f"LapPE size {in_dim} is too large for " f"desired embedding size of {out_dim}.")

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_in = FCLayer(2, hidden_dim, activation="none")

        if self.model_type == "Transformer":
            # Transformer model for LapPE
            model_kwargs.setdefault("nhead", 1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                batch_first=True,
                dropout=dropout,
                activation=self.activation,
                **model_kwargs,
            )
            self.pe_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif self.model_type == "DeepSet":
            # DeepSet model for LapPE (this will be followed by a sum pooling)
            self.pe_encoder = MLP(
                in_dim=hidden_dim,
                hidden_dims=hidden_dim,
                out_dim=hidden_dim,
                depth=num_layers,
                dropout=dropout,
                **model_kwargs,
            )
        elif self.model_type == "MLP":
            # MLP that will mix all eigenvalues and eigenvectors
            self.pe_encoder = MLP(
                in_dim=self.in_dim * hidden_dim,
                hidden_dims=hidden_dim,
                out_dim=hidden_dim,
                depth=num_layers_post,
                dropout=dropout,
                activation=activation,
                last_activation="none",
                **model_kwargs,
            )
        else:
            raise ValueError(f"Unexpected PE model {self.model_type}")

        self.post_mlp = None
        if num_layers_post > 0:
            # MLP to apply post pooling
            self.post_mlp = MLP(
                in_dim=hidden_dim,
                hidden_dims=hidden_dim,
                out_dim=out_dim,
                depth=num_layers_post,
                dropout=dropout,
                activation=activation,
                last_activation="none",
            )

    def parse_input_keys(
        self,
        input_keys: List[str],
    ) -> List[str]:
        r"""
        Parse the input keys and make sure they are supported for this encoder
        Parameters:
            input_keys: List of input keys to use from the data object.
        Returns:
            List of parsed input keys
        """
        if len(input_keys) != 2:
            raise ValueError(f"`{self.__class__}` only supports 2 keys")
        for key in input_keys:
            assert not key.startswith(
                "edge_"
            ), f"Input keys must be node features, not edge features, for encoder {self.__class__}"
            assert not key.startswith(
                "nodepair_"
            ), f"Input keys must be node features, not graph features, for encoder {self.__class__}"
            assert not key.startswith(
                "graph_"
            ), f"Input keys must be node features, not graph features, for encoder {self.__class__}"
        return input_keys

    def parse_output_keys(
        self,
        output_keys: List[str],
    ) -> List[str]:
        r"""
        parse the output keys
        Parameters:
            output_keys: List of output keys to add to the data object.
        Returns:
            List of parsed output keys
        """
        for key in output_keys:
            assert not key.startswith(
                "edge_"
            ), f"Edge encodings are not supported for encoder {self.__class__}"
            assert not key.startswith(
                "nodepair_"
            ), f"Edge encodings are not supported for encoder {self.__class__}"
            assert not key.startswith(
                "graph_"
            ), f"Graph encodings are not supported for encoder {self.__class__}"
        return output_keys

    def forward(
        self,
        batch: Batch,
        key_prefix: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Forward pass of the encoder.
        Parameters:
            batch: pyg Batches of graphs
            key_prefix: Prefix to use for the input and output keys.
        Returns:
            output dictionary with keys as specified in `output_keys` and their output embeddings.
        """
        # input_keys = self.parse_input_keys_with_prefix(key_prefix)
        eigvals, eigvecs = batch[self.input_keys[0]], batch[self.input_keys[1]]

        # Random flipping to the Laplacian encoder
        if self.training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device, dtype=eigvecs.dtype)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat(
            (eigvecs.unsqueeze(2), eigvals.unsqueeze(2)), dim=2
        )  # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        if self.first_normalization:
            pos_enc = self.first_normalization(pos_enc)
        pos_enc = self.linear_in(pos_enc)  # (Num nodes) x (Num Eigenvectors) x hidden_dim

        # PE encoder: a Transformer or DeepSet model
        if self.model_type == "Transformer":
            pos_enc = self.pe_encoder(src=pos_enc, src_key_padding_mask=empty_mask[:, :, 0])
            # (Num nodes) x (Num Eigenvectors) x hidden_dim
        elif self.model_type == "DeepSet":
            pos_enc = self.pe_encoder(pos_enc)
            # (Num nodes) x (Num Eigenvectors) x hidden_dim
        elif self.model_type == "MLP":
            pos_enc = torch.flatten(pos_enc, start_dim=-2, end_dim=-1)
            pos_enc = self.pe_encoder(pos_enc)
            # (Num nodes) x hidden_dim
        else:
            raise ValueError(f"Unexpected PE model {self.model_type}")

        if self.model_type in ["Transformer", "DeepSet"]:
            # Mask out padded nodes
            pos_enc[empty_mask[..., 0]] = 0  # (Num nodes) x (Num Eigenvectors) x hidden_dim
            pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x hidden_dim

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x out_dim

        output = {key: pos_enc for key in self.output_keys}

        return output

    def make_mup_base_kwargs(
        self,
        divide_factor: float = 2.0,
        factor_in_dim: bool = False,
    ) -> Dict[str, Any]:
        r"""
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension
        Returns:
            Dictionary of kwargs to be used to create the base model.
        """
        base_kwargs = super().make_mup_base_kwargs(divide_factor, factor_in_dim)
        base_kwargs.update(
            dict(
                hidden_dim=round(self.hidden_dim / divide_factor),
                model_type=self.model_type,
                num_layers_post=self.num_layers_post,
                dropout=self.dropout,
                **self.model_kwargs,
            )
        )
        return base_kwargs
