from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from goli.nn.base_layers import MLP, get_norm, FCLayer


class LapPENodeEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(
        self,
        on_keys: List[str],
        in_dim: int,  # Size of Laplace PE embedding
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        model_type: str = "DeepSet",  # 'Transformer' or 'DeepSet'
        num_layers_post=0,  # Num. layers to apply after pooling
        dropout=0.0,
        first_normalization=None,
        use_prefix: bool = True,
        **model_kwargs,
    ):
        super().__init__()

        # Parse the `on_keys`.
        self.on_keys = self.parse_on_keys(on_keys)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.model_type = model_type
        self.num_layers_post = num_layers_post
        self.dropout = dropout
        self.first_normalization = first_normalization
        self.model_kwargs = model_kwargs

        if model_type not in ["Transformer", "DeepSet"]:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type

        if out_dim - in_dim < 1:
            raise ValueError(f"LapPE size {in_dim} is too large for " f"desired embedding size of {out_dim}.")

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = FCLayer(2, in_dim, activation="none")
        self.first_normalization = get_norm(first_normalization, dim=in_dim)

        if model_type == "Transformer":
            # Transformer model for LapPE
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_dim, nhead=1, batch_first=True, dropout=dropout, **model_kwargs
            )
            self.pe_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # DeepSet model for LapPE
            self.pe_encoder = MLP(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                layers=num_layers,
                dropout=dropout,
                **model_kwargs,
            )

        self.post_mlp = None
        if num_layers_post > 0:
            # MLP to apply post pooling
            self.post_mlp = MLP(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=in_dim,
                layers=num_layers_post,
                dropout=dropout,
                **model_kwargs,
            )

    def parse_on_keys(self, on_keys):
        if len(on_keys) != 2:
            raise ValueError(f"`{self.__class__}` only supports 2 keys")

        return on_keys

    def forward(self, batch: Batch, key_prefix: Optional[str] = None) -> Dict[str, torch.Tensor]:

        on_keys = self.on_keys
        if (key_prefix is not None) and (self.use_prefix):
            on_keys = [f"{key_prefix}/{k}" for k in on_keys]
        eigvals, eigvecs = batch[on_keys[0]], batch[on_keys[1]]

        # Random flipping to the Laplacian encoder
        if self.training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
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
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder: a Transformer or DeepSet model
        if self.model_type == "Transformer":
            pos_enc = self.pe_encoder(src=pos_enc, src_key_padding_mask=empty_mask[:, :, 0])
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2), 0.0)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        output = {"node": pos_enc}

        return output

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension
        """
        return dict(
            on_keys=self.on_keys,
            in_dim=round(self.in_dim / divide_factor) if factor_in_dim else self.in_dim,
            hidden_dim=round(self.hidden_dim / divide_factor),
            out_dim=round(self.out_dim / divide_factor),
            num_layers=self.num_layers,
            model_type=self.model_type,
            num_layers_post=self.num_layers_post,
            dropout=self.dropout,
            first_normalization=self.first_normalization,
            **self.model_kwargs,
        )
