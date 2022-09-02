from typing import List, Dict
import torch
import torch.nn as nn


from goli.nn.base_layers import MLP, get_norm

# ANDY: Here
class LapPENodeEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self,
                on_keys: Dict,
                in_dim: int, # Size of Laplace PE embedding
                hidden_dim: int,
                out_dim: int,
                model_type, # 'Transformer' or 'DeepSet'
                num_layers,
                num_layers_post=0, # Num. layers to apply after pooling
                dropout=0.,
                first_normalization=None,
                **model_kwargs):
        super().__init__()

        # Parse the `on_keys`.
        self.on_keys = self.parse_on_keys(on_keys)

        if model_type not in ['Transformer', 'DeepSet']:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type

        if out_dim - in_dim < 1:
            raise ValueError(f"LapPE size {in_dim} is too large for "
                             f"desired embedding size of {out_dim}.")

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = nn.Linear(2, in_dim)
        self.first_normalization = get_norm(first_normalization, dim=in_dim)


        #! Andy: check if these are desired architecture, a lot of new hyperparameters here for the encoder
        if model_type == 'Transformer':
            # Transformer model for LapPE
            encoder_layer = nn.TransformerEncoderLayer(
                    d_model=in_dim,
                    nhead=1,
                    batch_first=True,
                    dropout=dropout,
                    **model_kwargs)
            self.pe_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers)
        else:
            # DeepSet model for LapPE
            self.pe_encoder = MLP(
                    in_dim=2,
                    hidden_dim=hidden_dim,
                    out_dim=in_dim,
                    layers=num_layers,
                    dropout=dropout,
                    **model_kwargs)

        self.post_mlp = None
        if num_layers_post > 0:
            # MLP to apply post pooling
            self.post_mlp = MLP(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=in_dim,
                    layers=num_layers_post,
                    dropout=dropout,
                    **model_kwargs)

    def parse_on_keys(self, on_keys):
        if len(on_keys) != 2:
            raise ValueError(f"`{self.__class__}` only supports 2 keys")
        if ("eigvals" not in on_keys.keys()) and ("eigvecs" not in on_keys.keys()):
            raise ValueError(f"`on_keys` must contain the keys 'eigvals' and eigvecs. Provided {on_keys}")
        return on_keys

    def forward(self, eigvals, eigvecs):

        if self.training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)


        pos_enc = torch.cat((eigvecs.unsqueeze(2), eigvals.unsqueeze(2)), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        if self.first_normalization:
            pos_enc = self.first_normalization(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder: a Transformer or DeepSet model
        if self.model_type == 'Transformer':
            pos_enc = self.pe_encoder(src=pos_enc,
                                      src_key_padding_mask=empty_mask[:, :, 0])
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        output = {"node": pos_enc}

        return output

