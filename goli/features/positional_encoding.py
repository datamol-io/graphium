import numpy as np
import torch

from goli.features.spectral import compute_laplacian_positional_eigvecs


def graph_positional_encoder(adj, pos_type: str, num_pos: int, disconnect: bool = True, **kwargs):

    pos_type = pos_type.lower()

    if pos_type == "laplacian_eigvec":
        _, eigvecs = compute_laplacian_positional_eigvecs(
            adj=adj, num_pos=num_pos, disconnect=disconnect, **kwargs
        )
        pos_enc = eigvecs
    
    elif pos_type == "laplacian_eigvec_eigval":
        eigvals_tile, eigvecs = compute_laplacian_positional_eigvecs(
            adj=adj, num_pos=num_pos, disconnect=disconnect, **kwargs
        )
        pos_enc = np.concatenate((eigvecs, eigvals_tile), axis=1)
    else:
        raise ValueError(f"Unknown `pos_type`: {pos_type}")

    
    pos_enc = torch.as_tensor(pos_enc)

    return pos_enc

