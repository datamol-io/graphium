from typing import Tuple, Optional, Dict
import numpy as np
from scipy.sparse import spmatrix
import torch

from goli.features.spectral import compute_laplacian_positional_eigvecs


def get_all_positional_encoding(
    adj: Tuple[np.ndarray, spmatrix],
    pos_encoding_as_features: Optional[Dict] = None,
    pos_encoding_as_directions: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    
    """

    pos_enc_feats, pos_enc_dir = None, None
    pos_encoding_as_features = {} if pos_encoding_as_features is None else pos_encoding_as_features
    pos_encoding_as_directions = {} if pos_encoding_as_directions is None else pos_encoding_as_directions

    # Get the positional encoding for the features
    if len(pos_encoding_as_features) > 0:
        pos_enc_feats = graph_positional_encoder(adj, **pos_encoding_as_features)

    # Get the positional encoding for the directions
    if len(pos_encoding_as_directions) > 0:
        if pos_encoding_as_directions == pos_encoding_as_features:
            pos_enc_dir = pos_enc_feats
        else:
            pos_enc_dir = graph_positional_encoder(adj, **pos_encoding_as_directions)

    return pos_enc_feats, pos_enc_dir


def graph_positional_encoder(
    adj: Tuple[np.ndarray, spmatrix], pos_type: str, num_pos: int, disconnect: bool = True, **kwargs
) -> np.ndarray:
    r"""
    Get a positional encoding that depends on the parameters.

    Parameters:

        adj: Adjacency matrix of the graph

        pos_type: The type of positional encoding to use. Supported types are:
            
            - laplacian_eigvec: the
            - laplacian_eigvec_eigval

    """

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
