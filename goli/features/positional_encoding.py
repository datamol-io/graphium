from typing import Tuple, Optional, Dict, Union
import numpy as np
from scipy.sparse import spmatrix
import torch

from goli.features.spectral import compute_laplacian_positional_eigvecs
from goli.features.rw import compute_rwse
def get_all_positional_encoding(
    adj: Union[np.ndarray, spmatrix],
    num_nodes: int,
    pos_encoding_as_features: Optional[Dict] = None,
    pos_encoding_as_directions: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Get features positional encoding and direction positional encoding.

    Parameters:
        adj: Adjacency matrix of the graph
        pos_encoding_as_features: keyword arguments for function `graph_positional_encoder`
            to generate positional encoding for node features.
        pos_encoding_as_directions: keyword arguments for function `graph_positional_encoder`
            to generate positional encoding for directional features.
    """

    #pos_enc_feats_sign_flip, pos_enc_feats_no_flip, pos_enc_dir = None, None, None
    pos_enc_dir = None
    pos_encoding_as_features = {} if pos_encoding_as_features is None else pos_encoding_as_features
    pos_encoding_as_directions = {} if pos_encoding_as_directions is None else pos_encoding_as_directions

    pe_dict = {}

    # Get the positional encoding for the features
    if len(pos_encoding_as_features) > 0:
        pe_dict = graph_positional_encoder(
            adj, num_nodes, pos_encoding_as_features
        )

    # Get the positional encoding for the directions
    # seem to be quite hard coded, I just replaced the entries with dictionary entries
    if len(pos_encoding_as_directions) > 0:
        if pos_encoding_as_directions == pos_encoding_as_features:

            # Concatenate the sign-flip and non-sign-flip positional encodings
            if pe_dict["pos_enc_feats_sign_flip"] is None:
                pos_enc_dir = pe_dict["pos_enc_feats_no_flip"]
            elif pe_dict["pos_enc_feats_no_flip"] is None:
                pos_enc_dir = pe_dict["pos_enc_feats_sign_flip"]
            else:
                pos_enc_dir = np.concatenate((pe_dict["pos_enc_feats_no_flip"], pe_dict["pos_enc_feats_sign_flip"]), axis=1)

        else:
            pe_dict = graph_positional_encoder(adj, **pos_encoding_as_directions)
            pos_enc_dir1 = pe_dict["pos_enc_feats_sign_flip"]
            pos_enc_dir2 = pe_dict["pos_enc_feats_no_flip"]
            # Concatenate both positional encodings
            if pos_enc_dir1 is None:
                pos_enc_dir = pos_enc_dir2
            elif pos_enc_dir2 is None:
                pos_enc_dir = pos_enc_dir1
            else:
                pos_enc_dir = np.concatenate((pos_enc_dir1, pos_enc_dir2), axis=1)
    return pe_dict, pos_enc_dir

def graph_positional_encoder(
    adj: Union[np.ndarray, spmatrix],
    num_nodes: int,
    pos_arg: Dict) -> np.ndarray:
    r"""
    Get a positional encoding that depends on the parameters.

    Parameters:

        adj: Adjacency matrix of the graph

        pos_type: The type of positional encoding to use. Supported types are:

            - laplacian_eigvec: the
            - laplacian_eigvec_eigval

    """

    pos_type = pos_arg["pos_type"]



    pos_type = pos_type.lower()
    pe_dict = {}

    if pos_type == "laplacian_eigvec":
        _, eigvecs = compute_laplacian_positional_eigvecs(
            adj=adj, num_pos=pos_arg["num_pos"], disconnected_comp=pos_arg["disconnected_comp"]
        )
        pos_enc_sign_flip = eigvecs
        pos_enc_sign_flip = np.real(pos_enc_sign_flip).astype(np.float32)
        pe_dict["pos_enc_feats_sign_flip"] = pos_enc_sign_flip
        pe_dict["pos_enc_feats_no_flip"] = None

    elif pos_type == "laplacian_eigvec_eigval":
        eigvals_tile, eigvecs = compute_laplacian_positional_eigvecs(
            adj=adj, num_pos=pos_arg["num_pos"], disconnected_comp=pos_arg["disconnected_comp"]
        )
        pos_enc_sign_flip = eigvecs
        pos_enc_no_flip = eigvals_tile
        pos_enc_sign_flip = np.real(pos_enc_sign_flip).astype(np.float32)
        pos_enc_no_flip = np.real(pos_enc_no_flip).astype(np.float32)
        pe_dict["pos_enc_feats_sign_flip"] = pos_enc_sign_flip
        pe_dict["pos_enc_feats_no_flip"] = pos_enc_no_flip

    elif pos_type == "rwse":
        rwse_pe = compute_rwse(adj=adj, ksteps=pos_arg["ksteps"], num_nodes=num_nodes)
        rwse_pe = rwse_pe.astype(np.float32)
        pe_dict["rwse"] = rwse_pe

    else:
        raise ValueError(f"Unknown `pos_type`: {pos_type}")

    return pe_dict
