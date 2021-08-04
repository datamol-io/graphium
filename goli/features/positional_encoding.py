from typing import Tuple, Optional, Dict, Union, List,Any
import numpy as np
from scipy.sparse import spmatrix
import torch

from goli.features.spectral import compute_laplacian_positional_eigvecs, compute_centroid_effective_resistances, compute_centroid_effective_resistancesv1


def get_all_positional_encoding(
    adj: Union[np.ndarray, spmatrix],
    pos_encoding_as_features: Optional[List[Dict[str, Any]]] = None,
    pos_encoding_as_directions: Optional[List[Dict[str, Any]]] = None,
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

    if not isinstance(pos_encoding_as_features, list):
        pos_encoding_as_features = [pos_encoding_as_features]
    elif pos_encoding_as_features is None:
        pos_encoding_as_features = []

    if not isinstance(pos_encoding_as_directions, list):
        pos_encoding_as_directions = [pos_encoding_as_directions]
    elif pos_encoding_as_directions is None:
        pos_encoding_as_directions = []


    pos_enc_feats_sign_flip, pos_enc_feats_no_flip, pos_enc_dir = None, None, None
    pos_enc_feats_sign_flip_list = []
    pos_enc_feats_no_flip_list = []

    for i in range(len(pos_encoding_as_features)):
        
        pos_encoding_as_features_tmp = pos_encoding_as_features[i]

        pos_enc_feats_sign_flip_tmp, pos_enc_feats_no_flip_tmp = None, None

        # Get the positional encoding for the features
        pos_enc_feats_sign_flip_tmp, pos_enc_feats_no_flip_tmp = graph_positional_encoder(
                adj, **pos_encoding_as_features_tmp
            )

        pos_enc_feats_sign_flip_list.append(pos_enc_feats_sign_flip_tmp)
        pos_enc_feats_no_flip_list.append(pos_enc_feats_no_flip_tmp)

        if pos_enc_feats_sign_flip is None:
            pos_enc_feats_sign_flip = pos_enc_feats_sign_flip_tmp
        elif (pos_enc_feats_sign_flip is not None) and (pos_enc_feats_sign_flip_tmp is not None): 
            pos_enc_feats_sign_flip = np.concatenate((pos_enc_feats_sign_flip, pos_enc_feats_sign_flip_tmp), axis=1)

        if pos_enc_feats_no_flip is None:
            pos_enc_feats_no_flip = pos_enc_feats_no_flip_tmp
        elif (pos_enc_feats_no_flip is not None) and (pos_enc_feats_no_flip_tmp is not None): 
            pos_enc_feats_no_flip = np.concatenate((pos_enc_feats_no_flip, pos_enc_feats_no_flip_tmp), axis=1)

    for i in range(len(pos_encoding_as_directions)):

        pos_encoding_as_directions_tmp = pos_encoding_as_directions[i]
        pos_enc_dir_tmp = None

        # Get the positional encoding for the directions

        if pos_encoding_as_directions_tmp in pos_encoding_as_features:

            idx = pos_encoding_as_features.index(pos_encoding_as_directions_tmp)

            # Concatenate the sign-flip and non-sign-flip positional encodings 
            if pos_enc_feats_sign_flip_list[idx] is None:
                pos_enc_dir_tmp = pos_enc_feats_no_flip_list[idx]
            elif pos_enc_feats_no_flip_list[idx] is None:
                pos_enc_dir_tmp = pos_enc_feats_sign_flip_list[idx]
            else:
                pos_enc_dir_tmp = np.concatenate((pos_enc_feats_sign_flip_list[idx], pos_enc_feats_no_flip_list[idx]), axis=1)

        else:
            pos_enc_dir1, pos_enc_dir2 = graph_positional_encoder(adj, **pos_encoding_as_directions_tmp)
            # Concatenate both positional encodings
            if pos_enc_dir1 is None:
                pos_enc_dir_tmp = pos_enc_dir2
            elif pos_enc_dir2 is None:
                pos_enc_dir_tmp = pos_enc_dir1
            else:
                pos_enc_dir_tmp = np.concatenate((pos_enc_dir1, pos_enc_dir2), axis=1)
        
        if pos_enc_dir is None:
            pos_enc_dir = pos_enc_dir_tmp
        elif (pos_enc_dir is not None) and (pos_enc_dir_tmp is not None): 
            pos_enc_dir = torch.cat([pos_enc_dir, pos_enc_dir_tmp], dim=-1)





    return pos_enc_feats_sign_flip, pos_enc_feats_no_flip, pos_enc_dir


def graph_positional_encoder(
    adj: Union[np.ndarray, spmatrix], pos_type: str, num_pos: int, disconnected_comp: bool = True, **kwargs
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
    pos_enc_sign_flip, pos_enc_no_flip = None, None

    if pos_type == "laplacian_eigvec":
        _, eigvecs = compute_laplacian_positional_eigvecs(
            adj=adj, num_pos=num_pos, disconnected_comp=disconnected_comp, **kwargs
        )
        pos_enc_sign_flip = eigvecs

    elif pos_type == "laplacian_eigvec_eigval":
        eigvals_tile, eigvecs = compute_laplacian_positional_eigvecs(
            adj=adj, num_pos=num_pos, disconnected_comp=disconnected_comp, **kwargs
        )
        pos_enc_sign_flip = eigvecs
        pos_enc_no_flip = eigvals_tile

    elif pos_type == "centroid_effective_resistances":
        centroid_idx, electric_potential = compute_centroid_effective_resistances(
            adj=adj, num_pos=num_pos, disconnected_comp=disconnected_comp, **kwargs
        )
        pos_enc_no_flip = electric_potential

    elif pos_type == "centroid_effective_resistances_v1":
        centroid_idx, electric_potential = compute_centroid_effective_resistancesv1(
            adj=adj, num_pos=num_pos, disconnected_comp=disconnected_comp, **kwargs
        )
        pos_enc_no_flip = electric_potential

    else:
        raise ValueError(f"Unknown `pos_type`: {pos_type}")

    if pos_enc_sign_flip is not None:
        pos_enc_sign_flip = torch.as_tensor(np.real(pos_enc_sign_flip)).to(torch.float32)

    if pos_enc_no_flip is not None:
        pos_enc_no_flip = torch.as_tensor(np.real(pos_enc_no_flip)).to(torch.float32)
    if (pos_enc_no_flip is not None) and (pos_enc_no_flip.ndim == 1):
        pos_enc_no_flip = pos_enc_no_flip.view(-1,1)


    return pos_enc_sign_flip, pos_enc_no_flip
