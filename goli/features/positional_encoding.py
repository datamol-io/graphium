from typing import Tuple, Optional, Dict, Union
import numpy as np
from scipy.sparse import spmatrix
from collections import OrderedDict

from goli.features.spectral import compute_laplacian_positional_eigvecs
from goli.features.rw import compute_rwse


def get_all_positional_encoding(
    adj: Union[np.ndarray, spmatrix],
    num_nodes: int,
    pos_encoding_as_features: Optional[Dict] = None,
    pos_encoding_as_directions: Optional[Dict] = None,
) -> Tuple["OrderedDict[str, np.ndarray]", "OrderedDict[str, np.ndarray]"]:
    r"""
    Get features positional encoding and direction positional encoding.

    Parameters:
        adj: Adjacency matrix of the graph
        pos_encoding_as_features: keyword arguments for function `graph_positional_encoder`
            to generate positional encoding for node features.
        pos_encoding_as_directions: keyword arguments for function `graph_positional_encoder`
            to generate positional encoding for directional features,
            for exemple, with the DGN model (Directional Graph Networks)

    Returns:
        pe_dict: Dictionary of positional and structural encodings
        pe_dir_dict: Dictionary of positional and structural encodings to be used for directional
            features, for exemple, with the DGN model (Directional Graph Networks)
    """

    pos_encoding_as_features = {} if pos_encoding_as_features is None else pos_encoding_as_features
    pos_encoding_as_directions = {} if pos_encoding_as_directions is None else pos_encoding_as_directions

    pe_dict, pe_dir_dict = OrderedDict(), OrderedDict()

    # Get the positional encoding for the features
    if len(pos_encoding_as_features) > 0:
        for pos in pos_encoding_as_features["pos_types"]:
            pos_args = pos_encoding_as_features["pos_types"][pos]
            this_pe = graph_positional_encoder(adj, num_nodes, pos_args)
            this_pe = {f"{pos}/{key}": val for key, val in this_pe.items()}
            pe_dict.update(this_pe)

    # Get the positional encoding for the directions (useful for directional GNNs and asymetric pooling)
    if len(pos_encoding_as_directions) > 0:
        for pos in pos_encoding_as_directions["pos_types"]:
            pos_args = pos_encoding_as_directions["pos_types"][pos]
            this_pe = graph_positional_encoder(adj, num_nodes, pos_args)
            this_pe = {f"{pos}/{key}": val for key, val in this_pe.items()}
            pe_dir_dict.update(this_pe)

    return pe_dict, pe_dir_dict


def graph_positional_encoder(adj: Union[np.ndarray, spmatrix], num_nodes: int, pos_arg: Dict) -> np.ndarray:
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
            adj=adj,
            num_pos=pos_arg["num_pos"],
            disconnected_comp=pos_arg["disconnected_comp"],
        )
        eigvecs = np.real(eigvecs).astype(np.float32)
        pe_dict["eigvecs"] = eigvecs
        pe_dict["eigvals"] = None

    elif pos_type == "laplacian_eigvec_eigval":
        eigvals_tile, eigvecs = compute_laplacian_positional_eigvecs(
            adj=adj,
            num_pos=pos_arg["num_pos"],
            disconnected_comp=pos_arg["disconnected_comp"],
        )
        eigvecs = np.real(eigvecs).astype(np.float32)
        eigvals_tile = np.real(eigvals_tile).astype(np.float32)
        pe_dict["eigvecs"] = eigvecs
        pe_dict["eigvals"] = eigvals_tile

    elif pos_type == "rwse":
        rwse_pe = compute_rwse(adj=adj, ksteps=pos_arg["ksteps"], num_nodes=num_nodes)
        rwse_pe = rwse_pe.astype(np.float32)
        pe_dict["rwse"] = rwse_pe

    else:
        raise ValueError(f"Unknown `pos_type`: {pos_type}")

    return pe_dict
