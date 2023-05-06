from typing import Tuple, Union, Optional, Dict, Any
from copy import deepcopy
import numpy as np
import torch
from scipy.sparse import spmatrix
from collections import OrderedDict

from goli.features.spectral import compute_laplacian_pe
from goli.features.rw import compute_rwse
from goli.features.electrostatic import compute_electrostatic_interactions
from goli.features.commute import compute_commute_distances
from goli.features.graphormer import compute_graphormer_distances
from goli.features.transfer_pos_level import transfer_pos_level


def get_all_positional_encodings(
        adj: Union[np.ndarray, spmatrix],
        num_nodes: int,
        pos_encoding_as_features: Optional[Dict] = None,
) -> Tuple[OrderedDict[str, np.ndarray]]:
    r"""
    Get features positional encoding.

    Parameters:
        adj [num_nodes, num_nodes]: Adjacency matrix of the graph
        num_nodes: Number of nodes in the graph
        pos_encoding_as_features: keyword arguments for function `graph_positional_encoder`
            to generate positional encoding for node features.

    Returns:
        pe_dict: Dictionary of positional and structural encodings
    """

    pos_encoding_as_features = {} if pos_encoding_as_features is None else pos_encoding_as_features

    pe_dict = OrderedDict()
    
    # Initialize cache
    cache = {}

    # Get the positional encoding for the features
    if len(pos_encoding_as_features) > 0:
        for pos_name in pos_encoding_as_features["pos_types"]:
            pos_args = pos_encoding_as_features["pos_types"][pos_name]
            pos_type = pos_args["pos_type"]
            pos_level = pos_args["pos_level"]
            this_pe, cache = graph_positional_encoder(deepcopy(adj), num_nodes, pos_type, pos_args, cache)
            if pos_level == 'node':
                pe_dict.update({f"{pos_type}": this_pe})
            else:
                pe_dict.update({f"{pos_level}_{pos_type}": this_pe})

    return pe_dict


def graph_positional_encoder(
        adj: Union[np.ndarray, spmatrix],
        num_nodes: int,
        pos_type: str,
        pos_arg: Dict[str, Any],
        cache: Dict[str, Any]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    r"""
    Get a positional encoding that depends on the parameters.

    Parameters:
        adj [num_nodes, num_nodes]: Adjacency matrix of the graph
        num_nodes: Number of nodes in the graph
        pos_type: Type of positional encoding
        pos_args: Arguments 
            pos_type: The type of positional encoding to use. Supported types are:
                - laplacian_eigvec              \
                - laplacian_eigval               \  -> cache connected comps. & eigendecomp.
                - rwse
                - electrostatic                 \
                - commute                        \  -> cache pinvL
                - graphormer
            pos_level: Positional level to output
                - node
                - edge
                - nodepair
                - graph
            cache: Dictionary of cached objects

    Returns:
        pe: Positional or structural encoding
        cache: Updated dictionary of cached objects
    """
    
    pos_type = pos_type.lower()
    pos_level = pos_arg["pos_level"]
    pos_level = pos_level.lower()

    # Convert to numpy array
    if isinstance(adj, torch.sparse.Tensor):
        adj = adj.to_dense().numpy()
    elif isinstance(adj, torch.Tensor):
        adj = adj.numpy()
    adj = adj.astype(np.float64)

    # Calculate positional encoding
    if pos_type == "laplacian_eigvec":
        pe, base_level, cache = compute_laplacian_pe(adj, pos_arg["num_pos"], cache, pos_type, pos_arg["disconnected_comp"])

    elif pos_type == "laplacian_eigval":
        pe, base_level, cache = compute_laplacian_pe(adj, pos_arg["num_pos"], cache, pos_type, pos_arg["disconnected_comp"])

    elif pos_type == "rw_return_probs":
        pe, base_level, cache = compute_rwse(adj.astype(np.float32), ksteps=pos_arg["ksteps"], num_nodes=num_nodes, cache=cache, pos_type=pos_type)

    elif pos_type == "rw_transition_probs":
        pe, base_level, cache = compute_rwse(adj.astype(np.float32), ksteps=pos_arg["ksteps"], num_nodes=num_nodes, cache=cache, pos_type=pos_type)

    elif pos_type == "electrostatic":
        pe, base_level, cache = compute_electrostatic_interactions(adj, cache)

    elif pos_type == "commute":
        pe, base_level, cache = compute_commute_distances(adj, num_nodes, cache)

    elif pos_type == "graphormer":
        pe, base_level, cache = compute_graphormer_distances(adj, num_nodes, cache)

    else:
        raise ValueError(f"Unknown `pos_type`: {pos_type}")
    
    pe = np.real(pe).astype(np.float32)
    
    # Convert between different pos levels
    pe = transfer_pos_level(pe, base_level, pos_level, adj, num_nodes, cache)
    
    return pe, cache