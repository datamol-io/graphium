from typing import Tuple, Optional, Dict, Union
from copy import deepcopy
import numpy as np
import torch
from scipy.sparse import spmatrix
from collections import OrderedDict

from torch_geometric.utils.sparse import dense_to_sparse

from goli.features.spectral import compute_laplacian_positional_eigvecs
from goli.features.rw import compute_rwse
from goli.features.electrostatic import compute_electrostatic_interactions
from goli.features.commute import compute_commute_distances
from goli.features.graphormer import compute_graphormer_distances


def get_all_positional_encoding(
    adj: Union[np.ndarray, spmatrix],
    num_nodes: int,
    pos_encoding_as_features: Optional[Dict] = None,
) -> Tuple["OrderedDict[str, np.ndarray]", "OrderedDict[str, np.ndarray]"]:
    r"""
    Get features positional encoding.

    Parameters:
        adj: Adjacency matrix of the graph
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
        for pos_type in pos_encoding_as_features["pos_types"]:
            pos_args = pos_encoding_as_features["pos_types"][pos_type]
            pos_level = pos_args["pos_level"]
            this_pe, cache = graph_positional_encoder(deepcopy(adj), num_nodes, pos_args, cache)
            this_pe = {f"{pos_level}_{pos_type}": this_pe}
            pe_dict.update(this_pe)

    return pe_dict


def graph_positional_encoder(
    adj: Union[np.ndarray, spmatrix], num_nodes: int, pos_arg: Dict, cache: dict[np.ndarray]
) -> Dict[str, np.ndarray]:
    r"""
    Get a positional encoding that depends on the parameters.

    Parameters:
        adj: Adjacency matrix of the graph

        pos_type: The type of positional encoding to use. Supported types are:
            - laplacian_eigvec              \
            - laplacian_eigvec_eigval        \  -> cashe eigendecomp
            - rwse
            - electrostatic                 \
            - commute                        \  -> cashe pinvL
            - graphormer

    Returns:
        pe_dict: Dictionary of positional and structural encodings

    """
    pos_type = pos_arg["pos_type"]
    pos_type = pos_type.lower()
    pos_level = pos_arg["pos_level"]
    pos_level = pos_level.lower()

    # Convert to numpy array
    if isinstance(adj, torch.sparse.Tensor):
        adj = adj.to_dense().numpy()
    elif isinstance(adj, torch.Tensor):
        adj = adj.numpy()

    # Calculate positional encoding
    if pos_type == "laplacian_eigvec":
        _, pe, base_level, cache = compute_laplacian_positional_eigvecs(adj, pos_arg["num_pos"], cache, pos_arg["disconnected_comp"])

    elif pos_type == "laplacian_eigval":
        pe, _, base_level, cache = compute_laplacian_positional_eigvecs(adj, pos_arg["num_pos"], cache, pos_arg["disconnected_comp"])

    elif pos_type == "rwse":
        pe, base_level = compute_rwse(adj=adj.astype(np.float32), ksteps=pos_arg["ksteps"], num_nodes=num_nodes)

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
    if base_level == "pair":
        
        if pos_level == "edge":
            pe = pair_to_edge(pe, adj)

        elif pos_level == "node":
            pe = pair_to_node(pe, stats_list=pos_arg["stats"])

    # TODO: Implement conversion between other pos levels (e.g., node -> pair/edge)

    return pe, cache


def pair_to_edge(
        pe: np.ndarray,
        adj: np.ndarray
) ->  np.ndarray:
    r"""
    Get a edge-level positional encoding from a nodepair-level positional encoding.

    Parameters:
        pe [num_nodes, num_nodes]: Nodepair-level positional encoding
        adj [num_nodes, num_nodes]: Adjacency matrix of the graph
    
    Returns:
        pe [num_edges]: Edge-level positional encoding

    """

    edge_pe = np.where(adj != 0, pe, 0)
    edge_pe = torch.from_numpy(edge_pe)
    edge_pe = dense_to_sparse(edge_pe)

    return edge_pe


def pair_to_node(
        pe:  np.ndarray,
        stats_list: list = [np.min, np.mean, np.std]
) ->  np.ndarray:
    r"""
    Get a node-level positional encoding from a graph-level positional encoding.

    Parameters:
        pe [num_nodes, num_nodes]: Nodepair-level positional encoding
        num_nodes [int]: Number of nodes of graph
    
    Returns:
        pe [num_nodes, 2 * len(stats_list)]: Node-level positional encoding

    """

    node_pe_list = []

    for stat in stats_list:
        node_pe_list.append(stat(pe, axis=0), stat(pe, axis=1))
    node_pe = np.stack(node_pe_list, axis=-1)

    return node_pe


def graph_to_node(
        pe:  np.ndarray,
        num_nodes: int
) ->  np.ndarray:
    r"""
    Get a node-level positional encoding from a nodepair-level positional encoding.

    Parameters:
        pe [float]: Nodepair-level positional encoding
        num_nodes [int]: Number of nodes in the graph
    
    Returns:
        pe [num_nodes]: Node-level positional encoding

    """

    node_pe = None

    return node_pe
