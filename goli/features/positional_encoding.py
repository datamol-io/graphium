from typing import Tuple, Optional, Dict, Union
from copy import deepcopy
import numpy as np
import torch
from scipy.sparse import spmatrix, issparse, coo_matrix
from collections import OrderedDict

from torch_geometric.utils.sparse import dense_to_sparse

from goli.features.spectral import compute_laplacian_positional_eigvecs
from goli.features.rw import compute_rwse
from goli.features.electrostatic import compute_electrostatic_interactions
from goli.features.commute import compute_commute_distances
from goli.features.graphormer import compute_graphormer_distances


def get_all_positional_encodings(
        adj: Union[np.ndarray, spmatrix],
        num_nodes: int,
        pos_encoding_as_features: Optional[Dict] = None,
) -> Tuple["OrderedDict[str, np.ndarray]", "OrderedDict[str, np.ndarray]"]:
    r"""
    Get features positional encoding.

    Parameters:
        adj (np.ndarray, [num_nodes, num_nodes]): Adjacency matrix of the graph
        num_nodes (int): Number of nodes in the graph
        pos_encoding_as_features (dict): keyword arguments for function `graph_positional_encoder`
            to generate positional encoding for node features.

    Returns:
        pe_dict (dict): Dictionary of positional and structural encodings
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
        pos_arg: dict,
        cache: dict[np.ndarray]
) -> Dict[str, np.ndarray]:
    r"""
    Get a positional encoding that depends on the parameters.

    Parameters:
        adj (np.ndarray, [num_nodes, num_nodes]): Adjacency matrix of the graph
        num_nodes (int): Number of nodes in the graph
        pos_type (str): Type of positional encoding
        pos_args (dict): Arguments 
            pos_type (str): The type of positional encoding to use. Supported types are:
                - laplacian_eigvec              \
                - laplacian_eigval               \  -> cache connected comps. & eigendecomp.
                - rwse
                - electrostatic                 \
                - commute                        \  -> cache pinvL
                - graphormer
            pos_level (str): Positional level to output
                - node
                - edge
                - nodepair
                - graph
            cache (dict): Dictionary of cached objects

    Returns:
        pe (np.ndarray): Positional or structural encoding
        cache (dict): Updated dictionary of cached objects
    """
    
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
    if base_level == "node":

        if pos_level == "node":
            pass

        elif pos_level == "edge":
            pe = node_to_edge(pe, adj)
        
        elif pos_level == "nodepair":
            pe = node_to_nodepair(pe, num_nodes)

        elif pos_level == "graph":
            raise NotImplementedError("Transfer function (node -> graph) not yet implemented.")
        
        else:
            raise ValueError(f"Unknown `pos_level`: {pos_level}")
     
    elif base_level == "nodepair":

        if pos_level == "node":
            pe = nodepair_to_node(pe)

        elif pos_level == "edge":
            pe = nodepair_to_edge(pe, adj)
            if len(pe.shape) == 1:
                pe = np.expand_dims(pe, -1)
        
        elif pos_level == "nodepair":
            if len(pe.shape) == 2:
                pe = np.expand_dims(pe, -1)

        elif pos_level == "graph":
            raise NotImplementedError("Transfer function (nodepair -> graph) not yet implemented.")
        
        else:
            raise ValueError(f"Unknown `pos_level`: {pos_level}")
   
    elif base_level in ["edge", "graph"]:
        raise NotImplementedError("Transfer function (edge/graph -> *) not yet implemented.")

    else:
        raise ValueError(f"Unknown `base_level`: {base_level}")

    # TODO: Implement conversion between other positional levels (node -> graph, edge/graph -> *).

    return pe, cache


# Transfer functions between different levels, i.e., node, edge, nodepair and graph level.

# TODO: 
#   - Implement missing transfer functions below
#   - Are transfer functions graph -> edge/nodepair and edge -> graph needed?


def node_to_edge(
        pe:  np.ndarray,
        adj: np.ndarray
) ->  np.ndarray:
    r"""
    Get an edge-level positional encoding from a node-level positional encoding.
     -> For each edge, concatenate features from sender and receiver node.

    Parameters:
        pe (np.ndarray, [num_nodes, num_feat]): Node-level positional encoding
        adj (np.ndarray, [num_nodes, num_nodes]): Adjacency matrix of the graph
    
    Returns:
        edge_pe (np.ndarray, [2 * num_edges, 2 * num_feat]): Edge-level positional encoding
    """

    if not issparse(adj):
        adj = coo_matrix(adj, dtype=np.float16)

    edge_pe = np.concatenate((pe[adj.row], pe[adj.col]), axis=-1)

    return edge_pe


def node_to_nodepair(
        pe: np.ndarray,
        num_nodes: int
) ->  np.ndarray:
    r"""
    Get a nodepair-level positional encoding from a node-level positional encoding.
     -> Concatenate features from node i and j at position (i,j) in nodepair_pe.

    Parameters:
        pe (np.ndarray, [num_nodes, num_feat]): Node-level positional encoding
        num_nodes (int): Number of nodes in the graph
    
    Returns:
        nodepair_pe (np.ndarray, [num_nodes, num_nodes, 2 * num_feat]): Nodepair-level positional encoding
    """
    
    if pe.shape[0] != num_nodes:
        raise ValueError(f"{pe.shape[0]} != {num_nodes}")

    expanded_pe = np.expand_dims(pe, axis=1)
    expanded_pe = np.repeat(expanded_pe, repeats=num_nodes, axis=1)

    nodepair_pe = np.concatenate([expanded_pe, expanded_pe.transpose([1,0,2])], axis=-1)

    return nodepair_pe


def node_to_graph(
        pe: np.ndarray,
        num_nodes: int
) ->  np.ndarray:
    r"""
    Get a graph-level positional encoding from a node-level positional encoding.
     -> E.g., min/max/mean-pooling of node features.

    Parameters:
        pe (np.ndarray, [num_nodes, num_feat]): Node-level positional encoding
        num_nodes (int): Number of nodes in the graph
    
    Returns:
        graph_pe (np.ndarray, [1, num_feat]): Graph-level positional encoding
    """

    raise NotImplementedError("Transfer function (node -> graph) not yet implemented.")


def edge_to_node(
        pe:  np.ndarray,
        adj: np.ndarray
) ->  np.ndarray:
    r"""
    Get a node-level positional encoding from an edge-level positional encoding.
     -> E.g., min/max/mean-pooling of information from edges (i,j) that contain node i

    Parameters:
        pe (np.ndarray, [num_edges, num_feat]): Edge-level positional encoding
        adj (np.ndarray, [num_nodes, num_nodes]): Adjacency matrix of the graph
    
    Returns:
        node_pe (np.ndarray, [num_edges, num_feat]): Node-level positional encoding
    """

    raise NotImplementedError("Transfer function (edge -> node) not yet implemented.")


def edge_to_nodepair(
        pe:  np.ndarray,
        adj: np.ndarray
) ->  np.ndarray:
    r"""
    Get a nodepair-level positional encoding from an edge-level positional encoding.
     -> E.g., zero-padding of non-existing edges.

    Parameters:
        pe (np.ndarray, [num_edges, num_feat]): Edge-level positional encoding
        adj (np.ndarray, [num_nodes, num_nodes]): Adjacency matrix of the graph
    
    Returns:
        nodepair_pe (np.ndarray, [num_edges, num_edges, num_feat]): Nodepair-level positional encoding
    """

    raise NotImplementedError("Transfer function (edge -> nodepair) not yet implemented.")


def edge_to_graph(
        pe:  np.ndarray
) ->  np.ndarray:
    r"""
    Get a graph-level positional encoding from an edge-level positional encoding.

    Parameters:
        pe (np.ndarray, [num_edges, num_feat]): Edge-level positional encoding
    
    Returns:
        graph_pe (np.ndarray, [1, num_feat]): Graph-level positional encoding
    """

    raise NotImplementedError("Transfer function (edge -> graph) not yet implemented.")


def nodepair_to_node(
        pe:  np.ndarray,
        stats_list: list = [np.min, np.mean, np.std]
) ->  np.ndarray:
    r"""
    Get a node-level positional encoding from a graph-level positional encoding.
     -> Calculate statistics over rows & cols of input positional encoding

    Parameters:
        pe (np.ndarray, [num_nodes, num_nodes]): Nodepair-level positional encoding
        stats_list (list): List of statistics to calculate per row/col of nodepair-level pe
    
    Returns:
        node_pe (np.ndarray, [num_nodes, 2 * len(stats_list)]): Node-level positional encoding
    """

    node_pe_list = []

    for stat in stats_list:
        node_pe_list.append(stat(pe, axis=0))
        node_pe_list.append(stat(pe, axis=1))
    node_pe = np.stack(node_pe_list, axis=-1)

    return node_pe


def nodepair_to_edge(
        pe: np.ndarray,
        adj: np.ndarray
) ->  np.ndarray:
    r"""
    Get a edge-level positional encoding from a nodepair-level positional encoding.
     -> Mask and sparsify nodepair-level positional encoding

    Parameters:
        pe (np.ndarray, [num_nodes, num_nodes]): Nodepair-level positional encoding
        adj (np.ndarray, [num_nodes, num_nodes]): Adjacency matrix of the graph
    
    Returns:
        edge_pe (np.ndarray, [num_edges]): Edge-level positional encoding
    """
    # TODO: Support multi-dim. nodepair-level pes (np.ndarray, [num_nodes, num_nodes, num_feat])

    if issparse(adj):
        adj = adj.astype(np.float64)
        adj = adj.toarray()

    edge_pe = np.where(adj != 0, pe, 0)
    edge_pe = torch.from_numpy(edge_pe)
    _, edge_pe = dense_to_sparse(edge_pe)

    return edge_pe


def nodepair_to_graph(
        pe:  np.ndarray,
        num_nodes: int
) ->  np.ndarray:
    r"""
    Get a graph-level positional encoding from a nodepair-level positional encoding.
     -> E.g., min/max/mean-pooling of entries of input pe

    Parameters:
        pe (np.ndarray, [num_nodes, num_nodes, num_feat]): Nodepair-level positional encoding
        num_nodes (int): Number of nodes in the graph
    
    Returns:
        graph_pe (np.ndarray, [1, num_feat]): Graph-level positional encoding
    """

    raise NotImplementedError("Transfer function (nodepair -> graph) not yet implemented.")


def graph_to_node(
        pe:  np.ndarray,
        num_nodes: int
) ->  np.ndarray:
    r"""
    Get a node-level positional encoding from a nodepair-level positional encoding.
     -> E.g., expand dimension of graph-level pe

    Parameters:
        pe (np.ndarray, [num_feat]): Nodepair-level positional encoding
        num_nodes (int): Number of nodes in the graph
    
    Returns:
        node_pe (np.ndarray, [num_nodes, num_feat]): Node-level positional encoding
    """

    raise NotImplementedError("Transfer function (graph -> node) not yet implemented.")
