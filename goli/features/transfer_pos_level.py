from typing import Union

import torch
import numpy as np

from scipy.sparse import spmatrix, issparse, coo_matrix

from torch_geometric.utils import from_scipy_sparse_matrix


def transfer_pos_level(pe: np.ndarray, in_level: str, out_level: str, adj: Union[np.ndarray, spmatrix], num_nodes: int, cache: dict):
    r"""
    Transfer positional encoding between different positional levels (node, edge, nodepair, graph)

    Parameters:
        pe (np.ndarray): Input pe with pos_level defined by in_level
        in_level (str): pos_level of input pe
        out_level (str): desired pos_level of output pe
        adj (np.ndarray, [num_nodes, num_nodes]): Adjacency matrix of the graph
        num_nodes (int): Number of nodes in the graph
        cache (dict): Dictionary of cached objects

    Returns:
        pe (np.ndarry): Output pe with pos_level defined by out_level
    """

    if in_level == "node":

        if out_level == "node":
            pass

        elif out_level == "edge":
            pe, cache = node_to_edge(pe, adj, cache)
        
        elif out_level == "nodepair":
            pe = node_to_nodepair(pe, num_nodes)

        elif out_level == "graph":
            raise NotImplementedError("Transfer function (node -> graph) not yet implemented.")
        
        else:
            raise ValueError(f"Unknown `pos_level`: {out_level}")
     
    elif in_level == "edge":
        raise NotImplementedError("Transfer function (edge -> *) not yet implemented.")
    
    elif in_level == "nodepair":
        if len(pe.shape) == 2:
            pe = np.expand_dims(pe, -1)

        if out_level == "node":
            pe = nodepair_to_node(pe)

        elif out_level == "edge":
            pe = nodepair_to_edge(pe, adj, cache)
        
        elif out_level == "nodepair":
            pass

        elif out_level == "graph":
            raise NotImplementedError("Transfer function (nodepair -> graph) not yet implemented.")
        
        else:
            raise ValueError(f"Unknown `pos_level`: {out_level}")
   
    elif in_level == "graph":
        
        if out_level == "node":
            pe = graph_to_node(pe, num_nodes, cache)

        elif out_level in ["edge", "nodepair"]:
            raise NotImplementedError("Transfer function (graph -> edge/nodepair) not yet implemented.")

        else:
            raise ValueError(f"Unknown `pos_level`: {out_level}")
   
    else:
        raise ValueError(f"Unknown `pos_level`: {in_level}")

    return pe


# Transfer functions between different levels, i.e., node, edge, nodepair and graph level.

# TODO: 
#   - Implement missing transfer functions below
#   - Are transfer functions graph -> edge/nodepair and edge -> graph needed?


def node_to_edge(
        pe:  np.ndarray,
        adj: Union[np.ndarray, spmatrix],
        cache: dict
) ->  np.ndarray:
    r"""
    Get an edge-level positional encoding from a node-level positional encoding.
     -> For each edge, concatenate features from sender and receiver node.

    Parameters:
        pe (np.ndarray, [num_nodes, num_feat]): Node-level positional encoding
        adj (np.ndarray, [num_nodes, num_nodes]): Adjacency matrix of the graph
        cache (dict): Dictionary of cached objects
    
    Returns:
        edge_pe (np.ndarray, [2 * num_edges, 2 * num_feat]): Edge-level positional encoding
        cache (dict): Updated dictionary of cached objects
    """

    if not issparse(adj):
        if 'coo_adj' in cache:
            adj = cache['coo_adj']
        elif 'csr_adj' in cache:
            adj = cache['csr_adj']
        else:
            adj = coo_matrix(adj, dtype=np.float64)
            cache['coo_adj'] = adj
    
    edge_index, _ = from_scipy_sparse_matrix(adj)
    src, dst = edge_index[0], edge_index[1]

    edge_pe = np.concatenate((pe[src], pe[dst]), axis=-1)

    return edge_pe, cache


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
        adj: Union[np.ndarray, spmatrix]
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
        adj: Union[np.ndarray, spmatrix]
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
        pe (np.ndarray, [num_nodes, num_nodes, num_feat]): Nodepair-level positional encoding
        stats_list (list): List of statistics to calculate per row/col of nodepair-level pe
    
    Returns:
        node_pe (np.ndarray, [num_nodes, 2 * len(stats_list) * num_feat]): Node-level positional encoding
    """
    # TODO: Support multi-dim. nodepair-level pes (np.ndarray, [num_nodes, num_nodes, num_feat])
    # Done

    num_feat = pe.shape[-1]

    node_pe_list = []

    for stat in stats_list:
        for i in range(num_feat):
            node_pe_list.append(stat(pe[..., i], axis=0))
            node_pe_list.append(stat(pe[..., i], axis=1))
    node_pe = np.stack(node_pe_list, axis=-1)

    return node_pe


def nodepair_to_edge(
        pe: np.ndarray,
        adj: Union[np.ndarray, spmatrix],
        cache: dict
) ->  np.ndarray:
    r"""
    Get a edge-level positional encoding from a nodepair-level positional encoding.
     -> Mask and sparsify nodepair-level positional encoding

    Parameters:
        pe (np.ndarray, [num_nodes, num_nodes, num_feat]): Nodepair-level positional encoding
        adj (np.ndarray, [num_nodes, num_nodes]): Adjacency matrix of the graph
        cache (dict): Dictionary of cached objects
    
    Returns:
        edge_pe (np.ndarray, [num_edges, num_feat]): Edge-level positional encoding
    """
    # TODO: Support multi-dim. nodepair-level pes (np.ndarray, [num_nodes, num_nodes, num_feat])
    # Done

    num_feat = pe.shape[-1]
    
    if not isinstance(adj, coo_matrix):
        if 'coo_adj' in cache:
            adj = cache['coo_adj']
        else:
            adj = coo_matrix(adj, dtype=np.float64)
        cache['coo_adj'] = adj

    src, dst = adj.row, adj.col

    edge_pe = np.zeros((len(src), num_feat))

    for i in range(len(src)):
        edge_pe[i,...] = pe[src[i], dst[i]]

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
        pe:  Union[np.ndarray, list],
        num_nodes: int,
        cache: dict
) ->  np.ndarray:
    r"""
    Get a node-level positional encoding from a nodepair-level positional encoding.
     -> E.g., expand dimension of graph-level pe

    Parameters:
        pe (np.ndarray, [num_feat]): Nodepair-level positional encoding (or list of them if graph disconnected)
        num_nodes (int): Number of nodes in the graph
        cache (dict): Dictionary of cached objects
    
    Returns:
        node_pe (np.ndarray, [num_nodes, num_feat]): Node-level positional encoding
    """

    node_pe = None

    # The key 'components' is only in cache if disconnected_comp == True when computing base pe
    if 'components' in cache:
        if len(cache['components']) > 1:
            node_pe = np.zeros((num_nodes, len(pe)))
            components = cache['components']

            for i, component in enumerate(components):
                comp = list(component)
                node_pe[comp, :] = np.real(pe[i])

    if node_pe is None:
        node_pe = np.tile(pe, (num_nodes, 1))

    return node_pe
