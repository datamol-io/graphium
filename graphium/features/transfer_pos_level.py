from typing import Tuple, Union, List, Dict, Any, Optional

import numpy as np

from scipy.sparse import spmatrix, issparse, coo_matrix

from torch_geometric.utils import from_scipy_sparse_matrix


def transfer_pos_level(
    pe: np.ndarray,
    in_level: str,
    out_level: str,
    adj: Union[np.ndarray, spmatrix],
    num_nodes: int,
    cache: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    r"""
    Transfer positional encoding between different positional levels (node, edge, nodepair, graph)

    Parameters:
        pe: Input pe with pos_level defined by in_level
        in_level: pos_level of input pe
        out_level: desired pos_level of output pe
        adj [num_nodes, num_nodes]: Adjacency matrix of the graph
        num_nodes: Number of nodes in the graph
        cache: Dictionary of cached objects

    Returns:
        pe: Output pe with pos_level defined by out_level
    """

    if cache is None:
        cache = {}

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
            pe, cache = nodepair_to_edge(pe, adj, cache)

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
    pe: np.ndarray, adj: Union[np.ndarray, spmatrix], cache: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    r"""
    Get an edge-level positional encoding from a node-level positional encoding.
     -> For each edge, concatenate the sum and absolute difference of pe of source and destination node.

    Parameters:
        pe [num_nodes, num_feat]: Node-level positional encoding
        adj [num_nodes, num_nodes]: Adjacency matrix of the graph
        cache: Dictionary of cached objects

    Returns:
        edge_pe [2 * num_edges, 2 * num_feat]: Edge-level positional encoding
        cache: Updated dictionary of cached objects
    """

    if cache is None:
        cache = {}

    if not issparse(adj):
        if "coo_adj" in cache:
            adj = cache["coo_adj"]
        elif "csr_adj" in cache:
            adj = cache["csr_adj"]
        else:
            adj = coo_matrix(adj, dtype=np.float64)
            cache["coo_adj"] = adj

    edge_index, _ = from_scipy_sparse_matrix(adj)
    src, dst = edge_index[0], edge_index[1]

    pe_sum = pe[src] + pe[dst]
    pe_abs_diff = np.abs(pe[src] - pe[dst])

    edge_pe = np.concatenate((pe_sum, pe_abs_diff), axis=-1)

    return edge_pe, cache


def node_to_nodepair(pe: np.ndarray, num_nodes: int) -> np.ndarray:
    r"""
    Get a nodepair-level positional encoding from a node-level positional encoding.
     -> For each nodepair (i,j) concatenate the sum and absolute difference of pe at node i and j.

    Parameters:
        pe [num_nodes, num_feat]: Node-level positional encoding
        num_nodes: Number of nodes in the graph

    Returns:
        nodepair_pe [num_nodes, num_nodes, 2 * num_feat]: Nodepair-level positional encoding
    """

    expanded_pe = np.expand_dims(pe, axis=1)
    expanded_pe = np.repeat(expanded_pe, repeats=num_nodes, axis=1)

    pe_sum = expanded_pe + expanded_pe.transpose([1, 0, 2])
    pe_abs_diff = np.abs(expanded_pe - expanded_pe.transpose([1, 0, 2]))

    nodepair_pe = np.concatenate((pe_sum, pe_abs_diff), axis=-1)

    return nodepair_pe


def node_to_graph(pe: np.ndarray, num_nodes: int) -> np.ndarray:
    r"""
    Get a graph-level positional encoding from a node-level positional encoding.
     -> E.g., min/max/mean-pooling of node features.

    Parameters:
        pe [num_nodes, num_feat]: Node-level positional encoding
        num_nodes: Number of nodes in the graph

    Returns:
        graph_pe [1, num_feat]: Graph-level positional encoding
    """

    raise NotImplementedError("Transfer function (node -> graph) not yet implemented.")


def edge_to_node(pe: np.ndarray, adj: Union[np.ndarray, spmatrix]) -> np.ndarray:
    r"""
    Get a node-level positional encoding from an edge-level positional encoding.
     -> E.g., min/max/mean-pooling of information from edges (i,j) that contain node i

    Parameters:
        pe [num_edges, num_feat]: Edge-level positional encoding
        adj [num_nodes, num_nodes]: Adjacency matrix of the graph

    Returns:
        node_pe [num_edges, num_feat]: Node-level positional encoding
    """

    raise NotImplementedError("Transfer function (edge -> node) not yet implemented.")


def edge_to_nodepair(
    pe: np.ndarray, adj: Union[np.ndarray, spmatrix], num_nodes: int, cache: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    r"""
    Get a nodepair-level positional encoding from an edge-level positional encoding.
     -> Zero-padding of non-existing edges.

    Parameters:
        pe [num_edges, num_feat]: Edge-level positional encoding
        adj [num_nodes, num_nodes]: Adjacency matrix of the graph
        num_nodes: Number of nodes in the graph
        cache: Dictionary of cached objects

    Returns:
        nodepair_pe [num_edges, num_edges, num_feat]: Nodepair-level positional encoding
        cache: Updated dictionary of cached objects
    """

    if cache is None:
        cache = {}

    num_feat = pe.shape[-1]

    if not isinstance(adj, coo_matrix):
        if "coo_adj" in cache:
            adj = cache["coo_adj"]
        else:
            adj = coo_matrix(adj, dtype=np.float64)
        cache["coo_adj"] = adj

    dst, src = adj.row, adj.col

    nodepair_pe = np.zeros((num_nodes, num_nodes, num_feat))

    for i in range(len(dst)):
        nodepair_pe[dst[i], src[i], ...] = pe[i, ...]

    return nodepair_pe, cache


def edge_to_graph(pe: np.ndarray) -> np.ndarray:
    r"""
    Get a graph-level positional encoding from an edge-level positional encoding.

    Parameters:
        pe [num_edges, num_feat]: Edge-level positional encoding

    Returns:
        graph_pe [1, num_feat]: Graph-level positional encoding
    """

    raise NotImplementedError("Transfer function (edge -> graph) not yet implemented.")


def nodepair_to_node(pe: np.ndarray, stats_list: List = [np.min, np.mean, np.std]) -> np.ndarray:
    r"""
    Get a node-level positional encoding from a graph-level positional encoding.
     -> Calculate statistics over rows & cols of input positional encoding

    Parameters:
        pe [num_nodes, num_nodes, num_feat]: Nodepair-level positional encoding
        stats_list: List of statistics to calculate per row/col of nodepair-level pe

    Returns:
        node_pe [num_nodes, 2 * len(stats_list) * num_feat]: Node-level positional encoding
    """

    num_feat = pe.shape[-1]

    node_pe_list = []

    for stat in stats_list:
        for i in range(num_feat):
            node_pe_list.append(stat(pe[..., i], axis=0))
            node_pe_list.append(stat(pe[..., i], axis=1))
    node_pe = np.stack(node_pe_list, axis=-1)

    return node_pe


def nodepair_to_edge(
    pe: np.ndarray, adj: Union[np.ndarray, spmatrix], cache: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    r"""
    Get a edge-level positional encoding from a nodepair-level positional encoding.
     -> Mask and sparsify nodepair-level positional encoding

    Parameters:
        pe [num_nodes, num_nodes, num_feat]: Nodepair-level positional encoding
        adj [num_nodes, num_nodes]: Adjacency matrix of the graph
        cache: Dictionary of cached objects

    Returns:
        edge_pe [num_edges, num_feat]: Edge-level positional encoding
        cache: Updated dictionary of cached objects
    """

    if cache is None:
        cache = {}

    num_feat = pe.shape[-1]

    if not isinstance(adj, coo_matrix):
        if "coo_adj" in cache:
            adj = cache["coo_adj"]
        else:
            adj = coo_matrix(adj, dtype=np.float64)
        cache["coo_adj"] = adj

    dst, src = adj.row, adj.col

    edge_pe = np.zeros((len(dst), num_feat))

    for i in range(len(src)):
        edge_pe[i, ...] = pe[dst[i], src[i]]

    return edge_pe, cache


def nodepair_to_graph(pe: np.ndarray, num_nodes: int) -> np.ndarray:
    r"""
    Get a graph-level positional encoding from a nodepair-level positional encoding.
     -> E.g., min/max/mean-pooling of entries of input pe

    Parameters:
        pe [num_nodes, num_nodes, num_feat]: Nodepair-level positional encoding
        num_nodes: Number of nodes in the graph

    Returns:
        graph_pe [1, num_feat]: Graph-level positional encoding
    """

    raise NotImplementedError("Transfer function (nodepair -> graph) not yet implemented.")


def graph_to_node(
    pe: Union[np.ndarray, List], num_nodes: int, cache: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    r"""
    Get a node-level positional encoding from a nodepair-level positional encoding.
     -> E.g., expand dimension of graph-level pe

    Parameters:
        pe [num_feat]: Nodepair-level positional encoding (or list of them if graph disconnected)
        num_nodes: Number of nodes in the graph
        cache: Dictionary of cached objects

    Returns:
        node_pe [num_nodes, num_feat]: Node-level positional encoding
    """

    if cache is None:
        cache = {}

    node_pe = None

    # The key 'components' is only in cache if disconnected_comp == True when computing base pe
    if "components" in cache:
        if len(cache["components"]) > 1:
            node_pe = np.zeros((num_nodes, len(pe)))
            components = cache["components"]

            for i, component in enumerate(components):
                comp = list(component)
                node_pe[comp, :] = np.real(pe[i])

    if node_pe is None:
        node_pe = np.tile(pe, (num_nodes, 1))

    return node_pe
