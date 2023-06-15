from typing import Tuple, Union, Optional, Dict, Any, OrderedDict
from copy import deepcopy
import numpy as np
import torch
from scipy.sparse import spmatrix
from collections import OrderedDict as OderedDictClass

from graphium.features.spectral import compute_laplacian_pe
from graphium.features.rw import compute_rwse
from graphium.features.electrostatic import compute_electrostatic_interactions
from graphium.features.commute import compute_commute_distances
from graphium.features.graphormer import compute_graphormer_distances
from graphium.features.transfer_pos_level import transfer_pos_level


def get_all_positional_encodings(
    adj: Union[np.ndarray, spmatrix],
    num_nodes: int,
    pos_kwargs: Optional[Dict] = None,
) -> Tuple["OrderedDict[str, np.ndarray]"]:
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

    pos_kwargs = {} if pos_kwargs is None else pos_kwargs

    pe_dict = OderedDictClass()

    # Initialize cache
    cache = {}

    # Get the positional encoding for the features
    if len(pos_kwargs) > 0:
        for pos_name, this_pos_kwargs in pos_kwargs["pos_types"].items():
            this_pos_kwargs = deepcopy(this_pos_kwargs)
            pos_type = this_pos_kwargs.pop("pos_type", None)
            pos_level = this_pos_kwargs.pop("pos_level", None)
            this_pe, cache = graph_positional_encoder(
                deepcopy(adj),
                num_nodes,
                pos_type=pos_type,
                pos_level=pos_level,
                pos_kwargs=this_pos_kwargs,
                cache=cache,
            )
            if pos_level == "node":
                pe_dict.update({f"{pos_type}": this_pe})
            else:
                pe_dict.update({f"{pos_level}_{pos_type}": this_pe})

    return pe_dict


def graph_positional_encoder(
    adj: Union[np.ndarray, spmatrix],
    num_nodes: int,
    pos_type: Optional[str] = None,
    pos_level: Optional[str] = None,
    pos_kwargs: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    r"""
    Get a positional encoding that depends on the parameters.

    Parameters:
        adj [num_nodes, num_nodes]: Adjacency matrix of the graph
        num_nodes: Number of nodes in the graph
        pos_type: The type of positional encoding to use. If None, it must be provided by `pos_kwargs["pos_type"]`. Supported types are:
            - laplacian_eigvec              \
            - laplacian_eigval               \  -> cache connected comps. & eigendecomp.
            - rwse
            - electrostatic                 \
            - commute                        \  -> cache pinvL
            - graphormer
        pos_level: Positional level to output. If None, it must be provided by `pos_kwargs["pos_level"]`.
            - node
            - edge
            - nodepair
            - graph
        pos_kwargs: Extra keyword arguments for the positional encoding. Can include the keys pos_type and pos_level.
        cache: Dictionary of cached objects

    Returns:
        pe: Positional or structural encoding
        cache: Updated dictionary of cached objects
    """

    pos_kwargs = deepcopy(pos_kwargs)
    if pos_kwargs is None:
        pos_kwargs = {}
    if cache is None:
        cache = {}

    # Get the positional type
    pos_type2 = pos_kwargs.pop("pos_type", None)
    if pos_type is None:
        pos_type = pos_type2
    if pos_type2 is not None:
        assert (
            pos_type == pos_type2
        ), f"The positional type must be the same in `pos_type` and `pos_kwargs['pos_type']`. Provided: {pos_type} and {pos_type2}"
    assert pos_type is not None, "Either `pos_type` or `pos_kwargs['pos_type']` must be provided."

    # Get the positional level
    pos_level2 = pos_kwargs.pop("pos_level", None)
    if pos_level is None:
        pos_level = pos_level2
    if pos_level2 is not None:
        assert (
            pos_level == pos_level2
        ), f"The positional level must be the same in `pos_level` and `pos_kwargs['pos_level']`. Provided: {pos_level} and {pos_level2}"
    assert pos_level is not None, "Either `pos_level` or `pos_kwargs['pos_level']` must be provided."

    # Convert to numpy array
    if isinstance(adj, torch.sparse.Tensor):
        adj = adj.to_dense().numpy()
    elif isinstance(adj, torch.Tensor):
        adj = adj.numpy()
    adj = adj.astype(np.float64)

    # Calculate positional encoding
    if pos_type == "laplacian_eigvec":
        _, pe, base_level, cache = compute_laplacian_pe(adj, cache=cache, **pos_kwargs)

    elif pos_type == "laplacian_eigval":
        pe, _, base_level, cache = compute_laplacian_pe(adj, cache=cache, **pos_kwargs)

    elif pos_type == "rw_return_probs":
        pe, base_level, cache = compute_rwse(
            adj.astype(np.float32), num_nodes=num_nodes, cache=cache, pos_type=pos_type, **pos_kwargs
        )

    elif pos_type == "rw_transition_probs":
        pe, base_level, cache = compute_rwse(
            adj.astype(np.float32), num_nodes=num_nodes, cache=cache, pos_type=pos_type, **pos_kwargs
        )

    elif pos_type == "electrostatic":
        pe, base_level, cache = compute_electrostatic_interactions(adj, cache, **pos_kwargs)

    elif pos_type == "commute":
        pe, base_level, cache = compute_commute_distances(adj, num_nodes, cache, **pos_kwargs)

    elif pos_type == "graphormer":
        pe, base_level, cache = compute_graphormer_distances(adj, num_nodes, cache, **pos_kwargs)

    else:
        raise ValueError(f"Unknown `pos_type`: {pos_type}")

    # Convert to float32 and Convert between different pos levels
    if isinstance(pe, (list, tuple)):
        pe = [this_pe.astype(np.float32) for this_pe in pe]
        pe = [transfer_pos_level(this_pe, base_level, pos_level, adj, num_nodes, cache) for this_pe in pe]
    else:
        pe = np.real(pe).astype(np.float32)
        pe = transfer_pos_level(pe, base_level, pos_level, adj, num_nodes, cache)

    return pe, cache
