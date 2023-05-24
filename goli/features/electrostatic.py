from typing import Tuple, Union, Dict, Any

import numpy as np

from scipy.linalg import pinv
from scipy.sparse import spmatrix, issparse


def compute_electrostatic_interactions(
    adj: Union[np.ndarray, spmatrix], cache: Dict[str, Any]
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Compute electrostatic interaction of nodepairs.

    Parameters:
        adj [num_nodes, num_nodes]: Adjacency matrix
        cache: Dictionary of cached objects
    Returns:
        electrostatic [num_nodes, num_nodes]: 2D array with electrostatic interactions of node nodepairs
        base_level: Indicator of the output pos_level (node, edge, nodepair, graph) -> here nodepair
        cache: Updated dictionary of cached objects
    """

    base_level = "nodepair"

    if "electrostatic" in cache:
        electrostatic = cache["electrostatic"]

    else:
        if "pinvL" in cache:
            pinvL = cache["pinvL"]

        else:
            if issparse(adj):
                adj = adj.toarray()

            L = np.diagflat(np.sum(adj, axis=1)) - adj
            pinvL = pinv(L)
            cache["pinvL"] = pinvL

        electrostatic = pinvL - np.diag(pinvL)  # This means that the "ground" is set to any given atom
        cache["electrostatic"] = electrostatic

    return electrostatic, base_level, cache
