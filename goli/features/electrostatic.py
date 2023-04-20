from typing import Union

import numpy as np

from scipy.linalg import pinv
from scipy.sparse import spmatrix, issparse


def compute_electrostatic_interactions(adj: Union[np.ndarray, spmatrix]) -> np.ndarray:
    """
    Compute electrostatic interaction of node pairs.

    Parameters:
        adj [num_nodes, num_nodes]: Adjacency matrix
    Returns:
        electrostatic [num_nodes, num_nodes]: 2D array with electrostatic interactions of node pairs
    """

    if issparse(adj):
        adj = adj.toarray()

    L = np.diagflat(np.sum(adj, axis=1)) - adj
    pinvL = pinv(L)
    electrostatic = pinvL - np.diag(pinvL) # This means that the "ground" is set to any given atom

    return electrostatic