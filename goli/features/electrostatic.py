from typing import Union

import numpy as np

from scipy.linalg import pinv
from scipy.sparse import spmatrix, issparse


def compute_electrostatic_interactions(
        adj: Union[np.ndarray, spmatrix],
        cache: dict
) -> np.ndarray:
    """
    Compute electrostatic interaction of node pairs.

    Parameters:
        adj (np.ndarray, [num_nodes, num_nodes]): Adjacency matrix
        cache (dict): Dictionary of cached objects
    Returns:
        electrostatic (np.ndarray, [num_nodes, num_nodes]): 2D array with electrostatic interactions of node pairs
        base_level (str): Indicator of the output pos_level (node, edge, [node]pair, graph) -> here pair
        cache (dict): Updated dictionary of cached objects
    """

    base_level = 'pair'

    if 'electrostatic' in cache:
        electrostatic = cache['electrostatic']
    
    else:      
        if 'pinvL' in cache:
            pinvL = cache['pinvL']
        
        else:
            if issparse(adj):
                adj = adj.astype(np.float64)
                adj = adj.toarray()

            L = np.diagflat(np.sum(adj, axis=1)) - adj
            pinvL = pinv(L)
            cache['pinvL'] = pinvL
    
        electrostatic = pinvL - np.diag(pinvL) # This means that the "ground" is set to any given atom
        cache['electrostatic'] = electrostatic

    return electrostatic, base_level, cache