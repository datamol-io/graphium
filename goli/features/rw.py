from typing import Tuple, Union, Optional

from scipy import sparse
from scipy.sparse import spmatrix
import numpy as np
import torch

from torch_geometric.utils import to_dense_adj, from_scipy_sparse_matrix
from torch_scatter import scatter_add
from torch_geometric.utils.num_nodes import maybe_num_nodes


def compute_rwse(adj: Union[np.ndarray, spmatrix], ksteps: int, num_nodes: int) -> np.ndarray:
    """
    Parameters:
        adj: Adjacency matrix
        ksteps: Number of steps for the random walk
        num_nodes: Number of nodes in the graph
    Returns:
        2D array with shape (num_nodes, len(ksteps)) with Random-Walk landing probs
    """

    # Manually handles edge case of 1 atom molecules here
    if num_nodes == 1:
        rw_landing = np.ones((1, ksteps))
        return rw_landing

    # Get the edge indices from the adjacency matrix
    if type(adj) is np.ndarray:
        adj = sparse.csr_matrix(adj)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)

    # Compute the random-walk landing probability
    ksteps_range = range(1, ksteps + 1)
    rw_landing = get_rw_landing_probs(
        ksteps=ksteps_range, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes
    )
    rw_landing = rw_landing.numpy()

    return rw_landing


def get_rw_landing_probs(
    ksteps: int,
    edge_index: Tuple[torch.Tensor, torch.Tensor],
    edge_weight: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    space_dim: float = 0.0,
):
    """
    Compute Random Walk landing probabilities for given list of K steps.

    Parameters:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: Edge weights
        num_nodes: Number of nodes in the graph
        space_dim: Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.
    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(
            edge_index, max_num_nodes=num_nodes
        )  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing
