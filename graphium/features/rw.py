from typing import Tuple, Union, Optional, List, Dict, Any, Iterable

from scipy.sparse import issparse, spmatrix, coo_matrix
import numpy as np
import torch

from torch_geometric.utils import to_dense_adj, from_scipy_sparse_matrix
from torch_scatter import scatter_add
from torch_geometric.utils.num_nodes import maybe_num_nodes


def compute_rwse(
    adj: Union[np.ndarray, spmatrix],
    ksteps: Union[int, List[int]],
    num_nodes: int,
    cache: Dict[str, Any],
    pos_type: str = "rw_return_probs" or "rw_transition_probs",
    space_dim: int = 0,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Compute Random Walk Spectral Embedding (RWSE) for given list of K steps.

    Parameters:
        adj [num_nodes, num_nodes]: Adjacency matrix
        ksteps: List of numbers of steps for the random walks. If int, a list is generated from 1 to ksteps.
        num_nodes: Number of nodes in the graph
        cache: Dictionary of cached objects
        pos_type: Desired output
        space_dim: Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.
    Returns:
        Two possible outputs:
            rw_return_probs [num_nodes, len(ksteps)]: Random-Walk k-step landing probabilities
            rw_transition_probs [num_nodes, num_nodes, len(ksteps)]:  Random-Walk k-step transition probabilities
        base_level: Indicator of the output pos_level (node, edge, nodepair, graph) -> here either node or nodepair
        cache: Updated dictionary of cached objects
    """

    base_level = "node" if pos_type == "rw_return_probs" else "nodepair"

    # Manually handles edge case of 1 atom molecules here
    if not isinstance(ksteps, Iterable):
        ksteps = list(range(1, ksteps + 1))
    if num_nodes == 1:
        if pos_type == "rw_return_probs":
            return np.ones((1, len(ksteps))), base_level, cache
        else:
            return np.ones((1, 1, len(ksteps))), base_level, cache

    # Get the edge indices from the adjacency matrix
    if not issparse(adj):
        if "coo_adj" in cache:
            adj = cache["coo_adj"]
        elif "csr_adj" in cache:
            adj = cache["csr_adj"]
        else:
            adj = coo_matrix(adj, dtype=np.float64)
            cache["coo_adj"] = adj

    edge_index, edge_weight = from_scipy_sparse_matrix(adj)

    # Compute the random-walk transition probabilities
    if "ksteps" in cache:
        cached_k = cache["ksteps"]
        missing_k = [k for k in ksteps if k not in cached_k]
        if missing_k == []:
            pass
        elif min(missing_k) < min(cached_k):
            Pk_dict = get_Pks(missing_k, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
            cache["ksteps"] = sorted(missing_k + cache["ksteps"])
            for k in missing_k:
                cache["Pk"][k] = Pk_dict[k]
        else:
            start_k = min([max(cached_k), min(missing_k)])
            start_Pk = cache["Pk"][start_k]
            Pk_dict = get_Pks(
                missing_k,
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=num_nodes,
                start_Pk=start_Pk,
                start_k=start_k,
            )
            cache["ksteps"] = sorted(cache["ksteps"] + missing_k)
            for k in missing_k:
                cache["Pk"][k] = Pk_dict[k]
    else:
        Pk_dict = get_Pks(ksteps, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)

        cache["ksteps"] = list(Pk_dict.keys())
        cache["Pk"] = Pk_dict

    pe_list = []
    if pos_type == "rw_return_probs":
        for k in ksteps:
            pe_list.append(torch.diagonal(cache["Pk"][k], dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
    else:
        for k in ksteps:
            pe_list.append(cache["Pk"][k])

    pe = torch.stack(pe_list, dim=-1).numpy()

    return pe, base_level, cache


def get_Pks(
    ksteps: List[int],
    edge_index: Tuple[torch.Tensor, torch.Tensor],
    edge_weight: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    start_Pk: Optional[torch.Tensor] = None,
    start_k: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Compute Random Walk landing probabilities for given list of K steps.

    Parameters:
        ksteps: List of numbers of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: Edge weights
        num_nodes: Number of nodes in the graph

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    src = edge_index[0]
    deg = scatter_add(edge_weight, src, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv).float() @ to_dense_adj(
            edge_index, max_num_nodes=num_nodes
        )  # 1 x (Num nodes) x (Num nodes)

    if start_Pk is not None:
        Pk = start_Pk @ P.clone().detach().matrix_power(min(ksteps) - start_k)
    else:
        Pk = P.clone().detach().matrix_power(min(ksteps))

    Pk_dict = {}
    for k in range(min(ksteps), max(ksteps) + 1):
        Pk_dict[k] = Pk.squeeze(0)
        Pk = Pk @ P

    return Pk_dict
