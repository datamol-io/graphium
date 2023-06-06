from typing import Tuple, Union, Dict, Any
from scipy.linalg import eig
from scipy.sparse import csr_matrix, diags, issparse, spmatrix
import numpy as np
import torch
import networkx as nx

from graphium.utils.tensor import is_dtype_torch_tensor, is_dtype_numpy_array


def compute_laplacian_pe(
    adj: Union[np.ndarray, spmatrix],
    num_pos: int,
    cache: Dict[str, Any],
    disconnected_comp: bool = True,
    normalization: str = "none",
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    r"""
    Compute the Laplacian eigenvalues and eigenvectors of the Laplacian of the graph.

    Parameters:
        adj [num_nodes, num_nodes]: Adjacency matrix of the graph
        num_pos: Number of Laplacian eigenvectors to compute
        cache: Dictionary of cached objects
        disconnected_comp: Whether to compute the eigenvectors for each connected component
        normalization: Normalization to apply to the Laplacian

    Returns:
        Two possible outputs:
            eigvals [num_nodes, num_pos]: Eigenvalues of the Laplacian repeated for each node.
                This repetition is necessary in case of disconnected components, where
                the eigenvalues of the Laplacian are not the same for each node.
            eigvecs [num_nodes, num_pos]: Eigenvectors of the Laplacian
        base_level: Indicator of the output pos_level (node, edge, nodepair, graph) -> here node
        cache: Updated dictionary of cached objects
    """

    base_level = "node"

    # Sparsify the adjacency patrix
    if not issparse(adj):
        if "csr_adj" not in cache:
            adj = csr_matrix(adj, dtype=np.float64)
            cache["csr_adj"] = adj
        else:
            adj = cache["csr_adj"]

    # Compute the Laplacian, and normalize it
    if f"L_{normalization}_sp" not in cache:
        D = np.array(np.sum(adj, axis=1)).flatten()
        D_mat = diags(D)
        L = -adj + D_mat
        L_norm = normalize_matrix(L, degree_vector=D, normalization=normalization)
        cache[f"L_{normalization}_sp"] = L_norm
    else:
        L_norm = cache[f"L_{normalization}_sp"]

    components = []

    if disconnected_comp:
        if "components" not in cache:
            # Get the list of connected components
            components = list(nx.connected_components(nx.from_scipy_sparse_array(adj)))
            cache["components"] = components

        else:
            components = cache["components"]

    # Compute the eigenvectors for each connected component, and stack them together
    if len(components) > 1:
        if "lap_eig_comp" not in cache:
            eigvals = np.zeros((adj.shape[0], num_pos), dtype=np.complex64)
            eigvecs = np.zeros((adj.shape[0], num_pos), dtype=np.complex64)
            for component in components:
                comp = list(component)
                this_L = L_norm[comp][:, comp]
                this_eigvals, this_eigvecs = _get_positional_eigvecs(this_L, num_pos=num_pos)

                # Eigenvalues previously set to infinity are now set to 0
                # Any NaN in the eigvals or eigvecs will be set to 0
                this_eigvecs[~np.isfinite(this_eigvecs)] = 0.0
                this_eigvals[~np.isfinite(this_eigvals)] = 0.0

                eigvals[comp, :] = np.expand_dims(this_eigvals, axis=0)
                eigvecs[comp, :] = this_eigvecs
            cache["lap_eig_comp"] = (eigvals, eigvecs)

        else:
            eigvals, eigvecs = cache["lap_eig_comp"]

    else:
        if "lap_eig" not in cache:
            eigvals, eigvecs = _get_positional_eigvecs(L, num_pos=num_pos)

            # Eigenvalues previously set to infinity are now set to 0
            # Any NaN in the eigvals or eigvecs will be set to 0
            eigvecs[~np.isfinite(eigvecs)] = 0.0
            eigvals[~np.isfinite(eigvals)] = 0.0
            eigvals = np.repeat(np.expand_dims(eigvals, axis=0), adj.shape[0], axis=0)

            cache["lap_eig"] = (eigvals, eigvecs)

        else:
            eigvals, eigvecs = cache["lap_eig"]

    return eigvals, eigvecs, base_level, cache


def _get_positional_eigvecs(
    matrix: Union[np.ndarray, spmatrix],
    num_pos: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    compute the eigenvalues and eigenvectors of a matrix
    Parameters:
        matrix: Matrix to compute the eigenvalues and eigenvectors of
        num_pos: Number of eigenvalues and eigenvectors to compute
    Returns:
        eigvals: Eigenvalues of the matrix
        eigvecs: Eigenvectors of the matrix
    """
    mat_len = matrix.shape[0]
    eigvals, eigvecs = eig(matrix.todense())

    # Pad with non-sense eigenvectors if required
    if num_pos > mat_len:
        temp_EigVal = np.ones(num_pos - mat_len, dtype=np.float64) + float("inf")
        temp_EigVec = np.zeros((mat_len, num_pos - mat_len), dtype=np.float64)
        eigvals = np.concatenate([eigvals, temp_EigVal], axis=0)
        eigvecs = np.concatenate([eigvecs, temp_EigVec], axis=1)

    # Sort and keep only the first `num_pos` elements
    sort_idx = eigvals.argsort()
    eigvals = eigvals[sort_idx]
    eigvals = eigvals[:num_pos]
    eigvecs = eigvecs[:, sort_idx]
    eigvecs = eigvecs[:, :num_pos]

    # Normalize the eigvecs
    eigvecs = eigvecs / np.maximum(np.sqrt(np.sum(eigvecs**2, axis=0, keepdims=True)), 1e-4)

    return eigvals, eigvecs


def normalize_matrix(
    matrix: Union[np.ndarray, spmatrix],
    degree_vector=None,
    normalization: str = None,
) -> Union[np.ndarray, spmatrix]:
    r"""
    Normalize a given matrix using its degree vector

    Parameters
    ---------------

        matrix: torch.tensor(N, N) or scipy.sparse.spmatrix(N, N)
            A square matrix representing either an Adjacency matrix or a Laplacian.

        degree_vector: torch.tensor(N) or np.ndarray(N) or None
            A vector representing the degree of ``matrix``.
            ``None`` is only accepted if ``normalization==None``

        normalization: str or None, Default='none'
            Normalization to use on the eig_matrix

            - 'none' or ``None``: no normalization

            - 'sym': Symmetric normalization ``D^-0.5 L D^-0.5``

            - 'inv': Inverse normalization ``D^-1 L``

    Returns
    -----------
        matrix: torch.tensor(N, N) or scipy.sparse.spmatrix(N, N)
            The normalized matrix

    """

    # Transform the degree vector into a matrix
    if degree_vector is None:
        if not ((normalization is None) or (normalization.lower() == "none")):
            raise ValueError("`degree_vector` cannot be `None` if `normalization` is not `None`")
    else:
        if is_dtype_numpy_array(matrix.dtype):
            with np.errstate(divide="ignore", invalid="ignore"):
                degree_inv = np.expand_dims(degree_vector**-0.5, axis=1)
                degree_inv[np.isinf(degree_inv)] = 0
        elif is_dtype_torch_tensor(matrix.dtype):
            degree_inv = torch.unsqueeze(degree_vector**-0.5, dim=1)
            degree_inv[torch.isinf(degree_inv)] = 0

    # Compute the normalized matrix
    if (normalization is None) or (normalization.lower() == "none"):
        pass
    elif normalization.lower() == "sym":
        matrix = degree_inv * matrix * degree_inv.T
    elif normalization.lower() == "inv":
        matrix = (degree_inv**2) * matrix
    else:
        raise ValueError(
            f'`normalization` should be `None`, `"None"`, `"sym"` or `"inv"`, but `{normalization}` was provided'
        )

    return matrix
