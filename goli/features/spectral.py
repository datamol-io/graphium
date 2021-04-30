from typing import Optional, Tuple

import scipy as sp
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from scipy.sparse import csr_matrix, diags, issparse, spmatrix
import numpy as np
import torch
import networkx as nx

from goli.utils.tensor import is_dtype_torch_tensor, is_dtype_numpy_array


def compute_laplacian_positional_eigvecs(
    adj: Tuple[np.ndarray, spmatrix],
    num_pos: int,
    disconnected_comp: bool = True,
    normalization: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:

    # Sparsify the adjacency patrix
    if issparse(adj):
        adj = adj.astype(np.float64)
    else:
        adj = csr_matrix(adj, dtype=np.float64)

    # Compute tha Laplacian, and normalize it
    D = np.array(np.sum(adj, axis=1)).flatten()
    D_mat = diags(D)
    L = -adj + D_mat
    L_norm = normalize_matrix(L, degree_vector=D, normalization=normalization)

    if disconnected_comp:
        # Get the list of connected components
        components = list(nx.connected_components(nx.from_scipy_sparse_matrix(adj)))
        eigvals_tile = np.zeros((L_norm.shape[0], num_pos), dtype=np.float64)
        eigvecs = np.zeros_like(eigvals_tile)

        # Compute the eigenvectors for each connected component, and stack them together
        for component in components:
            comp = list(component)
            this_L = L_norm[comp][:, comp]
            this_eigvals, this_eigvecs = _get_positional_eigvecs(this_L, num_pos=num_pos)
            eigvecs[comp, :] = this_eigvecs
            eigvals_tile[comp, :] = this_eigvals
    else:
        eigvals, eigvecs = _get_positional_eigvecs(L, num_pos=num_pos)
        eigvals_tile = np.tile(eigvals, (L_norm.shape[0], 1))

    # Eigenvalues previously set to infinite are now set to 0
    eigvals_tile[np.isinf(eigvals_tile)] = 0

    return eigvals_tile, eigvecs


def _get_positional_eigvecs(matrix, num_pos: int):

    mat_len = matrix.shape[0]
    if num_pos < mat_len - 1:  # Compute the k-lowest eigenvectors
        eigvals, eigvecs = eigs(matrix, k=num_pos, which="SR", tol=0)

    else:  # Compute all eigenvectors

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
    eigvecs = eigvecs / (np.sqrt(np.sum(eigvecs ** 2, axis=0, keepdims=True)) + 1e-8)

    return eigvals, eigvecs


def normalize_matrix(matrix, degree_vector=None, normalization: str = None):
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
            degree_inv = np.expand_dims(degree_vector ** -0.5, axis=1)
            degree_inv[np.isinf(degree_inv)] = 0
        elif is_dtype_torch_tensor(matrix.dtype):
            degree_inv = torch.unsqueeze(degree_vector ** -0.5, dim=1)
            degree_inv[torch.isinf(degree_inv)] = 0

    # Compute the normalized matrix
    if (normalization is None) or (normalization.lower() == "none"):
        pass
    elif normalization.lower() == "sym":
        matrix = degree_inv * matrix * degree_inv.T
    elif normalization.lower() == "inv":
        matrix = (degree_inv ** 2) * matrix
    else:
        raise ValueError(
            f'`normalization` should be `None`, `"None"`, `"sym"` or `"inv"`, but `{normalization}` was provided'
        )

    return matrix
