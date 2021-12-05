from typing import Tuple, Union

from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from scipy.sparse import csr_matrix, diags, issparse, spmatrix
import numpy as np
import torch
import networkx as nx

from goli.utils.tensor import is_dtype_torch_tensor, is_dtype_numpy_array


def compute_laplacian_positional_eigvecs(
    adj: Union[np.ndarray, spmatrix],
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
            eigvecs[comp, :] = np.real(this_eigvecs)
            eigvals_tile[comp, :] = np.real(this_eigvals)
    else:
        eigvals, eigvecs = _get_positional_eigvecs(L, num_pos=num_pos)
        eigvals_tile = np.tile(eigvals, (L_norm.shape[0], 1))

    # Eigenvalues previously set to infinite are now set to 0
    # Any NaN in the eigvals or eigvecs will be set to 0
    eigvecs[~np.isfinite(eigvecs)] = 0.0
    eigvals_tile[~np.isfinite(eigvals_tile)] = 0.0

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
    eigvecs = eigvecs / np.maximum(np.sqrt(np.sum(eigvecs ** 2, axis=0, keepdims=True)), 1e-4)

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
            with np.errstate(divide="ignore", invalid="ignore"):
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

def compute_centroid_effective_resistances(
    adj: Union[np.ndarray, spmatrix],
    num_pos: int,
    disconnected_comp: bool = True,
    normalization: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:

    # Sparsify the adjacency patrix
    if issparse(adj):
        adj = adj.astype(np.float64)
    else:
        adj = csr_matrix(adj, dtype=np.float64)

    matrix_inv = get_graph_props(adj, shift_to_zero_diag=True)

    if disconnected_comp:
        # Get the list of connected components
        components = list(nx.connected_components(nx.from_scipy_sparse_matrix(adj)))
        electric_poetntial_tile = np.zeros((matrix_inv.shape[0], num_pos), dtype=np.float64)
        # eigvecs = np.zeros_like(eigvals_tile)

        # Compute the eigenvectors for each connected component, and stack them together
        for component in components:
            comp = list(component)
            this_L = matrix_inv[comp][:, comp]
            centroid_idx = np.argmax(np.sum(this_L, axis=1))
            electric_potential = this_L[centroid_idx, :]
            electric_potential = np.transpose(np.asarray(electric_potential.todense()))
            # eigvecs[comp, :] = this_eigvecs
            electric_poetntial_tile[comp, :] = electric_potential
    else:
        centroid_idx = np.argmax(np.sum(matrix_inv, axis=1))
        electric_poetntial_tile = matrix_inv[centroid_idx, :]


    return centroid_idx, electric_poetntial_tile


def compute_centroid_effective_resistancesv1(
    adj: Union[np.ndarray, spmatrix],
    num_pos: int,
    disconnected_comp: bool = True,
    normalization: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:

    # Sparsify the adjacency patrix
    if issparse(adj):
        adj = adj.astype(np.float64)
    else:
        adj = csr_matrix(adj, dtype=np.float64)

    matrix_inv = get_graph_props(adj,normalize_L=None,shift_to_zero_diag=False)

    if disconnected_comp:
        # Get the list of connected components
        components = list(nx.connected_components(nx.from_scipy_sparse_matrix(adj)))
        electric_poetntial_tile = np.zeros((matrix_inv.shape[0], num_pos), dtype=np.float64)
        # eigvecs = np.zeros_like(eigvals_tile)

        # Compute the eigenvectors for each connected component, and stack them together
        for component in components:
            comp = list(component)
            this_L = matrix_inv[comp][:, comp]
            eigval, eigvec = np.linalg.eig(this_L.todense())
            D_L_inv = np.diag(eigval)
            D_L_inv_sqrt = np.sqrt(abs(D_L_inv))
            Y = np.matmul(np.matmul(eigvec, D_L_inv_sqrt), eigvec.T)
            y_norm = list(np.linalg.norm(Y, ord=2, axis=1))
            min_val = np.min(y_norm)
            centroid_idx = [i for i, x in enumerate(y_norm) if x == min_val]
            effective_resistance = []
            for i in range(0, Y.shape[0]):
                eff = np.linalg.norm(Y[i, :] - Y[centroid_idx, :])
                effective_resistance.append(eff)
            # electric_potential = this_L[centroid_idx, :]
            # electric_potential = np.transpose(np.asarray(electric_potential.todense()))
            # eigvecs[comp, :] = this_eigvecs
            electric_poetntial_tile[comp, :] = np.asarray(effective_resistance)[:,np.newaxis]
    else:
        centroid_idx = np.argmax(np.sum(matrix_inv, axis=1))
        electric_poetntial_tile = matrix_inv[centroid_idx, :]


    return centroid_idx, electric_poetntial_tile




def get_graph_props(A, normalize_L=None, shift_to_zero_diag=False):


    D = np.array(np.sum(A, axis=1)).flatten()
    D_mat = diags(D)
    L = -A + D_mat
    L = np.asarray(L.todense())

    # ran = range(A.shape[0])
    # D = np.zeros_like(A)
    # D[ran, ran] = np.abs(np.sum(A, axis=1) - A[ran, ran])
    # L = D - A

    if (normalize_L is None) or (normalize_L=='none') or (normalize_L == False):
        pass
    elif (normalize_L == 'inv'):
        Dinv = np.linalg.inv(D)
        L = np.matmul(Dinv, L)  # Normalized laplacian
    elif (normalize_L == 'sym'):
        Dinv = np.sqrt(np.linalg.inv(D))
        L = np.matmul(np.matmul(Dinv, L), Dinv)
    elif (normalize_L == 'abs'):
        L = np.abs(L)
    else:
        raise ValueError('unsupported normalization option')

    eigval, eigvec = np.linalg.eig(L)
    eigval = np.real(eigval)
    eigidx = np.argsort(eigval)[::-1]
    eigval = eigval[eigidx]
    eigvec = eigvec[:, eigidx]


    L_inv = np.linalg.pinv(L)

    if shift_to_zero_diag:
        L_inv_diag = L_inv[np.eye(L.shape[0])>0]
        L_inv = (L_inv - L_inv_diag[:, np.newaxis])

#     return D, L, L_inv, eigval, eigvec
    return csr_matrix(L_inv, dtype=np.float64)