from typing import Optional, Tuple

from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eigh, eig
from scipy.sparse import csr_matrix, diags, issparse, spmatrix
import numpy as np
import torch

from goli.utils.tensor import is_dtype_torch_tensor, is_dtype_numpy_array


def compute_laplacian_eigenfunctions(adj, num_pos: int, disconnect: bool = True, normalization: str = "none"):
    
    # TODO: IMPLEMENT THE DISCONNECT
    
    if issparse(adj):
        adj = adj.astype(np.float64)
    else:
        adj = csr_matrix(adj, dtype=np.float64)

    D = np.array(np.sum(adj, axis=1)).flatten()
    D_mat = diags(D)
    L = -adj + D_mat
    
    L_norm = normalize_matrix(L, degree_vector=D, normalization=normalization)

    # TODO: FIX THE PROBLEM WITH THE WHICH FUNCTION
    eigvals, eigvecs = compute_eigenfunctions(matrix=L_norm, degree_vector=D, normalization=normalization, k=num_pos)

    return eigvals, eigvecs



def compute_eigenfunctions(
    matrix, degree_vector=None,
    normalization:str=None, k:int=None, 
    **eig_kwargs) -> Tuple:
    r"""
    Compute the eigenvalues and eigenvectors of a given symmetric matrix.

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
        eigvals: torch.tensor(k or N) or np.ndarray(k or N)
            The eigenvalues of the input matrix.

        eigvecs: torch.tensor(N, k or N) or np.ndarray(N, k or N)
            The eigenvectors associated to ``self.eigvals``, with the ii-th column [:, ii] being the
            eigenvector associated to the ii-th element of ``self.eigvals``.

    Warning
    -------------

        If the matrix is not symmetric (e.g. using the 'inv' normalization),
        then complex values are ignored.

    """

    # Normalize the matrix
    matrix = normalize_matrix(matrix, degree_vector=degree_vector, normalization=normalization)
    if isinstance(matrix, (np.ndarray, spmatrix)):
        eigvals, eigvecs = compute_eigenfunctions_numpy(
            matrix, degree_vector=degree_vector, normalization=normalization, k=k, **eig_kwargs)
    else:
        eigvals, eigvecs = compute_eigenfunctions_torch(
            matrix, degree_vector=degree_vector, normalization=normalization, k=k, **eig_kwargs)

    return eigvals, eigvecs



def compute_eigenfunctions_numpy(
    matrix, degree_vector=None, normalization: Optional[str] = None, k: Optional[int] = None, **eig_kwargs
):

    if np.max(np.abs(matrix - matrix.T)) < 1e-8:
        # Compute the eigenvalues and eigenvectors
        if (k is None) or (k >= matrix.shape[0]):
            eigvals, eigvecs = eigh(matrix.toarray(), **eig_kwargs)
        else:
            k = min(k, matrix.shape[0])
            eigvals, eigvecs = eigsh(matrix, k=k, **eig_kwargs)

    else:
        # Compute the eigenvalues and eigenvectors and ignore imaginary part
        if (k is None) or (k >= matrix.shape[0]):
            eigvals, eigvecs = eig(matrix.toarray(), **eig_kwargs)
        else:
            k = min(k, matrix.shape[0])
            eigvals, eigvecs = eigs(matrix, k=k, **eig_kwargs)

        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

    return eigvals, eigvecs


def compute_eigenfunctions_torch(
    matrix, degree_vector=None, normalization: Optional[str] = None, k: Optional[int] = None, **eig_kwargs
):

    if torch.max((matrix - matrix.T).abs()) < 1e-8:
        # Compute the eigenvalues and eigenvectors
        which = eig_kwargs.pop("which", "SM")
        eigvals, eigvecs = torch.symeig(matrix, eigenvectors=True, **eig_kwargs)

    else:
        # Compute the eigenvalues and eigenvectors and ignore imaginary part
        which = eig_kwargs.pop("which", "SM")
        eigvals, eigvecs = torch.eig(matrix, eigenvectors=True, **eig_kwargs)
        eigvals = eigvals[:, 0]

    if k is not None:
        # Select the k-lowest or k-highest eigenvalues and vectors
        eigvals, eigvecs = sort_eigen(eigvals, eigvecs, descending=False)
        if which == "LM":
            eigvals = eigvals[-k:]
            eigvecs = eigvecs[:, -k:]
        elif which == "SM":
            eigvals = eigvals[:k]
            eigvecs = eigvecs[:, :k]
        else:
            raise ValueError(f'Unsupported `which={which}`. Only "SM" and "LM" are supported')

    return eigvals, eigvecs


def sort_eigen(eigvals, *eig_arrays, descending: bool = False) -> Tuple[np.ndarray]:
    r"""
    Sort the eigenvalues and other

    Parameters
    ---------------

        eigvals: torch.tensor(k or N) or np.ndarray(k or N)
            An array containing the eigenvalues.

        eig_arrays: torch.tensor(Any, N) or np.ndarray(Any, N)
            Other arrays that need to be sorted according to the eigenvalues,
            e.g. the eigenvectors

        descending: bool, Default=False
            Whether to sort in descending order of eigvals

    Returns
    -----------
        eigvals: torch.tensor(k or N) or np.ndarray(k or N)
            The sorted eigenvalues

        eig_arrays: torch.tensor(Any, N) or np.ndarray(Any, N)
            The array with columns sorted according to eigvals

    """

    if descending is None:
        pass
    else:
        # Sort the eigenvalues, depending on whether it is a numpy or torch type
        if is_dtype_numpy_array(eigvals.dtype):
            idx_sorted = np.argsort(eigvals)
        elif is_dtype_torch_tensor(eigvals.dtype):
            idx_sorted = torch.argsort(eigvals)

        # Sort the indexes
        if descending is True:
            idx_sorted = idx_sorted[::-1]
        elif descending is not False:
            raise ValueError(f"`descending` must be `None`, `True` or `False`. Provided `{descending}`")

        # Sort the eigenvalues and eigenvectors
        eigvals = eigvals[idx_sorted]
        arr_list = []
        for arr in list(eig_arrays):
            arr_list.append(arr[:, idx_sorted])

    return (eigvals, *arr_list)


def normalize_matrix(
            matrix, degree_vector=None, normalization:str=None):
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
        if not ((normalization is None) or (normalization.lower()=='none')):
            raise ValueError('`degree_vector` cannot be `None` if `normalization` is not `None`')
    else:
        if is_dtype_numpy_array(matrix.dtype):
            degree_inv = np.expand_dims(degree_vector ** -.5, axis=1)
        elif is_dtype_torch_tensor(matrix.dtype):
            degree_inv = torch.unsqueeze(degree_vector ** -.5, dim=1)

    # Compute the normalized matrix
    if (normalization is None) or (normalization.lower()=='none'):
        pass
    elif normalization.lower()=='sym':
        matrix = degree_inv * matrix * degree_inv.T
    elif normalization.lower()=='inv':
        matrix = (degree_inv**2) * matrix
    else:
        raise ValueError(f'`normalization` should be `None`, `"None"`, `"sym"` or `"inv"`, but `{normalization}` was provided')

    return matrix

