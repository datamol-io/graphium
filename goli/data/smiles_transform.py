from typing import Type, List, Dict, Union, Any, Callable, Optional, Tuple, Iterable

import os
import datamol as dm


def smiles_to_unique_mol_id(smiles: str) -> Optional[str]:
    """
    Convert a smiles to a unique MD5 Hash ID. Returns None if featurization fails.
    Parameters:
        smiles: A smiles string to be converted to a unique ID
    Returns:
        mol_id: a string unique ID
    """
    try:
        mol = dm.to_mol(mol=smiles)
        mol_id = dm.unique_id(mol)
    except:
        mol_id = ""
    if mol_id is None:
        mol_id = ""
    return mol_id


def did_featurization_fail(features: Any) -> bool:
    """
    Check if a featurization failed.
    """
    return (features is None) or isinstance(features, str)


class BatchingSmilesTransform:
    """
    Class to transform a list of smiles using a transform function
    """

    def __init__(self, transform: Callable):
        """
        Parameters:
            transform: Callable function to transform a single smiles
        """
        self.transform = transform

    def __call__(self, smiles_list: Iterable[str]) -> Any:
        """
        Function to transform a list of smiles
        """
        mol_id_list = []
        for smiles in smiles_list:
            mol_id_list.append(self.transform(smiles))
        return mol_id_list

    @staticmethod
    def parse_batch_size(numel: int, desired_batch_size: int, n_jobs: int) -> int:
        """
        Function to parse the batch size.
        The batch size is limited by the number of elements divided by the number of jobs.
        """
        assert ((n_jobs >= 0) or (n_jobs == -1)) and isinstance(
            n_jobs, int
        ), f"n_jobs must be a positive integer or -1, got {n_jobs}"
        assert (
            isinstance(desired_batch_size, int) and desired_batch_size >= 0
        ), f"desired_batch_size must be a positive integer, got {desired_batch_size}"

        if n_jobs == -1:
            n_jobs = os.cpu_count()
        if (n_jobs == 0) or (n_jobs == 1):
            batch_size = 1
        else:
            batch_size = min(desired_batch_size, numel // n_jobs)
        batch_size = max(1, batch_size)
        return batch_size


def smiles_to_unique_mol_ids(
    smiles: Iterable[str],
    n_jobs=-1,
    featurization_batch_size=1000,
    backend="loky",
    progress=True,
    progress_desc="mols to ids",
) -> List[Optional[str]]:
    """
    This function takes a list of smiles and finds the corresponding datamol unique_id
    in an element-wise fashion, returning the corresponding unique_ids.

    The ID is an MD5 hash of the non-standard InChiKey provided
    by `dm.to_inchikey_non_standard()`. It guarantees uniqueness for
    different tautomeric forms of the same molecule.

    Parameters:
        smiles: a list of smiles to be converted to mol ids
        n_jobs: number of jobs to run in parallel
        backend: Parallelization backend
        progress: Whether to display the progress bar

    Returns:
        ids: A list of MD5 hash ids
    """

    batch_size = BatchingSmilesTransform.parse_batch_size(
        numel=len(smiles), desired_batch_size=featurization_batch_size, n_jobs=n_jobs
    )

    unique_mol_ids = dm.parallelized_with_batches(
        BatchingSmilesTransform(smiles_to_unique_mol_id),
        smiles,
        batch_size=batch_size,
        progress=progress,
        n_jobs=n_jobs,
        backend=backend,
        tqdm_kwargs={"desc": f"{progress_desc}, batch={batch_size}"},
    )

    return unique_mol_ids
