"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs and Recursion Pharmaceuticals are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


from typing import Type, List, Dict, Union, Any, Callable, Optional, Tuple, Iterable

import os
import datamol as dm
from graphium.utils.enums import TaskLevel


def smiles_to_unique_mol_id_and_rank(smiles: str) -> Tuple[Optional[str], List[int]]:
    """
    Convert a smiles to a unique MD5 Hash ID. Returns None if featurization fails.
    Parameters:
        smiles: A smiles string to be converted to a unique ID
    Returns:
        mol_id: a string unique ID
    """
    canonical_rank = []
    mol_id = ""
    try:
        mol = dm.to_mol(mol=smiles)
        mol_id = dm.unique_id(mol)
        canonical_rank = dm.canonical_rank(mol)
    except:
        mol_id = ""
    if mol_id is None:
        mol_id = ""
    if canonical_rank is None:
        canonical_rank = []
    return mol_id, canonical_rank


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


def smiles_to_unique_mol_ids_and_rank(
    smiles: Iterable[str],
    n_jobs=-1,
    featurization_batch_size=1000,
    backend="loky",
    progress=True,
    progress_desc="mols to ids",
) -> Tuple[List[Optional[str]], List[List[int]]]:
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

    unique_mol_ids_and_ranks = dm.parallelized_with_batches(
        BatchingSmilesTransform(smiles_to_unique_mol_id_and_rank),
        smiles,
        batch_size=batch_size,
        progress=progress,
        n_jobs=n_jobs,
        backend=backend,
        tqdm_kwargs={"desc": f"{progress_desc}, batch={batch_size}"},
    )
    unique_mol_ids, canonical_ranks = zip(*unique_mol_ids_and_ranks)

    return unique_mol_ids, canonical_ranks


def get_canonical_ranks_pair(
    all_canonical_ranks: List[List[int]], all_task_levels: List[TaskLevel], unique_ids_inv: Iterable[int]
) -> List[Optional[Tuple[List[int], List[int]]]]:
    """
    This function takes a list of canonical ranks and task levels and returns a list of canonical ranks pairs.
    The canonical ranks pairs are used to check if the featurized ranks are the same as the canonical ranks.
    If the featurized ranks are different, we need to store them.

    Parameters:
        all_canonical_ranks: a list of canonical ranks. The ranks are a list of integers based on `rdkit.Chem.rdmolfiles.CanonicalRankAtoms`
        all_task_levels: a list of task levels
        unique_ids_inv: a list of indices

    Returns:
        canonical_ranks_pair: a list of canonical ranks pairs
    """

    if {len(all_canonical_ranks)} != set(len(all_canonical_ranks), len(all_task_levels), len(unique_ids_inv)):
        raise ValueError(
            f"all_canonical_ranks, all_task_levels, and unique_ids_inv must have the same length, got {len(all_canonical_ranks)}, {len(all_task_levels)}, {len(unique_ids_inv)}"
        )

    canonical_ranks_pair = []
    for ii, inv_idx in enumerate(unique_ids_inv):
        task_level_need_rank = all_task_levels[ii] != TaskLevel.GRAPH
        if task_level_need_rank:
            featurized_rank = all_canonical_ranks[inv_idx]
            this_rank = all_canonical_ranks[ii]

            # If the ranks are different, we need to store them
            if (
                (featurized_rank is not None)
                and (this_rank is not None)
                and (len(featurized_rank) > 0)
                and (len(this_rank) > 0)
                and (featurized_rank != this_rank)
            ):
                canonical_ranks_pair.append((featurized_rank, this_rank))
        else:
            canonical_ranks_pair.append(None)
    return canonical_ranks_pair
