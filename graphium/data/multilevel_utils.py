"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


import pandas as pd
import ast
import numpy as np
from typing import List, Optional, Tuple, Iterable
import itertools
import math

from graphium.utils.enums import TaskLevel


def extract_labels(df: pd.DataFrame, task_level: TaskLevel, label_cols: List[str]):
    """Extracts labels in label_cols from dataframe df for a given task_level.
    Returns a list of numpy arrays converted to the correct shape. Multiple
    targets are concatenated for each graph.
    """

    def unpack(graph_data):
        graph_data = pd.to_numeric(graph_data, errors="coerce")
        if isinstance(graph_data, str):
            graph_data_list = ast.literal_eval(graph_data)
            return np.array(graph_data_list)
        elif isinstance(graph_data, (int, float)):
            return np.array([graph_data])
        elif isinstance(graph_data, list):
            return np.array(graph_data)
        elif isinstance(graph_data, np.ndarray):
            if len(graph_data.shape) == 0:
                graph_data = np.expand_dims(graph_data, 0)
            if graph_data.shape[0] == 0:
                graph_data = np.array([np.nan])
                # TODO: Warning
            return graph_data
        else:
            raise ValueError(
                f"Graph data should be one of str, float, int, list, np.ndarray, got {type(graph_data)}"
            )

    def unpack_column(data: pd.Series):
        return data.apply(unpack)

    def merge_columns(data: pd.Series):
        data = data.to_list()
        data = [np.array([np.nan]) if not isinstance(d, np.ndarray) and math.isnan(d) else d for d in data]
        padded_data = itertools.zip_longest(*data, fillvalue=np.nan)
        data = np.stack(list(padded_data), 1).T
        return data

    unpacked_df: pd.DataFrame = df[label_cols].apply(unpack_column)
    output = unpacked_df.apply(merge_columns, axis="columns").to_list()

    if task_level == task_level.GRAPH:
        return np.concatenate(output)
    return output


def get_canonical_ranks_pair(
    all_canonical_ranks: List[List[int]], all_task_levels: List[TaskLevel], unique_ids_inv: Iterable[int]
) -> List[Optional[Tuple[List[int], List[int]]]]:
    """
    This function takes a list of canonical ranks and task levels and returns a list of canonical ranks pairs.
    The canonical ranks pairs are used to check if the featurized ranks are the same as the canonical ranks.
    If the featurized ranks are different, we need to store them.

    Parameters:
        all_canonical_ranks: a list of canonical ranks for all molecules.
            The ranks are a list of integers based on `rdkit.Chem.rdmolfiles.CanonicalRankAtoms`
        all_task_levels: a list of task levels
        unique_ids_inv: a list of indices mapping each molecule to another identical molecule, or itself.
            If the molecule is unique, the index is the same as the index of the molecule in the dataset.
            If the molecule is not unique, the index is the index of the molecule that will be featurized.

    Returns:
        canonical_ranks_pair: a list of canonical ranks pairs
    """

    if {len(all_canonical_ranks)} != {len(all_canonical_ranks), len(all_task_levels), len(unique_ids_inv)}:
        raise ValueError(
            f"all_canonical_ranks, all_task_levels, and unique_ids_inv must have the same length, got {len(all_canonical_ranks)}, {len(all_task_levels)}, {len(unique_ids_inv)}"
        )

    canonical_ranks_pair = [None] * len(all_canonical_ranks)
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
                canonical_ranks_pair[ii] = (featurized_rank, this_rank)
    return canonical_ranks_pair
