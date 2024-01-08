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
from typing import List
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
