import pandas as pd
import ast
import numpy as np
from typing import List


def extract_labels(df: pd.DataFrame, task_level: str, label_cols: List[str]):
    """Extracts labels in label_cols from dataframe df for a given task_level.
    Returns a list of numpy arrays converted to the correct shape. Multiple
    targets are concatenated for each graph.
    """

    def unpack(graph_data):
        if isinstance(graph_data, str):
            graph_data_list = ast.literal_eval(graph_data)
            return np.array(graph_data_list)
        elif isinstance(graph_data, (int, float)):
            return np.array([graph_data])
        elif isinstance(graph_data, list):
            return np.array(graph_data)
        elif isinstance(graph_data, np.ndarray):
            return graph_data
        else:
            raise ValueError(
                f"Graph data should be one of str, float, int, list, np.ndarray, got {type(graph_data)}"
            )

    def unpack_column(data: pd.Series):
        return data.apply(unpack)

    def merge_columns(data: pd.Series):
        data = data.to_list()
        if len(data) > 1:
            data = np.stack(data, 1)
        else:
            data = data[0]

        return data

    unpacked_df: pd.DataFrame = df[label_cols].apply(unpack_column)
    output = unpacked_df.apply(merge_columns, axis="columns").to_list()

    if task_level == "graph":
        return np.stack(output)
    return output
