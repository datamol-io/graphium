import pandas as pd
import ast
from functools import partial
import numpy as np
from typing import List


def np_repr_to_np(string: str):  # TODO: this needs to be removed.
    """Converts a string in numpy reprsentation
    (output of print(array)) to a numpy array."""
    if "." in string:
        dtype = float
    else:
        dtype = int

    translation = str.maketrans({c: None for c in "[]\n"})
    clean_string = string.translate(translation)

    char_list = clean_string.split(" ")
    clean_list = [c for c in char_list if c != ""]
    return np.fromiter(clean_list, dtype=dtype)


def extract_labels(df: pd.DataFrame, task_level: str, label_cols: List[str]):
    """Extracts labels in label_cols from dataframe df for a given task_level.
    Returns a list of numpy arrays converted to the correct shape. Multiple
    targets are concatenated for each graph.
    """

    def unpack(graph_data, base_type: str):
        if isinstance(graph_data, str):
            if base_type == "list":
                graph_data_list = ast.literal_eval(graph_data)
                return np.array(graph_data_list)
            elif base_type == "np":
                return np_repr_to_np(graph_data)
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
        base_type = None
        if task_level != "graph":
            name: str = data.name
            if name.endswith("np"):
                base_type = "np"
            elif name.endswith("list"):
                base_type = "list"
            else:
                raise ValueError(f"Expected {name} to indicate np or list.")
        return data.apply(partial(unpack, base_type=base_type))

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
