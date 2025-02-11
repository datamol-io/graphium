"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals, Graphcore Limited, and NVIDIA Corporation & Affiliates.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals, Graphcore Limited, and NVIDIA Corporation & Affiliates are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


import pandas as pd
import ast
import numpy as np
from typing import List
import itertools
import math


def extract_labels(df: pd.DataFrame, task_level: str, label_cols: List[str]):
    """Extracts the labels specified by label_cols from dataframe df.
    If task_level is "graph", each entry in df must be a single numeric value,
    and this function returns a single, 2D numpy array containing the data.
    If task_level is something else, each entry in df must be a numpy array,
    python list, or single numeric value, and this function returns both a 2D
    numpy array of data and a 1D numpy array of integers indicating the row
    number in the first array where each molecule's data starts, with an extra
    integer at the end that should equal the total number of rows in the first
    array.  The first array can have type float16, float32, or float64,
    depending on the largest precision of input data, and arrays of varying
    sizes across columns are padded with nan values, so that a single molecule
    occupies a fixed number of rows and len(label_cols) columns.
    """

    num_rows = df.shape[0]
    num_cols = len(label_cols)

    if task_level == "graph":
        output = np.empty((num_rows, num_cols), dtype=np.float64)

        for col_index, col in enumerate(label_cols):
            for i, v in enumerate(df[col]):
                if isinstance(v, float):
                    output[i, col_index] = v
                    continue

                v = pd.to_numeric(v, errors="coerce")

                if isinstance(v, (int, float)):
                    output[i, col_index] = v

                else:
                    raise ValueError(f"Graph data should be one of float or int, got {type(v)}")

        return output, None

    # First, find the max length of each row (likely the number of nodes or edges)
    # +1 is for the cumulative sum below
    begin_offsets = np.zeros((num_rows + 1,), dtype=np.int64)
    max_type = np.float16
    for col in label_cols:
        for i, v in enumerate(df[col]):
            if not isinstance(v, np.ndarray) and not isinstance(v, (int, float, list)):
                v = pd.to_numeric(v, errors="coerce")
            length = 0
            if isinstance(v, np.ndarray):
                if len(v.shape) == 1:
                    length = v.shape[0]
                elif len(v.shape) == 0:
                    length = 0
                else:
                    raise ValueError(
                        f"Graph data should be 1D np.ndarray, got ndarray with {len(v.shape)} dimensions"
                    )
                dtype = v.dtype
                if dtype == np.float64:
                    max_type = np.float64
                elif dtype == np.float32 and max_type == np.float16:
                    max_type = np.float32
            elif isinstance(v, (int, float)):
                length = 1
                max_type = np.float64
            elif isinstance(v, list):
                length = len(v)
                max_type = np.float64
            else:
                raise ValueError(f"Graph data should be one of float, int, list, np.ndarray, got {type(v)}")
            # The +1 is so that the cumulative sum below gives the beginning offsets
            begin_offsets[i + 1] = max(begin_offsets[i + 1], length)

    begin_offsets = np.cumsum(begin_offsets)
    full_num_rows = begin_offsets[-1]

    output = np.empty((full_num_rows, num_cols), dtype=max_type)

    # Now, fill in the values
    for col_index, col in enumerate(label_cols):
        for i, v in enumerate(df[col]):
            full_row = begin_offsets[i]
            end_row = begin_offsets[i + 1]

            if not isinstance(v, np.ndarray):
                v = pd.to_numeric(v, errors="coerce")

            if isinstance(v, np.ndarray):
                length = v.shape[0] if len(v.shape) == 1 else 0
                for j in range(length):
                    output[full_row + j, col_index] = v[j]

            elif isinstance(v, (int, float)):
                length = 1
                output[full_row, col_index] = v

            elif isinstance(v, list):
                length = len(v)
                for j in range(length):
                    output[full_row + j, col_index] = v[j]

            else:
                raise ValueError(f"Graph data should be one of float, int, list, np.ndarray, got {type(v)}")

            # Fill the rest of the rows in the column with nan
            if full_row + length != end_row:
                for row in range(full_row + length, end_row):
                    output[row, col_index] = np.nan

    return output, begin_offsets
