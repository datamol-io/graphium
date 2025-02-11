"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals, Graphcore, and NVIDIA Corporation & Affiliates.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals, Graphcore, and NVIDIA Corporation & Affiliates are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


from collections.abc import Mapping, Sequence

# from pprint import pprint
import torch
from numpy import ndarray
from scipy.sparse import spmatrix
from torch.utils.data.dataloader import default_collate
from typing import Union, List, Optional, Dict, Type, Any, Iterable
from torch_geometric.data import Data, Batch

from loguru import logger
from graphium.data.utils import get_keys
from graphium.data.dataset import torch_enum_to_dtype


def graphium_collate_fn(
    elements: Union[List[Any], Dict[str, List[Any]]],
    labels_num_cols_dict: Optional[Dict[str, Any]] = None,
    labels_dtype_dict: Optional[Dict[str, Any]] = None,
    mask_nan: Union[str, float, Type[None]] = "raise",
    do_not_collate_keys: List[str] = [],
) -> Union[Any, Dict[str, Any]]:
    """This collate function is identical to the default
    pytorch collate function but add support for `pyg.data.Data` to batch graphs.

    Beside pyg graph collate, other objects are processed the same way
    as the original torch collate function. See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    for more details.

    Note:
        If graphium needs to manipulate other tricky-to-batch objects. Support
        for them should be added to this single collate function.

    Parameters:

        elements:
            The elements to batch. See `torch.utils.data.dataloader.default_collate`.

        labels_num_cols_dict:
            (Note): This is an attribute of the `MultitaskDataset`.
            A dictionary of the form Dict[tasks, sizes] which has task names as keys
            and the size of the label tensor as value. The size of the tensor corresponds to how many
            labels/values there are to predict for that task.

        labels_dtype_dict:
            (Note): This is an attribute of the `MultitaskDataset`.
            A dictionary of the form Dict[tasks, dtypes] which has task names as keys
            and the dtype of the label tensor as value. This is necessary to ensure the missing labels are added with NaNs of the right dtype

        mask_nan:
            Deal with the NaN/Inf when calling the function `make_pyg_graph`.
            Some values become `Inf` when changing data type. This allows to deal
            with that.

            - "raise": Raise an error when there is a nan or inf in the featurization
            - "warn": Raise a warning when there is a nan or inf in the featurization
            - "None": DEFAULT. Don't do anything
            - "Floating value": Replace nans or inf by the specified value

        do_not_batch_keys:
            Keys to ignore for the collate

    Returns:
        The batched elements. See `torch.utils.data.dataloader.default_collate`.
    """

    # Skip any elements that failed
    if None in elements:
        elements = [e for e in elements if e is not None]

    elem = elements[0]
    if isinstance(elem, Mapping):
        if "features" in elem:
            num_nodes = [d["features"].num_nodes for d in elements]
            num_edges = [d["features"].num_edges for d in elements]
        else:
            num_nodes = [d["num_nodes"] for d in elements]
            num_edges = [d["num_edges"] for d in elements]

        batch = {}
        for key in elem:
            # Multitask setting: We have to pad the missing labels
            if key == "labels":
                labels = [d[key] for d in elements]
                batch[key] = collate_labels(
                    labels, labels_num_cols_dict, labels_dtype_dict, num_nodes, num_edges
                )
            elif key == "num_nodes" or key == "num_edges":
                continue

            # If a PyG Graph is provided, use the PyG batching
            elif isinstance(elem[key], Data):
                pyg_graphs = [d[key] for d in elements]
                batch[key] = collage_pyg_graph(pyg_graphs, num_nodes)

            # Ignore the collate for specific keys
            elif key in do_not_collate_keys:
                batch[key] = [d[key] for d in elements]
            # Otherwise, use the default torch batching
            else:
                batch[key] = default_collate([d[key] for d in elements])
        return batch
    elif isinstance(elements, Sequence) and isinstance(elem, Sequence):
        temp_elements = [{ii: sub_elem for ii, sub_elem in enumerate(elem)} for elem in elements]
        batch = graphium_collate_fn(temp_elements)
        return list(batch.values())
    elif isinstance(elements, Sequence) and not isinstance(elem, Sequence):
        temp_elements = [{"temp_key": elem} for elem in elements]
        batch = graphium_collate_fn(temp_elements)
        return batch["temp_key"]
    else:
        return default_collate(elements)


def collage_pyg_graph(
    pyg_graphs: List[Data], num_nodes: List[int],
):
    """
    Function to collate pytorch geometric graphs.
    Convert all numpy types to torch
    Convert edge indices to int64

    Parameters:
        pyg_graphs: List of PyG graphs
    """

    # Calculate maximum number of nodes per graph in current batch
    max_num_nodes_per_graph = max(num_nodes)

    for pyg_graph in pyg_graphs:
        for pyg_key in get_keys(pyg_graph):
            # pad nodepair-level positional encodings
            if pyg_key.startswith("nodepair_"):
                pyg_graph[pyg_key] = pad_nodepairs(
                    pyg_graph[pyg_key], pyg_graph.num_nodes, max_num_nodes_per_graph
                )

        # Convert edge index to int64
        pyg_graph.edge_index = pyg_graph.edge_index.to(torch.int64)

    return Batch.from_data_list(pyg_graphs)


def pad_to_expected_label_size(labels: torch.Tensor, label_rows: int, label_cols: int):
    """Determine difference of ``labels`` shape to expected shape `label_size` and pad
    with ``torch.nan`` accordingly.
    """
    if len(labels.shape) == 2 and label_rows == labels.shape[0] and label_cols == labels.shape[1]:
        return labels

    missing_dims = 2 - len(labels.shape)
    for _ in range(missing_dims):
        labels.unsqueeze(-1)

    pad_sizes = [label_cols - labels.shape[1], 0, label_rows - labels.shape[0], 0]

    if any([s < 0 for s in pad_sizes]):
        logger.warning(
            f"More labels available than expected. Will remove data to fit expected size. cols: {labels.shape[1]}->{label_cols}, rows: {labels.shape[0]}->{label_rows}"
        )

    return torch.nn.functional.pad(labels, pad_sizes, value=torch.nan)


def get_expected_label_rows(label_data: Data, task: str, num_nodes: int, num_edges: int):
    """Determines expected label size based on the specfic graph properties
    and the number of targets in the task-dataset.
    """
    if task.startswith("graph_"):
        num_labels = 1
    elif task.startswith("node_"):
        num_labels = num_nodes
    elif task.startswith("edge_"):
        num_labels = num_edges
    elif task.startswith("nodepair_"):
        raise NotImplementedError()
    else:
        print("Task name " + task + " in get_expected_label_rows")
        raise NotImplementedError()
    return num_labels


def collate_labels(
    labels: List[Data],
    labels_num_cols_dict: Optional[Dict[str, Any]] = None,
    labels_dtype_dict: Optional[Dict[str, Any]] = None,
    num_nodes: List[int] = None,
    num_edges: List[int] = None,
):
    """Collate labels for multitask learning.

    Parameters:
        labels: List of labels
        labels_num_cols_dict: Dict of the form Dict[tasks, sizes] which has task names as keys
            and the size of the label tensor as value. The size of the tensor corresponds to how many
            labels/values there are to predict for that task.
        labels_dtype_dict:
            (Note): This is an attribute of the `MultitaskDataset`.
            A dictionary of the form Dict[tasks, dtypes] which has task names as keys
            and the dtype of the label tensor as value. This is necessary to ensure the missing labels are added with NaNs of the right dtype

    Returns:
        A dictionary of the form Dict[tasks, labels] where tasks is the name of the task and labels
        is a tensor of shape (batch_size, *labels_num_cols_dict[task]).
    """
    if labels_num_cols_dict is not None:
        for index, this_label in enumerate(labels):
            label_keys_set = set(get_keys(this_label))
            empty_task_labels = set(labels_num_cols_dict.keys()) - label_keys_set
            for task in empty_task_labels:
                label_rows = get_expected_label_rows(this_label, task, num_nodes[index], num_edges[index])
                dtype = torch_enum_to_dtype(labels_dtype_dict[task])
                this_label[task] = torch.full(
                    (label_rows, labels_num_cols_dict[task]), fill_value=torch.nan, dtype=dtype
                )

            for task in label_keys_set - set(["x", "edge_index"]) - empty_task_labels:
                label_rows = get_expected_label_rows(this_label, task, num_nodes[index], num_edges[index])

                if not isinstance(this_label[task], (torch.Tensor)):
                    this_label[task] = torch.as_tensor(this_label[task])

                # Ensure explicit task dimension also for single task labels
                if len(this_label[task].shape) == 1:
                    # Distinguish whether target dim or entity dim is missing
                    if label_rows == this_label[task].shape[0]:
                        # num graphs/nodes/edges/nodepairs already matching
                        this_label[task] = this_label[task].unsqueeze(1)
                    else:
                        # data lost unless entity dim is supposed to be 1
                        if label_rows == 1:
                            this_label[task] = this_label[task].unsqueeze(0)
                        else:
                            raise ValueError(
                                f"Labels for {label_rows} nodes/edges/nodepairs expected, got 1."
                            )

                this_label[task] = pad_to_expected_label_size(
                    this_label[task], label_rows, labels_num_cols_dict[task]
                )

    return Batch.from_data_list(labels)


def pad_nodepairs(pe: torch.Tensor, num_nodes: int, max_num_nodes_per_graph: int):
    """
    This function zero-pads nodepair-level positional encodings to conform with the batching logic.

    Parameters:
        pe (torch.Tensor, [num_nodes, num_nodes, num_feat]): Nodepair pe
        num_nodes (int): Number of nodes of processed graph
        max_num_nodes_per_graph (int): Maximum number of nodes among graphs in current batch

    Returns:
        padded_pe (torch.Tensor, [num_nodes, max_num_nodes_per_graph, num_feat]): padded nodepair pe tensor
    """
    padded_pe = torch.zeros((num_nodes, max_num_nodes_per_graph, pe.size(-1)), dtype=pe.dtype)
    padded_pe[:, :num_nodes] = pe[:, :num_nodes]
    # Above, pe[:, :num_nodes] in the rhs is needed to "overwrite" zero-padding from previous epoch

    return padded_pe
