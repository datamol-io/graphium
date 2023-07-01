from collections.abc import Mapping, Sequence

# from pprint import pprint
import torch
from numpy import ndarray
from scipy.sparse import spmatrix
from torch.utils.data.dataloader import default_collate
from typing import Union, List, Optional, Dict, Type, Any, Iterable
from torch_geometric.data import Data, Batch

from graphium.features import GraphDict, to_dense_array
from graphium.utils.packing import fast_packing, get_pack_sizes, node_to_pack_indices_mask
from loguru import logger
from graphium.data.utils import get_keys


def graphium_collate_fn(
    elements: Union[List[Any], Dict[str, List[Any]]],
    labels_size_dict: Optional[Dict[str, Any]] = None,
    labels_dtype_dict: Optional[Dict[str, Any]] = None,
    mask_nan: Union[str, float, Type[None]] = "raise",
    do_not_collate_keys: List[str] = [],
    batch_size_per_pack: Optional[int] = None,
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

        labels_size_dict:
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

        batch_size_per_pack: The number of graphs to pack together.
            This is useful for using packing with the Transformer.
            If None, no packing is done.
            Otherwise, indices are generated to map the nodes to the pack they belong to under the key `"pack_from_node_idx"`,
            with an additional mask to indicate which nodes are from the same graph under the key `"pack_attn_mask"`.

    Returns:
        The batched elements. See `torch.utils.data.dataloader.default_collate`.
    """

    elem = elements[0]
    if isinstance(elem, Mapping):
        batch = {}
        for key in elem:
            # Multitask setting: We have to pad the missing labels
            if key == "labels":
                labels = [d[key] for d in elements]
                batch[key] = collate_labels(labels, labels_size_dict, labels_dtype_dict)

            # If the features are a dictionary containing GraphDict elements,
            # Convert to pyg graphs and use the pyg batching.
            elif isinstance(elem[key], GraphDict):
                pyg_graphs = [d[key].make_pyg_graph(mask_nan=mask_nan) for d in elements]
                batch[key] = collage_pyg_graph(pyg_graphs)

            # If a PyG Graph is provided, use the PyG batching
            elif isinstance(elem[key], Data):
                pyg_graphs = [d[key] for d in elements]
                batch[key] = collage_pyg_graph(pyg_graphs, batch_size_per_pack=batch_size_per_pack)

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


def collage_pyg_graph(pyg_graphs: Iterable[Union[Data, Dict]], batch_size_per_pack: Optional[int] = None):
    """
    Function to collate pytorch geometric graphs.
    Convert all numpy types to torch
    Convert edge indices to int64

    Parameters:
        pyg_graphs: Iterable of PyG graphs
        batch_size_per_pack: The number of graphs to pack together.
            This is useful for using packing with the Transformer,
    """

    # Calculate maximum number of nodes per graph in current batch
    num_nodes_list = []
    for pyg_graph in pyg_graphs:
        num_nodes_list.append(pyg_graph["num_nodes"])
    max_num_nodes_per_graph = max(num_nodes_list)

    pyg_batch = []
    for pyg_graph in pyg_graphs:
        for pyg_key in get_keys(pyg_graph):
            tensor = pyg_graph[pyg_key]

            # Convert numpy/scipy to Pytorch
            if isinstance(tensor, (ndarray, spmatrix)):
                tensor = torch.as_tensor(to_dense_array(tensor, tensor.dtype))

            # pad nodepair-level positional encodings
            if pyg_key.startswith("nodepair_"):
                pyg_graph[pyg_key] = pad_nodepairs(tensor, pyg_graph["num_nodes"], max_num_nodes_per_graph)
            else:
                pyg_graph[pyg_key] = tensor

        # Convert edge index to int64
        pyg_graph.edge_index = pyg_graph.edge_index.to(torch.int64)
        pyg_batch.append(pyg_graph)

    # Apply the packing at the mini-batch level. This is useful for using packing with the Transformer,
    # especially in the case of the large graphs being much larger than the small graphs.
    # CAREFUL!!! This changes the order of the graphs in the batch, without changing the order of the labels or other objects.
    # An error is raised temporarily.
    if batch_size_per_pack is not None:
        raise NotImplementedError(
            "Packing is not yet functional, as it changes the order of the graphs in the batch without changing the label order"
        )
        num_nodes = [g.num_nodes for g in pyg_batch]
        packed_graph_idx = fast_packing(num_nodes, batch_size_per_pack)

        # Get the node to pack indices and the mask
        pack_from_node_idx, pack_attn_mask = node_to_pack_indices_mask(packed_graph_idx, num_nodes)
        for pyg_graph in pyg_batch:
            pyg_graph.pack_from_node_idx = pack_from_node_idx
            pyg_graph.pack_attn_mask = pack_attn_mask

    return Batch.from_data_list(pyg_batch)


def pad_to_expected_label_size(labels: torch.Tensor, label_size: List[int]):
    """Determine difference of ``labels`` shape to expected shape `label_size` and pad
    with ``torch.nan`` accordingly.
    """
    if label_size == list(labels.shape):
        return labels

    missing_dims = len(label_size) - len(labels.shape)
    for _ in range(missing_dims):
        labels.unsqueeze(-1)

    pad_sizes = [(0, expected - actual) for expected, actual in zip(label_size, labels.shape)]
    pad_sizes = [item for before_after in pad_sizes for item in before_after]
    pad_sizes.reverse()

    if any([s < 0 for s in pad_sizes]):
        logger.warning(f"More labels available than expected. Will remove data to fit expected size.")

    return torch.nn.functional.pad(labels, pad_sizes, value=torch.nan)


def collate_pyg_graph_labels(pyg_labels: List[Data]):
    """
    Function to collate pytorch geometric labels.
    Convert all numpy types to torch

    Parameters:
        pyg_labels: Iterable of PyG label Data objects
    """
    pyg_batch = []
    for pyg_label in pyg_labels:
        for pyg_key in set(get_keys(pyg_label)) - set(["x", "edge_index"]):
            tensor = pyg_label[pyg_key]
            # Convert numpy/scipy to Pytorch
            if isinstance(tensor, (ndarray, spmatrix)):
                tensor = torch.as_tensor(to_dense_array(tensor, tensor.dtype))

            pyg_label[pyg_key] = tensor

        pyg_batch.append(pyg_label)

    return Batch.from_data_list(pyg_batch)


def get_expected_label_size(label_data: Data, task: str, label_size: List[int]):
    """Determines expected label size based on the specfic graph properties
    and the number of targets in the task-dataset.
    """
    if task.startswith("graph_"):
        num_labels = 1
    elif task.startswith("node_"):
        num_labels = label_data.x.size(0)
    elif task.startswith("edge_"):
        num_labels = label_data.edge_index.size(1)
    elif task.startswith("nodepair_"):
        raise NotImplementedError()
    return [num_labels] + label_size


def collate_labels(
    labels: List[Data],
    labels_size_dict: Optional[Dict[str, Any]] = None,
    labels_dtype_dict: Optional[Dict[str, Any]] = None,
):
    """Collate labels for multitask learning.

    Parameters:
        labels: List of labels
        labels_size_dict: Dict of the form Dict[tasks, sizes] which has task names as keys
            and the size of the label tensor as value. The size of the tensor corresponds to how many
            labels/values there are to predict for that task.
        labels_dtype_dict:
            (Note): This is an attribute of the `MultitaskDataset`.
            A dictionary of the form Dict[tasks, dtypes] which has task names as keys
            and the dtype of the label tensor as value. This is necessary to ensure the missing labels are added with NaNs of the right dtype

    Returns:
        A dictionary of the form Dict[tasks, labels] where tasks is the name of the task and labels
        is a tensor of shape (batch_size, *labels_size_dict[task]).
    """
    if labels_size_dict is not None:
        for this_label in labels:
            for task in labels_size_dict.keys():
                labels_size_dict[task] = list(labels_size_dict[task])
                if len(labels_size_dict[task]) >= 2:
                    labels_size_dict[task] = labels_size_dict[task][1:]
                elif not task.startswith("graph_"):
                    labels_size_dict[task] = [1]
            label_keys_set = set(get_keys(this_label))
            empty_task_labels = set(labels_size_dict.keys()) - label_keys_set
            for task in empty_task_labels:
                labels_size_dict[task] = get_expected_label_size(this_label, task, labels_size_dict[task])
                dtype = labels_dtype_dict[task]
                this_label[task] = torch.full([*labels_size_dict[task]], torch.nan, dtype=dtype)

            for task in label_keys_set - set(["x", "edge_index"]) - empty_task_labels:
                labels_size_dict[task] = get_expected_label_size(this_label, task, labels_size_dict[task])

                if not isinstance(this_label[task], (torch.Tensor)):
                    this_label[task] = torch.as_tensor(this_label[task])

                # Ensure explicit task dimension also for single task labels
                if len(this_label[task].shape) == 1:
                    # Distinguish whether target dim or entity dim is missing
                    if labels_size_dict[task][0] == this_label[task].shape[0]:
                        # num graphs/nodes/edges/nodepairs already matching
                        this_label[task] = this_label[task].unsqueeze(1)
                    else:
                        # data lost unless entity dim is supposed to be 1
                        if labels_size_dict[task][0] == 1:
                            this_label[task] = this_label[task].unsqueeze(0)
                        else:
                            raise ValueError(
                                f"Labels for {labels_size_dict[task][0]} nodes/edges/nodepairs expected, got 1."
                            )

                this_label[task] = pad_to_expected_label_size(this_label[task], labels_size_dict[task])

    return collate_pyg_graph_labels(labels)


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
