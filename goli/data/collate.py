from collections.abc import Mapping, Sequence

# from pprint import pprint
import torch
from numpy import ndarray
from scipy.sparse import spmatrix
from torch.utils.data.dataloader import default_collate
from typing import Union, List, Optional, Dict, Type, Any, Iterable
from torch_geometric.data import Data, Batch

from goli.features import GraphDict, to_dense_array
from goli.utils.packing import fast_packing, get_pack_sizes, node_to_pack_indices_mask


def goli_collate_fn(
    elements: Union[List[Any], Dict[str, List[Any]]],
    labels_size_dict: Optional[Dict[str, Any]] = None,
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
        If goli needs to manipulate other tricky-to-batch objects. Support
        for them should be added to this single collate function.

    Parameters:

        elements:
            The elements to batch. See `torch.utils.data.dataloader.default_collate`.

        labels_size_dict:
            (Note): This is an attribute of the `MultitaskDataset`.
            A dictionary of the form Dict[tasks, sizes] which has task names as keys
            and the size of the label tensor as value. The size of the tensor corresponds to how many
            labels/values there are to predict for that task.

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
            # If the features are a dictionary containing GraphDict elements,
            # Convert to pyg graphs and use the pyg batching.
            if isinstance(elem[key], GraphDict):
                pyg_graphs = [d[key].make_pyg_graph(mask_nan=mask_nan) for d in elements]
                batch[key] = collage_pyg_graph(pyg_graphs)

            # If a PyG Graph is provided, use the PyG batching
            elif isinstance(elem[key], Data):
                pyg_graphs = [d[key] for d in elements]
                batch[key] = collage_pyg_graph(pyg_graphs, batch_size_per_pack=batch_size_per_pack)

            # Ignore the collate for specific keys
            elif key in do_not_collate_keys:
                batch[key] = [d[key] for d in elements]

            # Multitask setting: We have to pad the missing labels
            elif key == "labels":
                labels = [d[key] for d in elements]
                batch[key] = collate_labels(labels, labels_size_dict)

            # Otherwise, use the default torch batching
            else:
                batch[key] = default_collate([d[key] for d in elements])
        return batch
    elif isinstance(elements, Sequence) and isinstance(elem, Sequence):
        temp_elements = [{ii: sub_elem for ii, sub_elem in enumerate(elem)} for elem in elements]
        batch = goli_collate_fn(temp_elements)
        return list(batch.values())
    elif isinstance(elements, Sequence) and not isinstance(elem, Sequence):
        temp_elements = [{"temp_key": elem} for elem in elements]
        batch = goli_collate_fn(temp_elements)
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
        for pyg_key in pyg_graph.keys:
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
    if batch_size_per_pack is not None:
        num_nodes = [g.num_nodes for g in pyg_batch]
        packed_graph_idx = fast_packing(num_nodes, batch_size_per_pack)

        # Get the node to pack indices and the mask
        pack_from_node_idx, pack_attn_mask = node_to_pack_indices_mask(packed_graph_idx, num_nodes)
        for pyg_graph in pyg_batch:
            pyg_graph.pack_from_node_idx = pack_from_node_idx
            pyg_graph.pack_attn_mask = pack_attn_mask

    return Batch.from_data_list(pyg_batch)


def collate_labels(
    labels: List[Dict[str, torch.Tensor]],
    labels_size_dict: Optional[Dict[str, Any]] = None,
):
    """Collate labels for multitask learning.

    Parameters:
        labels: List of labels
        labels_size_dict: Dict of the form Dict[tasks, sizes] which has task names as keys
            and the size of the label tensor as value. The size of the tensor corresponds to how many
            labels/values there are to predict for that task.

    Returns:
        A dictionary of the form Dict[tasks, labels] where tasks is the name of the task and labels
        is a tensor of shape (batch_size, *labels_size_dict[task]).
    """
    labels_dict = {}

    if labels_size_dict is not None:
        for this_label in labels:
            empty_task_labels = set(labels_size_dict.keys()) - set(this_label.keys())
            for task in empty_task_labels:
                this_label[task] = torch.full([*labels_size_dict[task]], torch.nan)
            for task in this_label.keys():
                if not isinstance(task, torch.Tensor):
                    this_label[task] = torch.as_tensor(this_label[task])
    labels_dict = default_collate(labels)

    return labels_dict


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
