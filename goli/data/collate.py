from collections.abc import Mapping, Sequence

# from pprint import pprint
import torch
from numpy import ndarray
from scipy.sparse import spmatrix
from torch.utils.data.dataloader import default_collate
from typing import Union, List, Optional, Dict, Type, Any, Iterable
from torch_geometric.data import Data, Batch

from goli.features import GraphDict, to_dense_array


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
    Returns:
        A dictionary of the batch
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
                batch[key] = collage_pyg_graph(pyg_graphs)

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

    pyg_batch = []
    for pyg_graph in pyg_graphs:
        for pyg_key in pyg_graph.keys:
            tensor = pyg_graph[pyg_key]

            # Convert numpy/scipy to Pytorch
            if isinstance(tensor, (ndarray, spmatrix)):
                tensor = torch.as_tensor(to_dense_array(tensor, tensor.dtype))

            pyg_graph[pyg_key] = tensor

        # Convert edge index to int64
        pyg_graph.edge_index = pyg_graph.edge_index.to(torch.int64)
        pyg_batch.append(pyg_graph)

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
                this_label[task] = torch.full((len(labels), *labels_size_dict[task]), torch.nan)
    labels_dict = default_collate(labels)

    return labels_dict
