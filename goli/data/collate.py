from collections.abc import Mapping
#from pprint import pprint
import torch
from torch.utils.data.dataloader import default_collate
from inspect import signature, _empty
<<<<<<< HEAD
from typing import Union, List, Optional, Dict, Type
import dgl
from torch_geometric.data import Data, Batch
=======
from typing import Union, List, Optional, Dict, Type, Any
>>>>>>> origin/master

from goli.features import GraphDict


def goli_collate_fn(
    elements,
<<<<<<< HEAD
    labels_size_dict: Optional[Dict[str, int]],
    mask_nan: Union[str, float, Type[None]] = "raise",
    do_not_collate_keys: List[str] = [],
=======
    labels_size_dict: Optional[Dict[str, Any]] = None,
    mask_nan: Union[str, float, Type[None]] = "raise", 
    do_not_collate_keys: List[str] = []
>>>>>>> origin/master
):
    """This collate function is identical to the default
    pytorch collate function but add support for `dgl.DGLGraph`
    objects and use `dgl.batch` to batch graphs.

    Beside dgl graph collate, other objects are processed the same way
    as the original torch collate function. See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    for more details.

    Important:
        Only dgl graph within a dict are currently supported. It should not be hard
        to support dgl graphs from other objects.

    Note:
        If goli needs to manipulate other tricky-to-batch objects. Support
        for them should be added to this single collate function.

    Parameters:

        elements:
            The elements to batch. See `torch.utils.data.dataloader.default_collate`.

        labels_size_dict:
            (Note): This is an attribute of the MultiTaskDGLDataset.
            A dictionary of the form Dict[tasks, sizes] which has task names as keys
            and the size of the label tensor as value. The size of the tensor corresponds to how many
            labels/values there are to predict for that task.

        mask_nan:
            Deal with the NaN/Inf when calling the function `dgl_dict_to_graph`.
            Some values become `Inf` when changing data type. This allows to deal
            with that.

            - "raise": Raise an error when there is a nan or inf in the featurization
            - "warn": Raise a warning when there is a nan or inf in the featurization
            - "None": DEFAULT. Don't do anything
            - "Floating value": Replace nans or inf by the specified value

        do_not_batch_keys:
            Keys to ignore for the collate
    """

    elem = elements[0]

    if isinstance(elem, Mapping):
        batch = {}
        for key in elem:
            # If the features are a dictionary containing DGLGraph elements,
            # Convert to DGLGraph and use the dgl batching.
            if isinstance(elem[key], GraphDict):
                graphs = [d[key].make_dgl_graph(mask_nan=mask_nan) for d in elements]
                batch[key] = dgl.batch(graphs)

            # If a DGLGraph is provided, use the dgl batching
            elif isinstance(elem[key], dgl.DGLGraph):
                batch[key] = dgl.batch([d[key] for d in elements])

            # If a PyG Graph is provided, use the PyG batching
            elif isinstance(elem[key], Data):
                batch[key] = Batch.from_data_list([d[key] for d in elements])

            # Ignore the collate for specific keys
            elif key in do_not_collate_keys:
                batch[key] = [d[key] for d in elements]

            # Multitask setting: We have to pad the missing labels
<<<<<<< HEAD
            elif key == "labels":
                if labels_size_dict is not None:  # If we have to pad for the MTL setting
                    for datum in elements:
                        nonempty_labels = datum["labels"].keys()
                        for label in labels_size_dict:
                            if label not in nonempty_labels:
                                datum["labels"][label] = torch.full(
                                    (labels_size_dict[label], len(elements)), torch.nan
                                )
                else:
                    batch[key] = default_collate([d[key] for d in elements])
=======
            elif key == 'labels':
                if labels_size_dict is not None:
                    for datum in elements:
                        empty_task_labels = set(labels_size_dict.keys()) - set(datum["labels"].keys())
                        for task in empty_task_labels:
                            datum['labels'][task] = torch.full((len(elements), labels_size_dict[task]), torch.nan)
                batch[key] = default_collate([datum[key] for datum in elements])
>>>>>>> origin/master
            # Otherwise, use the default torch batching
            else:
                batch[key] = default_collate([d[key] for d in elements])
#        print("The batch contains: ")
#        pprint(batch)
        return batch
    else:
        return default_collate(elements)
