from collections.abc import Mapping
from torch.utils.data.dataloader import default_collate
import dgl
from inspect import signature, _empty
from typing import Union, List

from goli.features import dgl_dict_to_graph


def goli_collate_fn(
    elements,
    mask_nan: Union[str, float, type(None)] = "raise",
    do_not_collate_keys: List[str] = []
):
    """This collate function is identical to the default
    pytorch collate function but add support for `dgl.DGLGraph`
    objects and use `dgl.batch` to batch graphs.

    Beside dgl graph collate, other objects are processed the same way
    as the original torch collate function. See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    for more details.

    Important:
        Only dgl graph within a dict are currently supported. It's should not be hard
        to support dgl graphs from other objects.

    Note:
        If goli needs to manipulate other tricky-to-batch objects. Support
        for them should be added to this single collate function.

    Parameters:

        elements:
            The elements to batch. See `torch.utils.data.dataloader.default_collate`.

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

    params = signature(dgl_dict_to_graph).parameters
    dgl_dict_mandatory_params = [key for key, val in params.items() if val.default == _empty]

    if isinstance(elem, Mapping):
        batch = {}
        for key in elem:

            # If the features are a dictionary containing DGLGraph elements,
            # Convert to DGLGraph and use the dgl batching.
            if isinstance(elem[key], Mapping) and all(
                [this_param in list(elem[key].keys()) for this_param in dgl_dict_mandatory_params]
            ):
                graphs = [dgl_dict_to_graph(**d[key], mask_nan=mask_nan) for d in elements]
                batch[key] = dgl.batch(graphs)

            # If a DGLGraph is provided, use the dgl batching
            elif isinstance(elem[key], dgl.DGLGraph):
                batch[key] = dgl.batch([d[key] for d in elements])

            # Ignore the collate for specific keys
            elif key in do_not_collate_keys:
                batch[key] = [d[key] for d in elements]

            # Otherwise, use the default torch batching
            else:
                batch[key] = default_collate([d[key] for d in elements])
        return batch
    else:
        return default_collate(elements)
