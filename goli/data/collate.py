from collections.abc import Mapping
from torch.utils.data.dataloader import default_collate
import dgl
from inspect import signature, _empty

from goli.features import dgl_dict_to_graph


def goli_collate_fn(elements):
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
                graphs = [dgl_dict_to_graph(**d[key]) for d in elements]
                batch[key] = dgl.batch(graphs)

            # If a DGLGraph is provided, use the dgl batching
            elif isinstance(elem[key], dgl.DGLGraph):
                batch[key] = dgl.batch([d[key] for d in elements])

            # Otherwise, use the default torch batching
            else:
                batch[key] = default_collate([d[key] for d in elements])
        return batch
    else:
        return default_collate(elements)
