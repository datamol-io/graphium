import collections.abc

from torch.utils.data.dataloader import default_collate

import dgl


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

    if isinstance(elem, collections.abc.Mapping):
        batch = {}
        for key in elem:
            if isinstance(elem[key], dgl.DGLGraph):
                batch[key] = dgl.batch([d[key] for d in elements])
            else:
                batch[key] = default_collate([d[key] for d in elements])
        return batch
    else:
        return default_collate(elements)
