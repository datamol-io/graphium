# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import List, Optional, Tuple, Union

import poptorch
import torch
from torch_geometric.data import Batch, Dataset
from goli.data.ipu_pad import Pad

class TupleCollator(object):
    """
    Collate a PyG Batch as a tuple of tensors
    """

    def __init__(self,
                 include_keys: Optional[Union[List[str], Tuple[str]]] = None):
        """
        :param include_keys (optional): Keys to include from the batch in the
            output tuple specified as either a list or tuple of strings. The
            ordering of the keys is preserved in the output tuple. The keys can
            be inferred through the Batch.keys property when not provided.
        """
        super().__init__()
        self.include_keys = include_keys
        channel = poptorch.profiling.Channel(self.__class__.__name__)
        channel.instrument(self, "__call__")

    def __call__(self, data_list):
        with poptorch.profiling.Channel("Batch").tracepoint("from_data_list"):
            batch = Batch.from_data_list(data_list)

        if self.include_keys is None:
            # Check the keys property for tensors and cache the result
            keys = filter(lambda k: torch.is_tensor(getattr(batch, k)),
                          batch.keys)
            self.include_keys = tuple(keys)

        assert all([hasattr(batch, k) for k in self.include_keys]), \
            "Batch is missing a required key: " \
            f"include_keys='{self.include_keys}'"

        return tuple(getattr(batch, k) for k in self.include_keys)


class CombinedBatchingCollator:
    """
    Collator object that manages the combined batch size defined as:

        combined_batch_size = mini_batch_size * device_iterations
                             * replication_factor * gradient_accumulation

    This is intended to be used in combination with the poptorch.DataLoader
    """

    def __init__(self, mini_batch_size, include_keys=None, collate_fn=None):
        """
        :param mini_batch_size (int): mini batch size used by the SchNet model
        :param include_keys (optional): Keys to include from the batch in the
            output tuple specified as either a list or tuple of strings. The
            ordering of the keys is preserved in the output tuple. The keys can
            be inferred through the Batch.to_dict method when not provided.
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.batch_to_tuple = TupleCollator(include_keys=include_keys)
        self.collate_fn = collate_fn

    # def __call__(self, batch):
    #     if (self.collate_fn != None):
    #         batch = self.collate_fn(batch)
    #     graphs = batch['features']
    #     num_items = len(graphs)
    #     assert num_items % self.mini_batch_size == 0, "Invalid batch size. " \
    #         f"Got {num_items} graphs and" \
    #         f"mini_batch_size={self.mini_batch_size}."

    #     num_mini_batches = num_items // self.mini_batch_size
    #     batches = [None] * num_mini_batches
    #     start = 0
    #     stride = self.mini_batch_size

    #     for i in range(num_mini_batches):
    #         slices = graphs[start:start + stride]
    #         batches[i] = self.batch_to_tuple(slices)
    #         start += stride

    #     num_outputs = len(batches[0])
    #     outputs = [None] * num_outputs

    #     for i in range(num_outputs):
    #         outputs[i] = torch.stack(tuple(item[i] for item in batches))

    #     '''
    #     convert tuple of torch tensors into pyg batch
    #     '''

    #     outputs = tuple(outputs)
    #     #batch['features'] = tuple(outputs)
    #     return outputs


    def __call__(self, batch):
        if (self.collate_fn != None):
            batch = self.collate_fn(batch)
        graphs = Batch.from_data_list(batch['features'])

        transform = Pad(max_num_nodes=self.mini_batch_size*20, max_num_edges=self.mini_batch_size*40)

        batch['features'] = transform(graphs)
        return batch


def create_dataloader(dataset: Dataset,
                      ipu_opts: Optional[poptorch.Options] = None,
                      batch_size: Optional[int] = 1,
                      include_keys: Optional[Union[List[str],
                                                   Tuple[str]]] = None,
                      collate_fn=None,
                      **kwargs):
    """
    Creates a poptorch.DataLoader for graph datasets
    Applies the mini-batching method of concatenating multiple graphs into a
    single graph with multiple disconnected subgraphs. See:
    https://pytorch-geometric.readthedocs.io/en/2.0.2/notes/batching.html

    :param dataset: The torch_geometric.data.Dataset instance from which to
        load the graph examples for the IPU.
    :param ipu_opts (optional): The poptorch.Options used by the
        poptorch.DataLoader. Will use the default options if not provided.
    :param batch_size (optional): How many graph examples to load in each batch
        (default: 1).
    :param include_keys (optional): Keys to include from the batch in the
        output tuple specified as either a list or tuple of strings. The
        ordering of the keys is preserved in the output tuple. The keys can
        be inferred through the Batch.to_dict method when not provided.
    :param **kwargs (optional): Additional arguments of
        :class:`poptorch.DataLoader`.
    """
    if ipu_opts is None:
        # Create IPU default options
        ipu_opts = poptorch.Options()

    # assert 'collate_fn' not in kwargs, \
    #     "Cannot set collate_fn with poppyg.create_dataloader. "\
    #     "Use poptorch.DataLoader directly if you need this functionality."

    collater = CombinedBatchingCollator(batch_size, include_keys, collate_fn=collate_fn)

    return poptorch.DataLoader(ipu_opts,
                               dataset=dataset,
                               batch_size=batch_size,
                               collate_fn=collater,
                               **kwargs)
