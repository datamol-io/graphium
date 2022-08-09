# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import List, Optional, Tuple, Union

import poptorch
import torch
from torch_geometric.data import Batch, Dataset
from goli.ipu.ipu_pad import Pad


class CombinedBatchingCollator:
    """
    Collator object that manages the combined batch size defined as:

        combined_batch_size = mini_batch_size * device_iterations
                             * replication_factor * gradient_accumulation

    This is intended to be used in combination with the poptorch.DataLoader
    """

    def __init__(self, mini_batch_size, collate_fn=None, max_num_nodes_per_graph=25, max_num_edges_per_graph=50, max_num_nodes=None, max_num_edges=None):
        """
        :param mini_batch_size (int): mini batch size used by the SchNet model
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.collate_fn = collate_fn

        # Get the maximum number of nodes
        if max_num_nodes is not None:
            assert max_num_nodes_per_graph is None, "Cannot use `max_num_nodes` and `max_num_nodes_per_graph` simultaneously"
            self.max_num_nodes = max_num_nodes
        elif max_num_nodes_per_graph is not None:
            assert max_num_nodes is None, "Cannot use `max_num_nodes` and `max_num_nodes_per_graph` simultaneously"
            self.max_num_nodes = max_num_nodes_per_graph * mini_batch_size
        else:
            raise ValueError("Must provide either `max_num_nodes` or `max_num_nodes_per_graph`")

        # Get the maximum number of edges
        if max_num_edges is not None:
            assert max_num_edges_per_graph is None, "Cannot use `max_num_edges` and `max_num_edges_per_graph` simultaneously"
            self.max_num_edges = max_num_edges
        elif max_num_edges_per_graph is not None:
            assert max_num_edges is None, "Cannot use `max_num_edges` and `max_num_edges_per_graph` simultaneously"
            self.max_num_edges = max_num_edges_per_graph * mini_batch_size
        else:
            raise ValueError("Must provide either `max_num_nodes` or `max_num_nodes_per_graph`")


    def __call__(self, batch):
        '''
        padding option 2 pad each batch to be same size
        '''
        if (self.collate_fn != None):
            batch = self.collate_fn(batch)

        transform = Pad(max_num_nodes=self.max_num_nodes, max_num_edges=self.max_num_edges, include_keys=['batch'])

        batch['features'] = transform(batch['features'])
        return batch


def create_ipu_dataloader(dataset: Dataset,
                      ipu_opts: Optional[poptorch.Options] = None,
                      batch_size: Optional[int] = 1,
                      collate_fn=None,
                      max_num_nodes_per_graph=25,
                      max_num_edges_per_graph=50,
                      max_num_nodes=None,
                      max_num_edges=None,
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
    :param **kwargs (optional): Additional arguments of
        :class:`poptorch.DataLoader`.
    """
    if ipu_opts is None:
        # Create IPU default options
        ipu_opts = poptorch.Options()

    collater = CombinedBatchingCollator(batch_size, collate_fn=collate_fn,
                                max_num_nodes_per_graph=max_num_nodes_per_graph,
                                max_num_edges_per_graph=max_num_edges_per_graph,
                                max_num_nodes=max_num_nodes,
                                max_num_edges=max_num_edges,)

    return poptorch.DataLoader(ipu_opts,
                               dataset=dataset,
                               batch_size=batch_size,
                               collate_fn=collater,
                               **kwargs)
