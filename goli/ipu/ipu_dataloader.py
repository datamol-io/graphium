# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import dataclass

import poptorch
import torch
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.transforms import BaseTransform


@dataclass
class IPUDataloaderOptions:
    r"""
    This data class stores the arguments necessary to instantiate a model for the Predictor.

        Parameters:
            model_class:
                pytorch module used to create a model

            model_kwargs:
                Key-word arguments used to initialize the model from `model_class`.
    """

    batch_size: int
    max_num_nodes: Optional[int] = None
    max_num_nodes_per_graph: Optional[int] = None
    max_num_edges: Optional[int] = None
    max_num_edges_per_graph: Optional[int] = None

    def set_kwargs(self):

        # Get the maximum number of nodes
        if self.max_num_nodes is not None:
            assert self.max_num_nodes_per_graph is None, "Cannot use `max_num_nodes` and `max_num_nodes_per_graph` simultaneously"
        elif self.max_num_nodes_per_graph is not None:
            assert self.max_num_nodes is None, "Cannot use `max_num_nodes` and `max_num_nodes_per_graph` simultaneously"
            self.max_num_nodes = self.max_num_nodes_per_graph * self.batch_size
        else:
            raise ValueError("Must provide either `max_num_nodes` or `max_num_nodes_per_graph`")

        # Get the maximum number of edges
        if self.max_num_edges is not None:
            assert self.max_num_edges_per_graph is None, "Cannot use `max_num_edges` and `max_num_edges_per_graph` simultaneously"
        elif self.max_num_edges_per_graph is not None:
            assert self.max_num_edges is None, "Cannot use `max_num_edges` and `max_num_edges_per_graph` simultaneously"
            self.max_num_edges = self.max_num_edges_per_graph * self.batch_size
        else:
            raise ValueError("Must provide either `max_num_nodes` or `max_num_nodes_per_graph`")


class CombinedBatchingCollator:
    """
    Collator object that manages the combined batch size defined as:

        combined_batch_size = batch_size * device_iterations
                             * replication_factor * gradient_accumulation

    This is intended to be used in combination with the poptorch.DataLoader
    """

    def __init__(self, batch_size, max_num_nodes, max_num_edges, collate_fn=None):
        """
        :param batch_size (int): mini batch size used by the SchNet model
        """
        super().__init__()
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges

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
                      ipu_dataloader_options: IPUDataloaderOptions,
                      ipu_opts: Optional[poptorch.Options] = None,
                      batch_size: Optional[int] = 1,
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
    :param **kwargs (optional): Additional arguments of
        :class:`poptorch.DataLoader`.
    """
    if ipu_opts is None:
        # Create IPU default options
        ipu_opts = poptorch.Options()

    collater = CombinedBatchingCollator(batch_size, collate_fn=collate_fn,
                                max_num_nodes=ipu_dataloader_options.max_num_nodes,
                                max_num_edges=ipu_dataloader_options.max_num_edges,)

    return poptorch.DataLoader(ipu_opts,
                               dataset=dataset,
                               batch_size=batch_size,
                               collate_fn=collater,
                               **kwargs)



class Pad(BaseTransform):
    """
    Data transform that applies padding to enforce consistent tensor shapes.
    """

    def __init__(self,
                 max_num_nodes: int,
                 max_num_edges: Optional[int] = None,
                 node_value: Optional[float] = None,
                 edge_value: Optional[float] = None,
                 include_keys: Optional[Union[List[str], Tuple[str]]] = None):
        """
        :param max_num_nodes (int): The maximum number of nodes
        :param max_num_edges (optional): The maximum number of edges.
        """
        super().__init__()
        self.max_num_nodes = max_num_nodes

        if max_num_edges:
            self.max_num_edges = max_num_edges
        else:
            # Assume fully connected graph
            self.max_num_edges = max_num_nodes * (max_num_nodes - 1)

        self.node_value = 0.0 if node_value is None else node_value
        self.edge_value = 0.0 if edge_value is None else edge_value
        self.include_keys = include_keys

    def validate(self, data):
        """
        Validates that the input graph does not exceed the constraints that:

          * the number of nodes must be <= max_num_nodes
          * the number of edges must be <= max_num_edges

        :returns: Tuple containing the number nodes and the number of edges
        """
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        assert num_nodes <= self.max_num_nodes, \
            f"Too many nodes. Graph has {num_nodes} nodes "\
            f"and max_num_edges is {self.max_num_nodes}."

        assert num_edges <= self.max_num_edges, \
            f"Too many edges. Graph has {num_edges} edges defined "\
            f"and max_num_edges is {self.max_num_edges}."

        return num_nodes, num_edges

    def __call__(self, data):
        num_nodes, num_edges = self.validate(data)
        num_pad_nodes = self.max_num_nodes - num_nodes
        num_pad_edges = self.max_num_edges - num_edges
        # Create a copy to update with padded features
        new_data = deepcopy(data)


        real_graphs = new_data.to_data_list()

        for g in real_graphs:
            g.graph_is_true = torch.tensor([1])
            g.node_is_true = torch.full([g.num_nodes], 1)
            g.edge_is_true = torch.full([g.num_edges], 1)


        #create fake graph with the needed # of nodes and edges
        fake = Data()
        fake.num_nodes = num_pad_nodes
        fake.num_edges = num_pad_edges
        fake.graph_is_true = torch.tensor([0])
        fake.node_is_true = torch.full([num_pad_nodes], 0)
        fake.edge_is_true = torch.full([num_pad_edges], 0)

        for key, value in real_graphs[0]:
            if not torch.is_tensor(value):
                continue

            if (key == "graph_is_true" or key == "node_is_true" or key == "edge_is_true"):
                continue

            dim = real_graphs[0].__cat_dim__(key, value)
            pad_shape = list(value.shape)

            if data.is_node_attr(key):
                pad_shape[dim] = num_pad_nodes
                pad_value = self.node_value
            elif data.is_edge_attr(key):
                pad_shape[dim] = num_pad_edges
                if key == "edge_index":
                    # Padding edges are self-loops on the first padding node
                    pad_value = 0
                else:
                    pad_value = self.edge_value
            else:
                continue

            pad_value = value.new_full(pad_shape, pad_value)
            fake[key] = torch.cat([pad_value], dim=dim)
        real_graphs.append(fake)
        new_data = Batch.from_data_list(real_graphs)

        if 'num_nodes' in new_data:
            new_data.num_nodes = self.max_num_nodes

        return new_data

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"max_num_nodes={self.max_num_nodes}, "
        s += f"max_num_edges={self.max_num_edges}, "
        s += f"node_value={self.node_value}, "
        s += f"edge_value={self.edge_value})"
        return s

