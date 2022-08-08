# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, Batch


'''
add is_true_graph, is_true_node, is_true_edge attributes
now assume the input is not a pyg graph but a pyg Batch object
'''
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
            f"and max_num_edges is {self.max_num_edges}."

        assert num_edges <= self.max_num_edges, \
            f"Too many edges. Graph has {num_edges} edges defined "\
            f"and max_num_edges is {self.max_num_edges}."

        return num_nodes, num_edges

    def __call__(self, data):
        #! andy: add is_true_graph, is_true_node, is_true_edge
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


        # args = () if self.include_keys is None else self.include_keys

        # for key, value in data(*args):
        #     if not torch.is_tensor(value):
        #         continue

        #     dim = data.__cat_dim__(key, value)
        #     pad_shape = list(value.shape)

        #     if data.is_node_attr(key):
        #         pad_shape[dim] = num_pad_nodes
        #         if (key == 'batch'):
        #             pad_value = num_graphs + 1
        #         else:
        #             pad_value = self.node_value
        #     elif data.is_edge_attr(key):
        #         pad_shape[dim] = num_pad_edges

        #         if key == "edge_index":
        #             # Padding edges are self-loops on the first padding node
        #             pad_value = num_nodes
        #         else:
        #             pad_value = self.edge_value
        #     else:
        #         continue

        #     pad_value = value.new_full(pad_shape, pad_value)
        #     new_data[key] = torch.cat([value, pad_value], dim=dim)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"max_num_nodes={self.max_num_nodes}, "
        s += f"max_num_edges={self.max_num_edges}, "
        s += f"node_value={self.node_value}, "
        s += f"edge_value={self.edge_value})"
        return s


# class Pad(BaseTransform):
#     """
#     Data transform that applies padding to enforce consistent tensor shapes.
#     """

#     def __init__(self,
#                  max_num_nodes: int,
#                  max_num_edges: Optional[int] = None,
#                  node_value: Optional[float] = None,
#                  edge_value: Optional[float] = None,
#                  include_keys: Optional[Union[List[str], Tuple[str]]] = None):
#         """
#         :param max_num_nodes (int): The maximum number of nodes
#         :param max_num_edges (optional): The maximum number of edges.
#         """
#         super().__init__()
#         self.max_num_nodes = max_num_nodes

#         if max_num_edges:
#             self.max_num_edges = max_num_edges
#         else:
#             # Assume fully connected graph
#             self.max_num_edges = max_num_nodes * (max_num_nodes - 1)

#         self.node_value = 0.0 if node_value is None else node_value
#         self.edge_value = 0.0 if edge_value is None else edge_value
#         self.include_keys = include_keys

#     def validate(self, data):
#         """
#         Validates that the input graph does not exceed the constraints that:

#           * the number of nodes must be <= max_num_nodes
#           * the number of edges must be <= max_num_edges

#         :returns: Tuple containing the number nodes and the number of edges
#         """
#         num_nodes = data.num_nodes
#         num_edges = data.num_edges

#         assert num_nodes <= self.max_num_nodes, \
#             f"Too many nodes. Graph has {num_nodes} nodes "\
#             f"and max_num_edges is {self.max_num_edges}."

#         assert num_edges <= self.max_num_edges, \
#             f"Too many edges. Graph has {num_edges} edges defined "\
#             f"and max_num_edges is {self.max_num_edges}."

#         return num_nodes, num_edges

#     def __call__(self, data):
#         num_nodes, num_edges = self.validate(data)
#         num_pad_nodes = self.max_num_nodes - num_nodes
#         num_pad_edges = self.max_num_edges - num_edges

#         # Create a copy to update with padded features
#         new_data = deepcopy(data)
#         args = () if self.include_keys is None else self.include_keys

#         for key, value in data(*args):
#             if not torch.is_tensor(value):
#                 continue

#             dim = data.__cat_dim__(key, value)
#             pad_shape = list(value.shape)

#             if data.is_node_attr(key):
#                 pad_shape[dim] = num_pad_nodes
#                 pad_value = self.node_value
#             elif data.is_edge_attr(key):
#                 pad_shape[dim] = num_pad_edges

#                 if key == "edge_index":
#                     # Padding edges are self-loops on the first padding node
#                     pad_value = num_nodes
#                 else:
#                     pad_value = self.edge_value
#             else:
#                 continue

#             pad_value = value.new_full(pad_shape, pad_value)
#             new_data[key] = torch.cat([value, pad_value], dim=dim)

#         if 'num_nodes' in new_data:
#             new_data.num_nodes = self.max_num_nodes

#         return new_data

#     def __repr__(self) -> str:
#         s = f"{self.__class__.__name__}("
#         s += f"max_num_nodes={self.max_num_nodes}, "
#         s += f"max_num_edges={self.max_num_edges}, "
#         s += f"node_value={self.node_value}, "
#         s += f"edge_value={self.edge_value})"
#         return s


