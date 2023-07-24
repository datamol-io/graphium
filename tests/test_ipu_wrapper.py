import unittest as ut

import torch
from torch_geometric.data import Batch, Data

from graphium.ipu.ipu_wrapper import PyGArgsParser

def _make_fake_graph(num_nodes, num_edges, hidden_dim=16):

    x = torch.rand(num_nodes, hidden_dim)
    edge_index = torch.randint(size=(2, num_edges), high=num_nodes)

    return Data(x=x, edge_index=edge_index)

class test_ArgParser(ut.TestCase):

    g1 = _make_fake_graph(num_nodes=6, num_edges=14)
    g2 = _make_fake_graph(num_nodes=7, num_edges=18)

    arg_parser = PyGArgsParser()
    
    def test_arg_parser(self):

        tensor_iterator = self.arg_parser.yieldTensors(self.g1)
        reconstructed_data = self.arg_parser.reconstruct(self.g1, tensor_iterator)

        self.assertTrue(isinstance(reconstructed_data, Data))
        self.assertTrue(reconstructed_data.keys() == self.g1.keys())
        for k in self.g1.keys():
            self.assertTrue(reconstructed_data[k].eq(self.g1[k]).all())

        batch = Batch.from_data_list([self.g1, self.g2])

        tensor_iterator = self.arg_parser.yieldTensors(batch)
        reconstructed_data = self.arg_parser.reconstruct(batch, tensor_iterator)

        self.assertTrue(isinstance(reconstructed_data, Data))
        self.assertTrue(reconstructed_data.keys() == batch.keys())
        for k in batch.keys():
            self.assertTrue(reconstructed_data[k].eq(batch[k]).all())
