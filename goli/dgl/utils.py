import torch
import dgl


class DGLCollate:
    def __init__(self, device, siamese=False):
        self.device = device
        self.siamese = siamese

    def __call__(self, samples):
        # The input `samples` is a list of pairs
        #  (graph, label).
        graphs, labels = map(list, zip(*samples))
        if (isinstance(graphs[0], (tuple, list))) and (len(graphs[0]) == 1):
            graphs = [g[0] for g in graphs]

        if self.siamese:
            graphs1, graphs2 = zip(*graphs)
            batched_graph = (dgl.batch(graphs1).to(self.device), dgl.batch(graphs2).to(self.device))
        else:
            batched_graph = dgl.batch(graphs).to(self.device)
        return batched_graph, torch.stack(labels, dim=0)
