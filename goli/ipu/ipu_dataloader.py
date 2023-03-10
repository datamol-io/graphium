# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Callable, Iterable, Optional, List, Tuple
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from loguru import logger
from torch import Tensor

import torch
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.transforms import BaseTransform

from goli.ipu.ipu_utils import import_poptorch


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
    mode: "poptorch.DataLoaderMode" = "Sync"

    def set_kwargs(self):
        # Get the maximum number of nodes
        if self.max_num_nodes is not None:
            assert (
                self.max_num_nodes_per_graph is None
            ), "Cannot use `max_num_nodes` and `max_num_nodes_per_graph` simultaneously"
        elif self.max_num_nodes_per_graph is not None:
            assert (
                self.max_num_nodes is None
            ), "Cannot use `max_num_nodes` and `max_num_nodes_per_graph` simultaneously"
            self.max_num_nodes = self.max_num_nodes_per_graph * self.batch_size
        else:
            raise ValueError("Must provide either `max_num_nodes` or `max_num_nodes_per_graph`")

        # Get the maximum number of edges
        if self.max_num_edges is not None:
            assert (
                self.max_num_edges_per_graph is None
            ), "Cannot use `max_num_edges` and `max_num_edges_per_graph` simultaneously"
        elif self.max_num_edges_per_graph is not None:
            assert (
                self.max_num_edges is None
            ), "Cannot use `max_num_edges` and `max_num_edges_per_graph` simultaneously"
            self.max_num_edges = self.max_num_edges_per_graph * self.batch_size
        else:
            raise ValueError("Must provide either `max_num_nodes` or `max_num_nodes_per_graph`")

        # poptorch mode
        poptorch = import_poptorch()
        if isinstance(self.mode, str):
            if self.mode.lower() == "sync":
                self.mode = poptorch.DataLoaderMode.Sync
            elif self.mode.lower() == "async":
                self.mode = poptorch.DataLoaderMode.Async
            elif self.mode.lower() == "asyncrebatched":
                self.mode = poptorch.DataLoaderMode.AsyncRebatched
            else:
                raise ValueError(f"`{self.mode}` not a valid parameter.")


class CombinedBatchingCollator:
    """
    Collator object that manages the combined batch size defined as:

        combined_batch_size = batch_size * device_iterations
                             * replication_factor * gradient_accumulation

    This is intended to be used in combination with the poptorch.DataLoader
    """

    def __init__(
        self,
        batch_size: int,
        max_num_nodes: int,
        max_num_edges: int,
        dataset_max_nodes_per_graph: int,
        dataset_max_edges_per_graph: int,
        collate_fn: Optional[Callable] = None,
    ):
        """
        Parameters:
            batch_size: mini batch size used by the model
            max_num_nodes: Maximum number of nodes in the batched padded graph
            max_num_edges: Maximum number of edges in the batched padded graph
            dataset_max_nodes_per_graph: Maximum number of nodes per graph in the full dataset
            dataset_max_edges_per_graph: Maximum number of edges per graph in the full dataset
            collate_fn: Function used to collate (or batch) the single data or graphs together
        """
        super().__init__()
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.dataset_max_nodes_per_graph = dataset_max_nodes_per_graph
        self.dataset_max_edges_per_graph = dataset_max_edges_per_graph

    def __call__(self, batch: Batch) -> Batch:
        """
        padding option to pad each batch to be same size.

        Parameters:
            batch: The batch of pyg-graphs to be padded

        Returns:
            batch: The padded batch
        """

        # Sort the batch such that large graphs are paired with small graphs
        num_nodes = [b["features"].num_nodes for b in batch]
        packed_indices = hybrid_packing(num_nodes, batch_size=self.batch_size)
        packs = [[batch[idx] for idx in pack] for pack in packed_indices]

        # Loop all mini-batches within the global batch
        all_batches = []
        for pack in packs:
            if self.collate_fn != None:
                local_batch = self.collate_fn(pack)

            transform = Pad(
                max_num_nodes=self.max_num_nodes,
                max_num_edges=self.max_num_edges,
                dataset_max_nodes_per_graph=self.dataset_max_nodes_per_graph,
                dataset_max_edges_per_graph=self.dataset_max_edges_per_graph,
            )

            local_batch["features"] = transform(local_batch["features"])
            all_batches.append(local_batch)

        out_batch = {}

        # Stack tensors in the first dimension to allow IPUs to differentiate between local and global graph
        out_batch["labels"] = {
            key: torch.stack([this_batch["labels"][key] for this_batch in all_batches], 0)
            for key in all_batches[0]["labels"].keys()
        }
        out_graphs = [this_batch["features"] for this_batch in all_batches]
        stacked_features = deepcopy(out_graphs[0])
        for key, val in out_graphs[0].items():
            if isinstance(val, torch.Tensor):
                stacked_features[key] = torch.stack([this_graph[key] for this_graph in out_graphs], dim=0)

        # TODO: Make this more robust, instead of hard-coding the keys
        out_batch["features"] = stacked_features
        out_batch["_batch_idx"] = torch.as_tensor(range(len(all_batches)), dtype=torch.int32).unsqueeze(-1)
        for key in all_batches[0].keys():
            if key not in ("features", "labels", "_types_conversion"):
                out_batch[key] = [this_batch[key] for this_batch in all_batches]

        for data_key, data_val in out_batch.items():
            if isinstance(data_val, Batch):
                for sub_key, sub_val in data_val.items():
                    if isinstance(sub_val, Tensor) and sub_val.dtype == torch.int64:
                        out_batch[data_key][sub_key] = sub_val.to(torch.int32)

        return out_batch


def create_ipu_dataloader(
    dataset: Dataset,
    ipu_dataloader_options: IPUDataloaderOptions,
    ipu_options: Optional["poptorch.Options"] = None,
    batch_size: Optional[int] = 1,
    collate_fn=None,
    **kwargs,
) -> "poptorch.DataLoader":
    """
    Creates a poptorch.DataLoader for graph datasets
    Applies the mini-batching method of concatenating multiple graphs into a
    single graph with multiple disconnected subgraphs. See:
    https://pytorch-geometric.readthedocs.io/en/2.0.2/notes/batching.html

    Parameters:

        dataset: The torch_geometric.data.Dataset instance from which to
            load the graph examples for the IPU.
        ipu_dataloader_options: The options to initialize the Dataloader for IPU
        ipu_options: The poptorch.Options used by the
            poptorch.DataLoader. Will use the default options if not provided.
        batch_size: How many graph examples to load in each batch
            (default: 1).
        collate_fn: The function used to collate batches
        **kwargs (optional): Additional arguments of :class:`poptorch.DataLoader`.

    Returns:
        The dataloader
    """
    poptorch = import_poptorch()

    if ipu_options is None:
        # Create IPU default options
        ipu_options = poptorch.Options()

    # Define the collater function
    collater = CombinedBatchingCollator(
        batch_size,
        collate_fn=collate_fn,
        max_num_nodes=ipu_dataloader_options.max_num_nodes,
        max_num_edges=ipu_dataloader_options.max_num_edges,
        dataset_max_nodes_per_graph=dataset.max_num_nodes_per_graph,
        dataset_max_edges_per_graph=dataset.max_num_edges_per_graph,
    )

    # Get the global batch size
    num_nodes = np.asarray([dataset[ii]["features"].num_nodes for ii in range(len(dataset))])
    accum = ipu_options.Training.gradient_accumulation
    repli = ipu_options._values["replication_factor"]
    device_iter = ipu_options._values["device_iterations"]
    combined_batch_size = batch_size * accum * repli * device_iter

    # Estimate the packing size needed
    max_pack_size, max_pack_size_per_graph = 0, 0
    for _ in range(4):
        this_max_pack_size, this_max_pack_size_per_graph = estimate_max_pack_node_size(
            num_nodes=num_nodes,
            batch_size=batch_size,
            combined_batch_size=combined_batch_size,
        )
        max_pack_size = max(max_pack_size, this_max_pack_size)
        max_pack_size_per_graph = max(max_pack_size_per_graph, this_max_pack_size_per_graph)

    max_num_nodes = collater.max_num_nodes
    # Log the estimated pack size, with warnings if too big or too small
    logger.info(
        f"Estimating pack max_pack_size={max_pack_size} or max_pack_size_per_graph={max_pack_size_per_graph}"
    )
    logger.info(f"Provided `max_num_nodes={max_num_nodes}`")
    if max_pack_size > max_num_nodes - 10:
        logger.warning(
            f"The value of `max_num_nodes={max_num_nodes}` seems to be insufficient compared to `max_pack_size={max_pack_size}` and will likely crash"
        )
    elif max_pack_size < max_num_nodes - 20:
        logger.warning(
            f"The value of `max_num_nodes={max_num_nodes}` seems to be large compared to `max_pack_size={max_pack_size}` and will likely waste memory"
        )

    return poptorch.DataLoader(
        options=deepcopy(ipu_options),
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collater,
        async_options={"early_preload": True},  # TODO: keep?
        **kwargs,
    )


class Pad(BaseTransform):
    """
    Data transform that applies padding to enforce consistent tensor shapes.
    """

    def __init__(
        self,
        max_num_nodes: int,
        dataset_max_nodes_per_graph,
        dataset_max_edges_per_graph,
        max_num_edges: Optional[int] = None,
        node_value: float = 0,
        edge_value: float = 0,
    ):
        """
        Parameters:
            max_num_nodes: The maximum number of nodes for the total padded graph
            dataset_max_nodes_per_graph: the maximum number of nodes per graph in the dataset
            dataset_max_edges_per_graph: the maximum number of edges per graph in the dataset
            max_num_edges: The maximum number of edges for the total padded graph
            node_value: Value to add to the node padding
            edge_value: Value to add to the edge padding
        """
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.dataset_max_nodes_per_graph = dataset_max_nodes_per_graph
        self.dataset_max_edges_per_graph = dataset_max_edges_per_graph

        if max_num_edges:
            self.max_num_edges = max_num_edges
        else:
            # Assume fully connected graph
            self.max_num_edges = max_num_nodes * (max_num_nodes - 1)

        self.node_value = node_value
        self.edge_value = edge_value

    def validate(self, data):
        """
        Validates that the input graph does not exceed the constraints that:

          * the number of nodes must be <= max_num_nodes
          * the number of edges must be <= max_num_edges

        Returns:
            Tuple containing the number nodes and the number of edges
        """
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        assert num_nodes <= self.max_num_nodes, (
            f"Too many nodes. Graph has {num_nodes} nodes " f"and max_num_nodes is {self.max_num_nodes}."
        )

        assert num_edges <= self.max_num_edges, (
            f"Too many edges. Graph has {num_edges} edges defined "
            f"and max_num_edges is {self.max_num_edges}."
        )

        return num_nodes, num_edges

    def __call__(self, batch: Batch) -> Batch:
        """
        Pad the batch with a fake graphs that has the desired
        number of nodes and edges.
        """
        num_nodes, num_edges = self.validate(batch)
        num_pad_nodes = self.max_num_nodes - num_nodes
        num_pad_edges = self.max_num_edges - num_edges
        # Create a copy to update with padded features
        new_batch = deepcopy(batch)

        real_graphs = new_batch.to_data_list()

        for g in real_graphs:
            g.graph_is_true = torch.tensor([1], dtype=bool)
            g.node_is_true = torch.full([g.num_nodes], True, dtype=bool)
            g.edge_is_true = torch.full([g.num_edges], True, dtype=bool)

        # create fake graph with the needed # of nodes and edges
        fake = Data()
        fake.num_nodes = num_pad_nodes
        fake.num_edges = num_pad_edges
        fake.graph_is_true = torch.tensor([False], dtype=bool)
        fake.node_is_true = torch.full([num_pad_nodes], False, dtype=bool)
        fake.edge_is_true = torch.full([num_pad_edges], False, dtype=bool)

        for key, value in real_graphs[0]:
            if not torch.is_tensor(value):
                continue

            if key == "graph_is_true" or key == "node_is_true" or key == "edge_is_true":
                continue

            dim = real_graphs[0].__cat_dim__(key, value)
            pad_shape = list(value.shape)

            if batch.is_node_attr(key):
                pad_shape[dim] = num_pad_nodes
                pad_value = self.node_value
            elif batch.is_edge_attr(key):
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
        new_batch = Batch.from_data_list(real_graphs)

        if "num_nodes" in new_batch:
            new_batch.num_nodes = self.max_num_nodes

        return new_batch

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"max_num_nodes={self.max_num_nodes}, "
        s += f"max_num_edges={self.max_num_edges}, "
        s += f"node_value={self.node_value}, "
        s += f"edge_value={self.edge_value})"
        return s


class MolPack:
    """
    Class that keeps track of the number of atoms and indices that are added
    to each pack. Useful when doing packing, or other forms of smart batching.
    A pack is a batch, but with optimized memory consumption.
    """

    def __init__(self):
        self.num_nodes = 0
        self.num_graphs = 0
        self.average_atom = 0
        self.indices = []

    def add_mol(self, num_nodes: int, idx: int) -> "MolPack":
        """
        Add a molecule and it's index to the batch

        Parameters:
            num_nodes: Number of atoms of the new molecule

            idx: Index associated to the molecule
        """
        self.num_nodes += num_nodes
        self.num_graphs += 1
        self.average_atom = self.num_nodes / self.num_graphs
        self.indices.append(idx)
        return self

    def expected_atoms(self, remaining_mean_num_nodes: float, batch_size: int) -> float:
        """
        Given a desired batch size, and given the remaining mean number of
        atoms, find the expected number of atoms of the current batch when it is full

        Parameters:
            remaining_mean_num_nodes: Average number of atoms per molecule
                left to be sampled and distributed across tasks.

            batch_size: Desired batch size

        Returns:
            expected_atoms: The expected number of atoms in this batch if we
                sample randomly the remaining molecules.
        """
        return self.num_nodes + ((batch_size - self.num_graphs) * remaining_mean_num_nodes)

    def __repr__(self) -> str:
        """
        Print the main attributes of the current class
        """
        return f"{self.__class__.__name__}(m: {self.num_graphs},\ta: {self.num_nodes},\tav: {self.average_atom:.1f})"


def smart_packing(num_nodes: List[int], batch_size: int) -> List[List[int]]:
    """
    Simple and fast algorithm for packing graphs such that each batch has roughly the
    same number of atoms.
    Has for-loop scalability issues `O(num_graphs * ipu_batch_size)` = `O(num_graphs^2 / batch_size)`

    Parameters:
        num_nodes: List of the number of atoms per molecule for the entire global batch.
            Must be of length `batch_size * ipu_batch_size`.

        batch_size: The batch size per iteration, considering a single device and single
            forward pass.
            The global batch size is `batch_size * device_iterations * replication_factor * gradient_accumulation`

    Returns:
        packed_indices: A list of packs, each containing a list of indices, such that
            if we collect `num_nodes` from the indices, then each pack has roughly the
            same total number of atoms.
    """

    # Sort the list
    num_nodes = np.asarray(num_nodes)
    argsort_num_nodes = np.argsort(num_nodes)
    sorted_num_nodes = num_nodes[argsort_num_nodes]
    ipu_batch_size = int(len(num_nodes) / batch_size)
    sorted_num_nodes, initial_num_nodes = (
        sorted_num_nodes[:-ipu_batch_size],
        sorted_num_nodes[-ipu_batch_size:],
    )
    reverse_cumsum = np.sum(sorted_num_nodes) - np.cumsum(sorted_num_nodes) + sorted_num_nodes[-1]

    # Start with the largest element in separate packs
    mol_batches = [
        MolPack().add_mol(initial_num_nodes[-ii - 1], argsort_num_nodes[-ii - 1])
        for ii in range(ipu_batch_size)
    ]

    # Loop from smallest to largest molecule, and add each molecule to the pack with smallest expected sum
    for ii, num_atom in enumerate(sorted_num_nodes):
        remaining_mean = reverse_cumsum[ii] / (len(sorted_num_nodes) - ii)
        max_expected, idx_max_expected = 0, 0
        for jj, m in enumerate(mol_batches):
            if m.num_graphs >= batch_size:
                continue
            expected = m.num_nodes + (
                (batch_size - m.num_graphs) * remaining_mean
            )  # Faster than calling m.expected_atoms
            if expected > max_expected:
                max_expected = expected
                idx_max_expected = jj
        mol_batches[idx_max_expected].add_mol(num_atom, argsort_num_nodes[ii])

    packed_indices = [batch.indices for batch in mol_batches]

    return packed_indices


def fast_packing(num_nodes: List[int], batch_size: int) -> List[List[int]]:
    """
    Super fast algorithm for packing graphs such that each batch has roughly the
    same number of atoms. Not as good as `smart_packing` but
    faster and more scalable for-loop complexity of `O(batch_size)`.

    Parameters:
        num_nodes: List of the number of atoms per molecule for the entire global batch.
            Must be of length `batch_size * ipu_batch_size`.

        batch_size: The batch size per iteration, considering a single device and single
            forward pass.
            The global batch size is `batch_size * device_iterations * replication_factor * gradient_accumulation`

    Returns:
        packed_indices: A list of packs, each containing a list of indices, such that
            if we collect `num_nodes` from the indices, then each pack has roughly the
            same total number of atoms.
    """
    num_nodes = np.asarray(num_nodes)
    argsort_num_nodes = np.argsort(num_nodes)
    ipu_batch_size = int(len(num_nodes) / batch_size)

    groups = []
    for ii in range(batch_size):
        group = argsort_num_nodes[ii * ipu_batch_size : (ii + 1) * ipu_batch_size]
        np.random.shuffle(group)
        groups.append(group)

    packed_indices = np.stack(groups, axis=1).tolist()
    return packed_indices


def hybrid_packing(num_nodes: List[int], batch_size: int) -> List[List[int]]:
    """
    Uses a combination of the `smart_packing` `O(n^2)` on the most important data points,
    and the `fast_packing` `O(n)` on the average-sized data points.

    Depending on the expected complexity

    Parameters:
        num_nodes: List of the number of atoms per molecule for the entire global batch.
            Must be of length `batch_size * ipu_batch_size`.

        batch_size: The batch size per iteration, considering a single device and single
            forward pass.
            The global batch size is `batch_size * device_iterations * replication_factor * gradient_accumulation`

    Returns:
        packed_indices: A list of packs, each containing a list of indices, such that
            if we collect `num_nodes` from the indices, then each pack has roughly the
            same total number of atoms.
    """

    # Determine the parameters based on the complexity of the smart-packing.
    # The bigger the complexity, the more the `fast_packing` algorithm becomes
    # statistically powerful, and the more speed benefits it provides.
    smart_packing_complexity = len(num_nodes) ** 2 / batch_size
    if smart_packing_complexity < 1e4:
        return smart_packing(num_nodes=num_nodes, batch_size=batch_size)
    elif smart_packing_complexity < 1e6:
        big, small = 3, 6
    elif smart_packing_complexity < 1e8:
        big, small = 2, 4
    elif smart_packing_complexity < 1e10:
        big, small = 1, 2
    else:
        return fast_packing(num_nodes=num_nodes, batch_size=batch_size)

    # Small datasets benefit from smart-packing, without compute burden
    ipu_batch_size = int(len(num_nodes) / batch_size)
    if len(num_nodes) < (big + small) * ipu_batch_size:
        return smart_packing(num_nodes=num_nodes, batch_size=batch_size)

    # Sort the list
    num_nodes = np.asarray(num_nodes)
    argsort_num_nodes = np.argsort(num_nodes)

    # Smallest and biggest graphs are often outliers and will benefit from the `smart_packing`
    biggest_graphs = argsort_num_nodes[-big * ipu_batch_size :]
    smallest_graphs = argsort_num_nodes[: small * ipu_batch_size]
    big_n_small_graphs = np.concatenate([biggest_graphs, smallest_graphs])
    big_n_small_packs = smart_packing(num_nodes[big_n_small_graphs], batch_size=big + small)
    big_n_small_indices = [big_n_small_graphs[pack] for pack in big_n_small_packs]
    big_n_small_nodes = [num_nodes[pack] for pack in big_n_small_indices]

    # Medium graphs will be packed faster
    medium_graphs = argsort_num_nodes[small * ipu_batch_size : -big * ipu_batch_size]
    medium_packs = fast_packing(num_nodes[medium_graphs], batch_size=batch_size - big - small)
    medium_indices = [medium_graphs[pack] for pack in medium_packs]
    medium_nodes = [num_nodes[pack] for pack in medium_indices]

    # Pack the big/small with the medium in a smart way
    big_n_small_sort = np.argsort(np.sum(np.stack(big_n_small_nodes, axis=1), axis=0))
    medium_sort = np.argsort(np.sum(np.stack(medium_nodes, axis=1), axis=0))
    packed_indices = [
        np.concatenate([medium_indices[medium_sort[ii]], big_n_small_indices[big_n_small_sort[-ii]]])
        for ii in range(len(medium_indices))
    ]

    return packed_indices


def get_pack_sizes(packed_indices, num_nodes):
    """
    Get the number of atoms of each pack
    """
    pack_sums = []
    for pack in packed_indices:
        pack_sum = 0
        for idx in pack:
            pack_sum += num_nodes[idx]
        pack_sums.append(pack_sum)
    return pack_sums


def estimate_max_pack_node_size(num_nodes: Iterable[int], batch_size: int, combined_batch_size: int):
    """
    Estimate the value of `max_num_nodes`, which represents the maximum number of nodes
    needed in a batch to fit the data.

    Parameters:
        num_nodes: Number of nodes for all the graphs in the dataset
        batch_size: The regular batch size per IPU
        combined_batch_size: batch_size * device_iterations
                             * replication_factor * gradient_accumulation

    """

    # Estimate the packing size needed
    rand_indices = np.arange(len(num_nodes))
    np.random.shuffle(rand_indices)
    max_pack_size = 0
    for ii in range(0, len(num_nodes), combined_batch_size):
        this_indices = rand_indices[ii : ii + combined_batch_size]
        choice = num_nodes[this_indices]
        if len(choice) == combined_batch_size:
            packed_indices = hybrid_packing(choice, batch_size)
            max_pack_size = max(max_pack_size, max(get_pack_sizes(packed_indices, num_nodes[this_indices])))
    max_pack_size_per_graph = max_pack_size / batch_size

    return max_pack_size, max_pack_size_per_graph


def node_to_pack_indices_mask(
    packed_indices: Iterable[Iterable[int]], all_num_nodes: Iterable[int], max_pack_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a list of packed indices, and the number of nodes in each graph,
    return a tensor of shape (sum(all_num_nodes), 2) where the first column
    is the pack index, and the second column is the node index within the pack.

    Can be used to generate a dense packing of the nodes as follows:
    ```
    # node_features: A tensor of shape (num_nodes, num_node_features)
    # num_packs: The number of packs desired
    # max_nodes_per_pack: The maximum number of nodes per pack
    # dense_pack: A tensor of shape (num_packs, max_nodes_per_pack, num_node_features)

    dense_pack = torch.zeros([num_packs, max_nodes_per_pack, num_node_features])
    dense_pack[node_to_pack_idx[:, 0], node_to_pack_idx[:, 1]] = node_features
    ```

    This is useful when using a Transformer, to avoid wasteful padding when the
    the longest sequence is much longer than the average sequence length.

    Parameters:
        packed_indices: A list of lists of graph indices, where each sub-list
                        represents a pack of graphs
        all_num_nodes: The number of nodes in each graph
        max_pack_size: The maximum number of nodes per pack. If None, will be
                          infered from the provided packs.
                          Useful to determine the shape of the `pack_attn_mask`.

    Returns:
        node_to_pack_idx: A tensor of shape (num_nodes, 2) where the first column
                        is the pack index, and the second column is the node index within the pack.

        pack_attn_mask: A tensor of shape (num_packs, max_pack_size, max_pack_size),
                            that represents the attention masking for each pack,
                            such that the graphs in the pack are masked out from each other.
    """

    all_num_nodes = torch.as_tensor(all_num_nodes, dtype=torch.long)
    cumsum_num_nodes = torch.cumsum(all_num_nodes, dim=0)
    if max_pack_size is None:
        pack_sizes = get_pack_sizes(packed_indices, all_num_nodes)
        max_pack_size = max(pack_sizes)

    # Get the node indices associated to the packs, with 0 padding
    node_to_pack_idx = torch.zeros(sum(all_num_nodes), 2, dtype=torch.long)
    pack_attn_mask = []  # masks for the attention
    for ii, pack in enumerate(packed_indices):
        jj = 0  # Counter for the number of nodes in the pack
        this_pack_attn_mask = torch.ones((max_pack_size, max_pack_size), dtype=torch.bool)
        for graph_idx in pack:
            num_nodes = all_num_nodes[graph_idx]
            node_idx = torch.arange(
                cumsum_num_nodes[graph_idx] - num_nodes, cumsum_num_nodes[graph_idx]
            )
            this_pack_attn_mask[jj : jj + num_nodes, jj : jj + num_nodes] = False
            node_to_pack_idx[node_idx, 0] = ii
            node_to_pack_idx[node_idx, 1] = jj + torch.arange(num_nodes)
            jj += num_nodes
        pack_attn_mask.append(this_pack_attn_mask)
    pack_attn_mask = torch.stack(pack_attn_mask, dim=0)

    return node_to_pack_idx, pack_attn_mask
