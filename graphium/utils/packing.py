# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import List, Tuple, Iterable, Optional
import numpy as np
import torch


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

    packed_indices = np.stack(
        [
            np.random.permutation(argsort_num_nodes[ii * ipu_batch_size : (ii + 1) * ipu_batch_size])
            for ii in range(batch_size)
        ],
        axis=0,
    ).T.tolist()
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
    elif smart_packing_complexity < 1e5:
        big, small = 3, 6
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
    dense_pack[pack_from_node_idx[:, 0], pack_from_node_idx[:, 1]] = node_features
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
        pack_from_node_idx: A tensor of shape (num_nodes, 2) where the first column
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
    pack_from_node_idx = torch.zeros(sum(all_num_nodes), 2, dtype=torch.long)
    pack_attn_mask = []  # masks for the attention
    for ii, pack in enumerate(packed_indices):
        jj = 0  # Counter for the number of nodes in the pack
        this_pack_attn_mask = torch.ones((max_pack_size, max_pack_size), dtype=torch.bool)
        for graph_idx in pack:
            num_nodes = all_num_nodes[graph_idx]
            node_idx = torch.arange(cumsum_num_nodes[graph_idx] - num_nodes, cumsum_num_nodes[graph_idx])
            this_pack_attn_mask[jj : jj + num_nodes, jj : jj + num_nodes] = False
            pack_from_node_idx[node_idx, 0] = ii
            pack_from_node_idx[node_idx, 1] = jj + torch.arange(num_nodes)
            jj += num_nodes
        pack_attn_mask.append(this_pack_attn_mask)
    pack_attn_mask = torch.stack(pack_attn_mask, dim=0)

    return pack_from_node_idx, pack_attn_mask
