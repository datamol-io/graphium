from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_scatter import scatter_add


def to_sparse_batch(x: Tensor, mask_idx: Tensor):
    """
    Reverse function of `to_dense_batch`
    """
    return torch.index_select(x.reshape(-1, x.shape[-1]), 0, mask_idx)


def to_dense_batch(
    x: Tensor,
    batch: Optional[Tensor] = None,
    fill_value: float = 0.0,
    max_num_nodes_per_graph: Optional[int] = None,
    batch_size: Optional[int] = None,
    drop_nodes_last_graph=False,
) -> Tuple[Tensor, Tensor]:
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
    N_{\max}}` is returned, holding information about the existence of
    fake-nodes in the dense representation.

    Parameters:
        x: Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch: Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        fill_value: The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        max_num_nodes_per_graph: The size of the output node dimension.
            (default: :obj:`None`)
        batch_size: The batch size. (default: :obj:`None`)
        drop_nodes_last_graph: Whether to drop the nodes of the last graphs that exceed
            the `max_num_nodes_per_graph`. Useful when the last graph is a padding.

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)
    """
    if batch is None and max_num_nodes_per_graph is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    if max_num_nodes_per_graph is None:
        max_num_nodes_per_graph = int(num_nodes.max())

    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes_per_graph)

    size = [batch_size * max_num_nodes_per_graph] + list(x.size())[1:]

    ##### CHANGES FROM PYG #####

    # `torch.new_full` not supported by poptorch
    out = torch.full(size, fill_value, dtype=x.dtype, device=x.device)
    # out = x.new_full(size, fill_value)    # TODO: Uncomment this line with the new SDK

    # In case the last graph represents padding. Drop the overflowing nodes.
    if drop_nodes_last_graph:
        num_nodes = num_nodes[:-1]
        idx[idx >= size[0]] = size[0] - 1

    # Raise error if num_nodes > max_num_nodes
    assert (
        num_nodes <= max_num_nodes_per_graph
    ).all(), f"Encountered graphs with {num_nodes.max()} nodes, greater than `max_num_nodes = {max_num_nodes_per_graph}`"

    ##### END CHANGES FROM PYG #####

    out[idx] = x
    out = out.view([batch_size, max_num_nodes_per_graph] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes_per_graph, dtype=torch.bool, device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes_per_graph)

    return out, mask, idx  # Added `idx` as a return
