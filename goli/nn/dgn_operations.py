from typing import List
import torch
from torch import nn
from functools import partial

from goli.nn.pna_operations import PNA_AGGREGATORS

EPS = 1e-8


def aggregate_dir_smooth(h, source_pos, dest_pos, h_in, dir_idx, **kwargs):
    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    h_mod = h * (grad.abs() / (torch.sum(grad.abs(), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    return torch.sum(h_mod, dim=1)


def aggregate_dir_softmax(h, source_pos, dest_pos, h_in, dir_idx, alpha, **kwargs):
    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    h_mod = h * torch.nn.Softmax(1)(alpha * (grad.abs()).unsqueeze(-1))
    return torch.sum(h_mod, dim=1)


def aggregate_dir_dx_abs(h, source_pos, dest_pos, h_in, dir_idx, **kwargs):
    return torch.abs(aggregate_dir_dx_no_abs(h, source_pos, dest_pos, h_in, dir_idx, **kwargs))


def aggregate_dir_dx_no_abs(h, source_pos, dest_pos, h_in, dir_idx, **kwargs):
    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    dir_weight = (grad / (torch.sum(grad.abs(), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    h_mod = h * dir_weight
    return torch.sum(h_mod, dim=1) - torch.sum(dir_weight, dim=1) * h_in


def aggregate_dir_dx_abs_balanced(h, source_pos, dest_pos, h_in, dir_idx, **kwargs):
    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    eig_front = torch.relu(grad) / (torch.sum(torch.relu(grad), keepdim=True, dim=1) + EPS)
    eig_back = torch.relu(-grad) / (torch.sum(torch.relu(-grad), keepdim=True, dim=1) + EPS)

    dir_weight = (eig_front.unsqueeze(-1) + eig_back.unsqueeze(-1)) / 2
    h_mod = h * dir_weight
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(dir_weight, dim=1) * h_in)


def aggregate_dir_forward(h, source_pos, dest_pos, h_in, dir_idx, **kwargs):
    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    weight_front = torch.relu(grad) / (torch.sum(torch.relu(grad), keepdim=True, dim=1) + EPS)
    return h * weight_front.unsqueeze(-1)


def aggregate_dir_backward(h, source_pos, dest_pos, h_in, dir_idx, **kwargs):
    return aggregate_dir_forward(h, -source_pos, -dest_pos, h_in, dir_idx, **kwargs)


def parse_dgn_aggregator(aggregators_name: List[str]):

    aggregators = []

    for agg_name in aggregators_name:
        agg_name = agg_name.lower()
        this_agg = None

        # Get the aggregator from PNA if not a directional aggregation
        if agg_name in PNA_AGGREGATORS.keys():
            this_agg = PNA_AGGREGATORS[agg_name]

        # If the directional, get the right aggregator
        elif "dir" == agg_name[:3]:
            agg_split = agg_name.split("/", maxsplit=1)[0]
            agg_dir, agg_fn_name = agg_split[0], agg_split[1]
            dir_idx = int(agg_dir[3:])

            # Initialize the functions
            if agg_fn_name == "smooth":
                this_agg = partial(aggregate_dir_smooth, dir_idx=dir_idx)
            elif agg_fn_name == "softmax":
                alpha = float(agg_split[2])
                this_agg = partial(aggregate_dir_softmax, dir_idx=dir_idx, alpha=alpha)
            elif agg_fn_name == "dx_abs":
                this_agg = partial(aggregate_dir_dx_abs, dir_idx=dir_idx)
            elif agg_fn_name == "dx_no_abs":
                this_agg = partial(aggregate_dir_dx_no_abs, dir_idx=dir_idx)
            elif agg_fn_name == "dx_abs_balanced":
                this_agg = partial(aggregate_dir_dx_abs_balanced, dir_idx=dir_idx)
            elif agg_fn_name == "forward":
                this_agg = partial(aggregate_dir_forward, dir_idx=dir_idx)
            elif agg_fn_name == "backward":
                this_agg = partial(aggregate_dir_backward, dir_idx=dir_idx)

        if this_agg is None:
            raise ValueError(f"aggregator `{agg_name}` not a valid choice.")

        aggregators.append(this_agg)

    return aggregators
