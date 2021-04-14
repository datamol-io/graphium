from typing import List
import torch
from torch import Tensor


EPS = 1e-8


def aggregate_dir_smooth(h: Tensor, source_pos: Tensor, dest_pos: Tensor, h_in: Tensor, dir_idx: int, **kwargs):
    r"""
    The aggregation is the following:

    $$y^{(l)} = |\hat{F}_k| h^{(l)}$$

    - $\hat{F}^+_k$ is the normalized directional field *k-th* directional field $F_k$
    - $y^{(l)}$ is the returned aggregated result at the *l-th* layer.
    - $h^{(l)}$ is the node features at the *l-th* layer.

    Parameters:

        h: The features to aggregate $h^{(l)}$
        source_pos: The positional encoding at the source node, used to compute the directional field
        dest_pos: The positional encoding at the destination node, used to compute the directional field
        h_in: The input features of the layer, before any operation.

    Returns:
        h_mod: The aggregated features $y^{(l)}$

    """
    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    h_mod = h * (grad.abs() / (torch.sum(grad.abs(), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    return torch.sum(h_mod, dim=1)


def aggregate_dir_softmax(h: Tensor, source_pos: Tensor, dest_pos: Tensor, h_in: Tensor, dir_idx: int, alpha: float, **kwargs):
    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    h_mod = h * torch.nn.Softmax(1)(alpha * (grad.abs()).unsqueeze(-1))
    return torch.sum(h_mod, dim=1)


def aggregate_dir_dx_abs(h: Tensor, source_pos: Tensor, dest_pos: Tensor, h_in: Tensor, dir_idx: int, **kwargs):
    r"""
    The aggregation is the following:

    $$y^{(l)} = |B_{dx}^k h^{(l)}|$$

    $$B_{dx}^k = \hat{F}_k - diag \left(\sum_j{\hat{F}_{k_{(:, j)}}} \right)$$

    - $\hat{F}^+_k$ is the normalized positive component of the directional field *k-th* directional field $F_k$
    - $y^{(l)}$ is the returned aggregated result at the *l-th* layer.
    - $h^{(l)}$ is the node features at the *l-th* layer.

    Parameters:

        h: The features to aggregate $h^{(l)}$
        source_pos: The positional encoding at the source node, used to compute the directional field
        dest_pos: The positional encoding at the destination node, used to compute the directional field
        h_in: The input features of the layer, before any operation.

    Returns:
        h_mod: The aggregated features $y^{(l)}$

    """
    return torch.abs(aggregate_dir_dx_no_abs(h, source_pos, dest_pos, h_in, dir_idx, **kwargs))


def aggregate_dir_dx_no_abs(h: Tensor, source_pos: Tensor, dest_pos: Tensor, h_in: Tensor, dir_idx: int, **kwargs):
    r"""
    The aggregation is the following:

    $$y^{(l)} = B_{dx}^k h^{(l)}$$

    $$B_{dx}^k = \hat{F}_k - diag \left(\sum_j{\hat{F}_{k_{(:, j)}}} \right)$$

    - $\hat{F}^+_k$ is the normalized positive component of the directional field *k-th* directional field $F_k$
    - $y^{(l)}$ is the returned aggregated result at the *l-th* layer.
    - $h^{(l)}$ is the node features at the *l-th* layer.

    Parameters:

        h: The features to aggregate $h^{(l)}$
        source_pos: The positional encoding at the source node, used to compute the directional field
        dest_pos: The positional encoding at the destination node, used to compute the directional field
        h_in: The input features of the layer, before any operation.

    Returns:
        h_mod: The aggregated features $y^{(l)}$

    """
    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    dir_weight = (grad / (torch.sum(grad.abs(), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    h_mod = h * dir_weight
    return torch.sum(h_mod, dim=1) - torch.sum(dir_weight, dim=1) * h_in


def aggregate_dir_dx_abs_balanced(h: Tensor, source_pos: Tensor, dest_pos: Tensor, h_in: Tensor, dir_idx: int, **kwargs):
    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    eig_front = torch.relu(grad) / (torch.sum(torch.relu(grad), keepdim=True, dim=1) + EPS)
    eig_back = torch.relu(-grad) / (torch.sum(torch.relu(-grad), keepdim=True, dim=1) + EPS)

    dir_weight = (eig_front.unsqueeze(-1) + eig_back.unsqueeze(-1)) / 2
    h_mod = h * dir_weight
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(dir_weight, dim=1) * h_in)


def aggregate_dir_forward(h: Tensor, source_pos: Tensor, dest_pos: Tensor, h_in: Tensor, dir_idx: int, **kwargs):
    r"""
    The aggregation is the following:

    $$y^{(l)} = \hat{F}^+_k h^{(l)}$$

    - $\hat{F}^+_k$ is the normalized positive component of the directional field *k-th* directional field $F_k$
    - $y^{(l)}$ is the returned aggregated result at the *l-th* layer.
    - $h^{(l)}$ is the node features at the *l-th* layer.

    Parameters:

        h: The features to aggregate $h^{(l)}$
        source_pos: The positional encoding at the source node, used to compute the directional field
        dest_pos: The positional encoding at the destination node, used to compute the directional field
        h_in: The input features of the layer, before any operation.

    Returns:
        h_mod: The aggregated features $y^{(l)}$
    """
    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    eig_front = torch.relu(grad) / (torch.sum(torch.relu(grad), keepdim=True, dim=1) + EPS)
    return h * eig_front.unsqueeze(-1)


def aggregate_dir_backward(h: Tensor, source_pos: Tensor, dest_pos: Tensor, h_in: Tensor, dir_idx: int, **kwargs):
    r"""
    The aggregation is the following:

    $$y^{(l)} = \hat{F}^-_k h^{(l)}$$

    - $\hat{F}^-_k$ is the normalized positive component of the directional field *k-th* directional field $F_k$
    - $y^{(l)}$ is the returned aggregated result at the *l-th* layer.
    - $h^{(l)}$ is the node features at the *l-th* layer.

    Parameters:

        h: The features to aggregate $h^{(l)}$
        source_pos: The positional encoding at the source node, used to compute the directional field
        dest_pos: The positional encoding at the destination node, used to compute the directional field
        h_in: The input features of the layer, before any operation.

    Returns:
        h_mod: The aggregated features $y^{(l)}$
    """
    return aggregate_dir_forward(h, -source_pos, -dest_pos, h_in, dir_idx, **kwargs)

