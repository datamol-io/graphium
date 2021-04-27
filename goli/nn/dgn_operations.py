from typing import Optional
import torch
from torch import Tensor


EPS = 1e-8


def get_grad_of_pos(
    source_pos: Tensor, dest_pos: Tensor, dir_idx: int, temperature: Optional[float] = None
) -> Tensor:
    r"""
    Get the vector field associated to the gradient of the positional
    encoding.

    $$F_k = \nabla pos_k$$

    or, if a temperature $T$ is provided

    $$F_k = softmax((\nabla pos_k)^+) - softmax((\nabla pos_k)^-)$$

    Where $F_k$ is the *k-th* directional field associated to the $k-th$ positional
    encoding.

    Parameters:

        source_pos: The positional encoding at the source node, used to compute the directional field
        dest_pos: The positional encoding at the destination node, used to compute the directional field
        h_in: The input features of the layer, before any operation.
        dir_idx: The index of the positional encoding ($k$ in the equation above)
        temperature: The temperature to use in the softmax of the directional field.
            If `None`, then the softmax is not applied on the field

    """

    grad = source_pos[:, :, dir_idx] - dest_pos[:, :, dir_idx]
    if temperature is not None:
        grad_plus = torch.nn.Softmax(1)(temperature * torch.relu(grad))
        grad_minus = -torch.nn.Softmax(1)(temperature * torch.relu(-grad))
        grad = grad_plus + grad_minus

    return grad


def aggregate_dir_smooth(
    h: Tensor,
    source_pos: Tensor,
    dest_pos: Tensor,
    h_in: Tensor,
    dir_idx: int,
    temperature: Optional[float] = None,
    **kwargs,
) -> Tensor:
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
        dir_idx: The index of the positional encoding ($k$ in the equation above)
        temperature: The temperature to use in the softmax of the directional field.
            If `None`, then the softmax is not applied on the field

    Returns:
        h_mod: The aggregated features $y^{(l)}$

    """
    grad = get_grad_of_pos(source_pos=source_pos, dest_pos=dest_pos, dir_idx=dir_idx, temperature=temperature)
    h_mod = h * (grad.abs() / (torch.sum(grad.abs(), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    return torch.sum(h_mod, dim=1)


def aggregate_dir_dx_abs(
    h: Tensor,
    source_pos: Tensor,
    dest_pos: Tensor,
    h_in: Tensor,
    dir_idx: int,
    temperature: Optional[float] = None,
    **kwargs,
) -> Tensor:
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
        dir_idx: The index of the positional encoding ($k$ in the equation above)
        temperature: The temperature to use in the softmax of the directional field.
            If `None`, then the softmax is not applied on the field

    Returns:
        h_mod: The aggregated features $y^{(l)}$

    """
    return torch.abs(aggregate_dir_dx_no_abs(h, source_pos, dest_pos, h_in, dir_idx, temperature, **kwargs))


def aggregate_dir_dx_no_abs(
    h: Tensor,
    source_pos: Tensor,
    dest_pos: Tensor,
    h_in: Tensor,
    dir_idx: int,
    temperature: Optional[float] = None,
    **kwargs,
) -> Tensor:
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
        dir_idx: The index of the positional encoding ($k$ in the equation above)
        temperature: The temperature to use in the softmax of the directional field.
            If `None`, then the softmax is not applied on the field

    Returns:
        h_mod: The aggregated features $y^{(l)}$

    """
    grad = get_grad_of_pos(source_pos=source_pos, dest_pos=dest_pos, dir_idx=dir_idx, temperature=temperature)
    dir_weight = (grad / (torch.sum(grad.abs(), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    h_mod = h * dir_weight

    h_dx = torch.sum(h_mod, dim=1)
    h_self = -torch.sum(dir_weight, dim=1) * h_in

    # In case h_in has more parameters than h (for example when concatenating edges),
    # the derivative is only computed for the features contained in h_in.
    h_dx[..., : h_in.shape[-1]] = h_dx[..., : h_in.shape[-1]] + h_self
    return h_dx


def aggregate_dir_dx_abs_balanced(
    h: Tensor,
    source_pos: Tensor,
    dest_pos: Tensor,
    h_in: Tensor,
    dir_idx: int,
    temperature: Optional[float] = None,
    **kwargs,
) -> Tensor:
    r"""
    The aggregation is the same as `aggregate_dir_dx_no_abs`, but the positive and
    negative parts of the field are normalized separately.


    Parameters:

        h: The features to aggregate $h^{(l)}$
        source_pos: The positional encoding at the source node, used to compute the directional field
        dest_pos: The positional encoding at the destination node, used to compute the directional field
        h_in: The input features of the layer, before any operation.
        dir_idx: The index of the positional encoding ($k$ in the equation above)
        temperature: The temperature to use in the softmax of the directional field.
            If `None`, then the softmax is not applied on the field

    Returns:
        h_mod: The aggregated features $y^{(l)}$

    """
    grad = get_grad_of_pos(source_pos=source_pos, dest_pos=dest_pos, dir_idx=dir_idx, temperature=temperature)
    eig_front = torch.relu(grad) / (torch.sum(torch.relu(grad), keepdim=True, dim=1) + EPS)
    eig_back = torch.relu(-grad) / (torch.sum(torch.relu(-grad), keepdim=True, dim=1) + EPS)

    dir_weight = (eig_front.unsqueeze(-1) + eig_back.unsqueeze(-1)) / 2
    h_mod = h * dir_weight

    h_dx = torch.sum(h_mod, dim=1)
    h_self = -torch.sum(dir_weight, dim=1) * h_in

    # In case h_in has more parameters than h (for example when concatenating edges),
    # the derivative is only computed for the features contained in h_in.
    h_dx[..., : h_in.shape[-1]] = h_dx[..., : h_in.shape[-1]] + h_self
    return torch.abs(h_dx)


def aggregate_dir_forward(
    h: Tensor,
    source_pos: Tensor,
    dest_pos: Tensor,
    h_in: Tensor,
    dir_idx: int,
    temperature: Optional[float] = None,
    **kwargs,
) -> Tensor:
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
        dir_idx: The index of the positional encoding ($k$ in the equation above)
        temperature: The temperature to use in the softmax of the directional field.
            If `None`, then the softmax is not applied on the field

    Returns:
        h_mod: The aggregated features $y^{(l)}$
    """
    grad = get_grad_of_pos(source_pos=source_pos, dest_pos=dest_pos, dir_idx=dir_idx, temperature=temperature)
    eig_front = torch.relu(grad) / (torch.sum(torch.relu(grad), keepdim=True, dim=1) + EPS)
    h_mod = h * eig_front.unsqueeze(-1)
    return torch.sum(h_mod, dim=1)


def aggregate_dir_backward(
    h: Tensor,
    source_pos: Tensor,
    dest_pos: Tensor,
    h_in: Tensor,
    dir_idx: int,
    temperature: Optional[float] = None,
    **kwargs,
) -> Tensor:
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
        dir_idx: The index of the positional encoding ($k$ in the equation above)
        temperature: The temperature to use in the softmax of the directional field.
            If `None`, then the softmax is not applied on the field

    Returns:
        h_mod: The aggregated features $y^{(l)}$
    """
    return aggregate_dir_forward(h, -source_pos, -dest_pos, h_in, dir_idx, temperature, **kwargs)


DGN_AGGREGATORS = {
    "smooth": aggregate_dir_smooth,
    "dx_abs": aggregate_dir_dx_abs,
    "dx_no_abs": aggregate_dir_dx_no_abs,
    "dx_abs_balanced": aggregate_dir_dx_abs_balanced,
    "forward": aggregate_dir_forward,
    "backward": aggregate_dir_backward,
}
