##### Code adapted from the `mup` package from Microsoft https://github.com/microsoft/mup

from torch.nn import Linear
from torch.nn.modules.conv import _ConvNd
from mup import get_shapes, assert_hidden_size_inf, MuReadout, rescale_linear_bias, save_base_shapes
from mup.shape import _zip_infshape_dict, _extract_shapes

from graphium.nn.base_layers import MuReadoutGraphium


def apply_infshapes(model, infshapes):
    """
    Modified from the regular `mup.apply_infshapes` by explicitly adding `base_dim` to the `MuReadoutGraphium`.
    This allows the code to work on IPUs.
    """
    for name, p in model.named_parameters():
        p.infshape = infshapes[name]
    for _, module in model.named_modules():
        if isinstance(module, MuReadoutGraphium):
            module.base_width = module.weight.infshape[-1].base_dim


def set_base_shapes(model, base, rescale_params=True, delta=None, savefile=None, do_assert=True):
    """Sets the `p.infshape` attribute for each parameter `p` of `model`.

    Code taken from the `mup` package from Microsoft https://github.com/microsoft/mup.
    No change except in the `apply_inf_shapes`, using the one from Graphium instead of `mup`

    Inputs:
        model: nn.Module instance
        base: The base model.
            Can be nn.Module, a dict of shapes, a str, or None.
            If None, then defaults to `model`
            If str, then treated as filename for yaml encoding of a dict of base shapes.
        rescale_params:
            assuming the model is initialized using the default pytorch init (or
            He initialization etc that scale the same way with fanin): If True
            (default), rescales parameters to have the correct (μP) variances.
        do_assert:
    Output:
        same object as `model`, after setting the `infshape` attribute of each parameter.
    """
    if base is None:
        base = model
    base_shapes = _extract_shapes(base)
    if delta is not None:
        delta_shapes = _extract_shapes(delta)
        base_shapes = _zip_infshape_dict(base_shapes, delta_shapes)
    shapes = get_shapes(model)
    infshapes = _zip_infshape_dict(base_shapes, shapes)
    if savefile is not None:
        save_base_shapes(infshapes, savefile)
    apply_infshapes(model, infshapes)
    if do_assert:
        assert_hidden_size_inf(model)
    if rescale_params:
        for name, module in model.named_modules():
            if isinstance(module, MuReadout):
                module._rescale_parameters()
            elif isinstance(module, (Linear, _ConvNd)):
                rescale_linear_bias(module)
    return model
