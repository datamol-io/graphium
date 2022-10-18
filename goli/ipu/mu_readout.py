import mup

class PopReadout(mup.MuReadout):

    """
    PopTorch-compatible replacement for `mup.MuReadout`

    Not quite a drop-in replacement for `mup.MuReadout` - you need to specify
    `base_width`.

    Set `base_width` to width of base model passed to `mup.set_base_shapes`
    to get same results on IPU and CPU. Should still "work" with any other
    value, but won't give the same results as CPU
    """

    def __init__(self, in_features, *args, base_width=None, **kwargs):

        if base_width is None:
            raise ValueError("base_width must be specified in PopReadout")

        self._base_width = base_width

        self._absolute_width = float(in_features)

        super().__init__(in_features, *args, **kwargs)

    def width_mult(self):
        return self._absolute_width / self._base_width
