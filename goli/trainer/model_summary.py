from typing import List, Tuple
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary


class ModelSummaryExtended(ModelSummary):
    r"""
    Generates a summary of all layers in a :class:`~pytorch_lightning.core.lightning.LightningModule`.
    The summary is extended to allow different levels.

    Args:
        model: The model to summarize (also referred to as the root module)
        mode: Can be one of

             - `top` (default): only the top-level modules will be recorded (the children of the root module)
             - `full`: summarizes all layers and their submodules in the root module
             - `top1`, `top2`, ..., `top11`: summarizes the k-top-level modules

    The string representation of this summary prints a table with columns containing
    the name, type and number of parameters for each layer.

    The root module may also have an attribute ``example_input_array`` as shown in the example below.
    If present, the root module will be called with it as input to determine the
    intermediate input- and output shapes of all layers. Supported are tensors and
    nested lists and tuples of tensors. All other types of inputs will be skipped and show as `?`
    in the summary table. The summary will also display `?` for layers not used in the forward pass.

    """

    MODE_TOP = "top"
    MODE_TOP2 = "top2"
    MODE_TOP3 = "top3"
    MODE_TOP4 = "top4"
    MODE_TOP5 = "top5"
    MODE_TOP6 = "top6"
    MODE_TOP7 = "top7"
    MODE_TOP8 = "top8"
    MODE_TOP9 = "top9"
    MODE_TOP10 = "top10"
    MODE_TOP11 = "top11"
    MODE_FULL = "full"
    MODE_DEFAULT = MODE_TOP2
    MODES = [
        MODE_TOP,
        MODE_TOP2,
        MODE_TOP3,
        MODE_TOP4,
        MODE_TOP5,
        MODE_TOP6,
        MODE_TOP7,
        MODE_TOP8,
        MODE_TOP9,
        MODE_TOP10,
        MODE_TOP11,
        MODE_FULL,
    ]

    @property
    def named_modules(self) -> List[Tuple[str, nn.Module]]:
        if self._mode == ModelSummaryExtended.MODE_FULL:
            mods = self._model.named_modules()
            mods = list(mods)[1:]  # do not include root module (LightningModule)
        elif self._mode == ModelSummaryExtended.MODE_TOP:
            # the children are the top-level modules
            mods = self._model.named_children()
        elif self._mode[:3] == "top":
            depth = int(self._mode[3:])
            mods_full = self._model.named_modules()
            mods_full = list(mods_full)[1:]  # do not include root module (LightningModule)
            mods = [mod for mod in mods_full if mod[0].count(".") < depth]
        else:
            mods = []
        return list(mods)
