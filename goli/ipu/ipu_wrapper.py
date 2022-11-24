from typing import Dict, Any, Optional, Callable, Union, Type, Tuple

from torch_geometric.data import Batch
from torch import Tensor
import torch
from inspect import _ParameterKind
from pytorch_lightning.strategies import IPUStrategy
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.plugins import IPUPlugin
from pytorch_lightning.trainer.states import RunningStage

from goli.trainer.predictor import PredictorModule
from goli.ipu.ipu_utils import import_poptorch

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
from loguru import logger

poptorch = import_poptorch()


def remove_pad_loss(preds: Dict[str, Tensor], targets: Dict[str, Tensor]):
    """
    helper function to remove the fake graph loss
    always reduce the last loss since it is the fake graph
    """
    for task in targets.keys():
        if targets[task].shape == preds[task].shape:
            continue
        else:
            preds[task] = preds[task][:-1]
    return preds

class DictIPUStrategy(IPUStrategy):

    def _step(self, stage: RunningStage, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        args = self._prepare_input(args)
        args = args[0]
        poptorch_model = self.poptorch_models[stage]
        self.lightning_module._running_torchscript = True
        for key_to_drop in ['_batch_idx', 'mol_ids', 'smiles']:
            args.pop(key_to_drop)
        out = poptorch_model(**args)
        self.lightning_module._running_torchscript = False
        return out


class PyGArgsParser(poptorch.ICustomArgParser):

    @staticmethod
    def sortedTensorKeys(struct):
        """
        Find all the keys that map to a tensor value in struct. The keys
        are returned in sorted order.
        """
        all_keys = sorted(struct.keys)

        def isTensor(k):
            return isinstance(struct[k], torch.Tensor)

        return filter(isTensor, all_keys)

    def yieldTensors(self, struct):
        """
        yield every torch.Tensor in struct in sorted order
        """
        for k in self.sortedTensorKeys(struct):
            yield struct[k]

    def reconstruct(self, original_structure, tensor_iterator):
        """
        Create a new instance with the same class type as the
        original_structure. This new instance will be initialized with tensors
        from the provided iterator and uses the same sorted keys from the
        yieldTensors() implementation.
        """
        tensor_keys = self.sortedTensorKeys(original_structure)
        kwargs = {k: next(tensor_iterator) for k in tensor_keys}

        for k in original_structure.keys:
            if k not in kwargs:
                # copy non-tensor properties to the new instance
                kwargs[k] = original_structure[k]

        cls = original_structure.__class__

        if issubclass(cls, Batch):
            kwargs['_base_cls'] = Data
            return Batch(**kwargs)

        return cls(**kwargs)


# PyG uses the BaseData object as the root for data and batch objects
poptorch.registerCustomArgParser(BaseData, PyGArgsParser())


class PredictorModuleIPU(PredictorModule):
    """
    This class wraps around the `PredictorModule` to make it work with IPU and the `IPUPluginGoli`.
    """

    def __init__(self, *args, **kwargs):
        # Import poptorch in a safe way that will work when working with cpu/gpu
        self.poptorch = import_poptorch()
        super().__init__(*args, **kwargs)

    @staticmethod
    def compute_loss(
        preds: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        weights: Optional[Tensor],
        loss_fun: Dict[str, Callable],
        target_nan_mask: Union[Type, str] = "ignore-flatten",
        multitask_handling: Optional[str] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        preds = remove_pad_loss(preds, targets)

        return PredictorModule.compute_loss(
            preds, targets, weights, loss_fun, target_nan_mask, multitask_handling
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        outputs["loss/train"] = outputs["loss"].mean()
        super().on_train_batch_end(outputs, batch, batch_idx)

    def training_step(self, features, labels) -> Dict[str, Any]:
        logger.warning('running training_step')
        features, labels = self.squeeze_input_dims(features, labels)
        dict_input = {'features': features, 'labels': labels}
        step_dict = super().training_step(dict_input, to_cpu=False)

        loss = step_dict.pop("loss")
        step_dict["loss"] = self.poptorch.identity_loss(loss, reduction="mean")
        return step_dict

    def validation_step(self, features, labels) -> Dict[str, Any]:
        logger.warning('running validation_step')
        features, labels = self.squeeze_input_dims(features, labels)
        dict_input = {'features': features, 'labels': labels}
        step_dict = super().validation_step(dict_input, to_cpu=False)

        return step_dict

    def test_step(self, **inputs) -> Dict[str, Any]:
        # Build a dictionary from the tuples
        dict_input = inputs
        step_dict = super().test_step(dict_input, to_cpu=False)

        return step_dict

    def predict_step(self, **inputs) -> Dict[str, Any]:
        # Build a dictionary from the tuples
        dict_input = inputs
        step_dict = super().predict_step(dict_input, to_cpu=False)

        return step_dict

    def squeeze_input_dims(self, features, labels):

        for key, tensor in features:
            if isinstance(tensor, torch.Tensor):
                features[key] = features[key].squeeze(0)

        for key in labels:
            labels[key] = labels[key].squeeze(0)

        return features, labels
