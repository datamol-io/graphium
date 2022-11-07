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
        print(f'args before prepare input {args}')
        args = self._prepare_input(args)
        args = args[0]
        poptorch_model = self.poptorch_models[stage]
        self.lightning_module._running_torchscript = True

        for key_to_drop in ['_batch_idx', 'mol_ids', 'smiles']:
            args.pop(key_to_drop)

        print(f'args before poptorch model {args}')
        out = poptorch_model(**args)
        print(f'output {out}')

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

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        outputs = {"loss/train": outputs["loss"].mean()}
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def training_step(self, features, labels) -> Dict[str, Any]:
        logger.warning('running training_step')
        features, labels = self.squeeze_input_dims(features, labels)
        dict_input = {'features': features, 'labels': labels}
        concatenated_metrics_logs = super().training_step(dict_input, to_cpu=False)

        loss = concatenated_metrics_logs.pop("loss")
        loss = self.poptorch.identity_loss(loss, reduction="mean")
        return loss  # Limitation that only the loss can be returned

    def validation_step(self, features, labels) -> Dict[str, Any]:
        logger.warning('running validation_step')
        features, labels = self.squeeze_input_dims(features, labels)
        dict_input = {'features': features, 'labels': labels}
        step_dict = super().validation_step(dict_input, to_cpu=False)

        # The output dict must be converted to a tuple
        step_dict = self._clean_output_batch(step_dict)
        return step_dict

    def test_step(self, **inputs) -> Dict[str, Any]:
        # Build a dictionary from the tuples
        dict_input = inputs
        step_dict = super().test_step(dict_input, to_cpu=False)

        # The output dict must be converted to a tuple
        step_dict = self._clean_output_batch(step_dict)
        return step_dict

    def predict_step(self, **inputs) -> Dict[str, Any]:
        # Build a dictionary from the tuples
        dict_input = inputs
        step_dict = super().predict_step(dict_input, to_cpu=False)

        # The output dict must be converted to a tuple
        step_dict = self._clean_output_batch(step_dict)
        return step_dict

    def training_epoch_end(self, outputs: Dict[str, Any]):
        # Limited. Since only the loss can be returned in `training_step`
        return

    def validation_epoch_end(self, outputs: Dict[str, Any]):
        # Retrieve the dict structure of the output batch from the tuple
        retrieved_outputs = self._retrieve_output_batch(outputs)
        return super().validation_epoch_end(retrieved_outputs)

    def test_epoch_end(self, outputs: Dict[str, Any]):
        # Retrieve the dict structure of the output batch from the tuple
        retrieved_outputs = self._retrieve_output_batch(outputs)
        return super().test_epoch_end(retrieved_outputs)

    def predict_epoch_end(self, outputs: Dict[str, Any]):
        # Retrieve the dict structure of the output batch from the tuple
        retrieved_outputs = self._retrieve_output_batch(outputs)
        return super().test_epoch_end(retrieved_outputs)

    def _retrieve_output_batch(self, outputs):
        """
        A limitation of the IPU is that only Tuples can be returned from
        `validation_step` and `test_step`.

        Here, we rebuild a dictionary from the tuples so that the code
        remains compatible with the original `[STAGE]_epoch_end` methods.

        The keys to rebuild the dict must be contained in `self._output_step_structure`
        as a `Dict[Dict[str]]`, with `self._output_step_structure["_others"]` representing the keys of
        the non-nested dictionaries.

        """

        new_outputs = []
        for batch in outputs:
            new_outputs.append({})
            # Get the keys of the nested dictionaries
            for ii, struct in enumerate(self._output_step_structure):
                new_outputs[-1][struct[0]] = {}
                for jj, key in enumerate(struct[1:]):
                    new_outputs[-1][struct[0]][key] = batch[ii][jj]

            # Pop the non-nested dictionaries, and re-add them
            others = new_outputs[-1].pop("_others", {})
            new_outputs[-1].update(others)
        return new_outputs

    def _clean_output_batch(self, out_batch):
        """
        The output batch cannot contain `Dict[Tensor]` or `Dict[Dict[Tensor]]`,
        only nested tuples. And they must all be the same depth.

        This function converts every `Dict[Tensor]` or `Dict[Dict[Tensor]]` to
        `Tuple[Tuple[Tensor]]` and stores the keys `self._output_step_structure`.

        In the case of non-nested dict, they will be stored under the key `_others`
        to ensure that everything has the same nesting depth.
        """

        # Transform Dict[Tensor] into Dict[Dict[Tensor]] by grouping them
        cleaned_batch, others = {}, {}
        for key, val in out_batch.items():
            if isinstance(val, dict):
                cleaned_batch[key] = val
            elif isinstance(val, Tensor):
                others[key] = val
        if len(others) > 0:
            cleaned_batch["_others"] = others

        # Save the structure of the dict somewhere
        output_step_structure = []
        for key, val in cleaned_batch.items():
            output_step_structure.append([key])
            for sub_key, sub_val in val.items():
                output_step_structure[-1].append(sub_key)
        self._output_step_structure = output_step_structure

        # Convert Dict[Dict[Tensor]] into Tuple[Tuple[Tensor]]
        new_dict = {}
        for key, val in cleaned_batch.items():
            new_dict[key] = tuple(val.values())
        cleaned_batch = tuple(new_dict.values())

        return cleaned_batch

    def squeeze_input_dims(self, features, labels):

        for key, tensor in features:
            if isinstance(tensor, torch.Tensor):
                features[key] = features[key].squeeze(0)

        for key in labels:
            labels[key] = labels[key].squeeze()

        return features, labels
