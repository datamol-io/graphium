from typing import Dict, Any, Optional, Callable, Union, Type, Tuple

from torch_geometric.data import Batch
from torch import Tensor
from inspect import _ParameterKind
from pytorch_lightning.plugins import IPUPlugin
from pytorch_lightning.trainer.states import RunningStage

from goli.trainer.predictor import PredictorModule
from goli.ipu.ipu_utils import import_poptorch


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


class IPUPluginGoli(IPUPlugin):
    """
    `IPUPluginGoli` modifies the `IPUPlugin` for compatibility with the Goli and Pytorch-Lightning training pipeline.
    Modifies the `.self` method to make it compatible with the rest of Goli.
    """

    def _step(self, stage: RunningStage, *args: Any, **kwargs: Any):
        """
        The `Goli` Dataloader generates Dictionaries as inputs for the `Predictor`.
        But the IPU does not support the following:
          - Dictionaries of inputs. Must pass Tuples. NamedTuples won't work, they will be converted to Tuples.
          - Only Tensors can be passed to the model. Anything else (strings, numpy array, etc) will cause compilation to fail.

        This function overloads the regular `_step` function to do the following:
          1. Create `Tuple[Tensor]` from the inputs of type `Dict[Tensor]` or `Tensor` or `pyg.Data.Batch` and pass them to `super()._step`
          2. Save the keys of the `Dict` structure as attributes of the model: `self.model.module._keys_tensor_dict` and `self.model.module._keys_tensor`
          3. Convert every non-Tensor to Tuples and pass to the module directly.
          4. Run the `super()._step` function, then remove every parameter that was added to the model

        The model must take care of re-building the Dict[Tensor] and `Batch` from the tuples. See class `PredictorModuleIPU`.

        """

        # Arguments for the loop
        new_args, all_keys = [], []
        keys_tensor_dict, keys_batch, keys_tensor, keys_others = {}, {}, {}, {}

        # Loop every argument. If a dict or pyg graph is found, split into tensors
        for data_key, data_val in args[0].items():
            if isinstance(data_val, (Dict, Batch)):
                for sub_key, sub_val in data_val.items():
                    if isinstance(sub_val, Tensor):
                        new_args.append(sub_val)
                        all_keys.append(sub_key)

                        # Append the keys for the tensors
                        if isinstance(data_val, Dict):
                            if data_key not in keys_tensor_dict:
                                keys_tensor_dict[data_key] = {}
                            keys_tensor_dict[data_key][sub_key] = len(all_keys) - 1

                        # Append the keys for the pyg Batch
                        elif isinstance(data_val, Batch):
                            if data_key not in keys_batch:
                                keys_batch[data_key] = {}
                            keys_batch[data_key][sub_key] = len(all_keys) - 1
                    else:
                        if data_key not in keys_others:
                            keys_others[data_key] = {}
                        keys_others[data_key][sub_key] = sub_val

            elif isinstance(data_val, Tensor):
                new_args.append(data_val)
                all_keys.append(data_key)
                keys_tensor[data_key] = len(all_keys) - 1
            else:
                keys_others[data_key] = data_val

        # Tell the module what are the labels associated to the labels and batches
        self.model.module._keys_tensor_dict = keys_tensor_dict
        self.model.module._keys_tensor = keys_tensor
        self.model.module._keys_batch = keys_batch
        self.model.module._keys_others = keys_others

        # Walk-around to set the variable names for the poptorch model
        self.poptorch_models[stage]._args_parser._varnames = all_keys
        self.poptorch_models[stage]._args_parser._var_kinds = [_ParameterKind.VAR_POSITIONAL] * len(all_keys)

        # Run the step using only tuple of tensors
        out = super()._step(stage, *new_args, **kwargs)

        # Remove the keys from the module after the step is executed
        self.model.module._keys_tensor_dict = None
        self.model.module._keys_tensor = None
        self.model.module._keys_batch = None
        self.model.module._keys_others = None

        return out


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
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        preds = remove_pad_loss(preds, targets)

        return PredictorModule.compute_loss(preds, targets, weights, loss_fun, target_nan_mask)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self._concatenated_metrics_logs["loss"] = outputs
        outputs = self._concatenated_metrics_logs
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def training_step(self, *inputs) -> Dict[str, Any]:
        # Build a dictionary from the tuples
        dict_input = self._build_dict_input(*inputs)
        concatenated_metrics_logs = super().training_step(dict_input, to_cpu=False)
        loss = concatenated_metrics_logs.pop("loss")
        self._concatenated_metrics_logs = concatenated_metrics_logs
        loss = self.poptorch.identity_loss(loss, reduction="mean")
        return loss  # Limitation that only the loss can be returned

    def validation_step(self, *inputs) -> Dict[str, Any]:
        # Build a dictionary from the tuples
        dict_input = self._build_dict_input(*inputs)
        step_dict = super().validation_step(dict_input, to_cpu=False)

        # The output dict must be converted to a tuple
        step_dict = self._clean_output_batch(step_dict)
        return step_dict

    def test_step(self, *inputs) -> Dict[str, Any]:
        # Build a dictionary from the tuples
        dict_input = self._build_dict_input(*inputs)
        step_dict = super().test_step(dict_input, to_cpu=False)

        # The output dict must be converted to a tuple
        step_dict = self._clean_output_batch(step_dict)
        return step_dict

    def predict_step(self, *inputs) -> Dict[str, Any]:
        # Build a dictionary from the tuples
        dict_input = self._build_dict_input(*inputs)
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

    def _build_dict_input(self, *inputs):
        """
        The method `IPUPluginGoli._step` converts the `Dict` structure into Tuples
        to allow the IPU tracer to compile correctly.

        This method rebuilds the `Dict` structure from the saved keys to allow
        to use the same code as the CPU or GPU.

        It processes the following attributes, which must be set by `IPUPluginGoli._step`:
          - _keys_batch: The keys for rebuilding the `pyg.Data.Batch`
          - _keys_tensor_dict: The keys for rebuilding the `Dict[Dict[Tensor]]`
          - _keys_tensor: The keys for rebuilding the `Dict[Tensor]`
          - _keys_others: The keys for non-Tensor inputs, which are passed directly to the model.
        """

        batch = {}

        # Unsqueeze the first dimension that is always 1 here, but is due to the global batch size
        # (device_iterations * replication_factor * gradient_accumulation)
        if all([this_input.shape[0] == 1 for this_input in inputs]):
            inputs = [this_input.squeeze(0) for this_input in inputs]

        # Initialize the batch of pyg objects
        for key, pyg_elems in self._keys_batch.items():
            pyg_batch = Batch()
            for pyg_key, idx in pyg_elems.items():
                pyg_batch[pyg_key] = inputs[idx]
            batch[key] = pyg_batch

        # Initialize the dictionaries of tensors, such as the multitask labels
        for key, this_dict in self._keys_tensor_dict.items():
            tensor_dict = {}
            for tensor_key, idx in this_dict.items():
                tensor_dict[tensor_key] = inputs[idx]
            batch[key] = tensor_dict

        # Initialize the tensors
        for key, idx in self._keys_tensor.items():
            batch[key] = inputs[idx]

        # Initialize the other elements
        for key, val in self._keys_others.items():
            if isinstance(val, dict):
                if key not in batch.keys():
                    batch[key] = {}
                # Update the dict or pyg-Batch
                for sub_key, sub_val in val.items():
                    batch[sub_key] = sub_val
            else:
                batch[key] = val

        # Get the current index for non-tensor elements
        batch_idx = batch.pop("_batch_idx")
        batch_idx = batch_idx.squeeze(-1).item()

        non_tensor_keys = set(self._keys_others.keys()) - (
            set(self._keys_batch.keys()) | set(self._keys_tensor.keys()) | set(self._keys_tensor_dict.keys())
        )
        for key in non_tensor_keys:
            batch[key] = batch[key][batch_idx]

        # Convert the tensors to their full dtype (instead of the reduced dtype used to increase data transfer speed)
        for key, new_dtype in self._keys_others["_types_conversion"][batch_idx].items():
            batch["features"][key] = batch["features"][key].to(new_dtype)

        return batch
