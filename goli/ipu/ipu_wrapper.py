from typing import Dict, Any

from torch_geometric.data import Batch
from torch import Tensor
from inspect import _ParameterKind
from pytorch_lightning.plugins import IPUPlugin
from pytorch_lightning.trainer.states import RunningStage

from goli.trainer.predictor import PredictorModule
from goli.ipu.ipu_utils import get_poptorch


class IPUPluginGoli(IPUPlugin):

    def _step(self, stage: RunningStage, *args: Any, **kwargs: Any):

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

    def __init__(self, *args, **kwargs):
        self.poptorch = get_poptorch()
        super().__init__(*args, **kwargs)

    def training_step(self, *inputs) -> Dict[str, Any]:
        dict_input = self._build_dict_input(*inputs)
        step_dict = super().training_step(dict_input, to_cpu=False)
        loss = self.poptorch.identity_loss(step_dict["loss"], reduction="mean")
        return loss # Limitation that only the loss can be returned

    def validation_step(self, *inputs) -> Dict[str, Any]:
        dict_input = self._build_dict_input(*inputs)
        step_dict = super().validation_step(dict_input, to_cpu=False)
        step_dict = self._clean_output_batch(step_dict)
        return step_dict

    def test_step(self, *inputs) -> Dict[str, Any]:
        dict_input = self._build_dict_input(*inputs)
        step_dict = super().test_step(dict_input, to_cpu=False)
        step_dict = self._clean_output_batch(step_dict)
        return step_dict

    def predict_step(self, *inputs) -> Dict[str, Any]:
        dict_input = self._build_dict_input(*inputs)
        step_dict = super().test_step(dict_input, to_cpu=False)
        step_dict = self._clean_output_batch(step_dict)
        return step_dict

    def training_epoch_end(self, outputs: Dict[str, Any]):
        # Limited. Since only the loss can be returned in `training_step`
        return

    def validation_epoch_end(self, outputs: Dict[str, Any]):
        retrieved_outputs = self._retrieve_output_batch(outputs)
        return super().validation_epoch_end(retrieved_outputs)

    def test_epoch_end(self, outputs: Dict[str, Any]):
        retrieved_outputs = self._retrieve_output_batch(outputs)
        return super().test_epoch_end(retrieved_outputs)

    def predict_epoch_end(self, outputs: Dict[str, Any]):
        retrieved_outputs = self._retrieve_output_batch(outputs)
        return super().test_epoch_end(retrieved_outputs)

    def _retrieve_output_batch(self, outputs):
        new_outputs = []
        for batch in outputs:
            new_outputs.append({})
            for ii, struct in enumerate(self._output_step_structure):
                new_outputs[-1][struct[0]] = {}
                for jj, key in enumerate(struct[1:]):
                    new_outputs[-1][struct[0]][key] = batch[ii][jj]
            others = new_outputs[-1].pop("_others", {})
            new_outputs[-1].update(others)
        return new_outputs

    def _clean_output_batch(self, out_batch):

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

        batch = {}

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

        return batch
