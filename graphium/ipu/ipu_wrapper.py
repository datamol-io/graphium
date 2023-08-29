from typing import Dict, Any, Optional, Callable, Union, Type, Tuple, Iterable

from torch_geometric.data import Batch
from torch import Tensor
from lightning_graphcore import IPUStrategy
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.trainer.states import RunningStage

from graphium.trainer.predictor import PredictorModule
from graphium.ipu.ipu_utils import import_poptorch

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
from loguru import logger
import functools
import collections
from graphium.data.utils import get_keys

poptorch = import_poptorch()


class PyGArgsParser(poptorch.ICustomArgParser):
    """
    This class is responsible for converting a PyG Batch from and to
    a tensor of tuples. This allows PyG Batch to be used as inputs to
    IPU programs. Copied from poppyg repo, in the future import from
    the repo directly.
    """

    @staticmethod
    def sortedTensorKeys(struct: BaseData) -> Iterable[str]:
        """
        Find all the keys that map to a tensor value in struct. The keys
        are returned in sorted order.
        """
        all_keys = sorted(get_keys(struct))

        def isTensor(k: str) -> bool:
            return isinstance(struct[k], torch.Tensor)

        return filter(isTensor, all_keys)

    def yieldTensors(self, struct: BaseData):
        """
        yield every torch.Tensor in struct in sorted order
        """
        for k in self.sortedTensorKeys(struct):
            yield struct[k]

    def reconstruct(self, original_structure: BaseData, tensor_iterator: Iterable[Tensor]):
        """
        Create a new instance with the same class type as the
        original_structure. This new instance will be initialized with tensors
        from the provided iterator and uses the same sorted keys from the
        yieldTensors() implementation.
        """
        tensor_keys = self.sortedTensorKeys(original_structure)
        kwargs = {k: next(tensor_iterator) for k in tensor_keys}

        for k in get_keys(original_structure):
            if k not in kwargs:
                # copy non-tensor properties to the new instance
                kwargs[k] = original_structure[k]

        cls = original_structure.__class__

        if issubclass(cls, Batch):
            kwargs["_base_cls"] = Data
            return Batch(**kwargs)

        return cls(**kwargs)


# PyG uses the BaseData object as the root for data and batch objects
poptorch.registerCustomArgParser(BaseData, PyGArgsParser())


class PredictorModuleIPU(PredictorModule):
    """
    This class wraps around the `PredictorModule` to make it work with IPU and the `IPUPluginGraphium`.
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
        target_nan_mask: Union[Type, str] = "ignore",
        multitask_handling: Optional[str] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        return PredictorModule.compute_loss(
            preds, targets, weights, loss_fun, target_nan_mask, multitask_handling
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        outputs = self.convert_from_fp16(outputs)
        outputs["loss"] = outputs["loss"][outputs["loss"] != 0].mean()
        super().on_train_batch_end(outputs, batch, batch_idx)

    def training_step(self, batch, batch_idx) -> Dict[str, Any]:
        features, labels = batch["features"], batch["labels"]
        features, labels = self.squeeze_input_dims(features, labels)
        dict_input = {"features": features, "labels": labels}
        step_dict = super().training_step(dict_input, to_cpu=False)

        loss = step_dict.pop("loss")
        step_dict["loss"] = self.poptorch.identity_loss(loss, reduction="mean")
        return step_dict

    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        features, labels = batch["features"], batch["labels"]
        features, labels = self.squeeze_input_dims(features, labels)
        dict_input = {"features": features, "labels": labels}
        step_dict = super().validation_step(dict_input, to_cpu=False)

        return step_dict

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        # Build a dictionary from the tuples
        features, labels = batch["features"], batch["labels"]
        features, labels = self.squeeze_input_dims(features, labels)
        dict_input = {"features": features, "labels": labels}
        step_dict = super().test_step(dict_input, to_cpu=False)

        return step_dict

    def predict_step(self, **inputs) -> Dict[str, Any]:
        # Build a dictionary from the tuples
        dict_input = inputs
        step_dict = super().predict_step(dict_input, to_cpu=False)

        return step_dict

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        # convert data that will be tracked
        outputs = self.convert_from_fp16(outputs)
        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def evaluation_epoch_end(self, outputs: Any):
        outputs = self.convert_from_fp16(outputs)
        super().evaluation_epoch_end(outputs)

    def on_test_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        outputs = self.convert_from_fp16(outputs)
        super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def configure_optimizers(self, impl=None):
        if impl is None:
            dtype = self.precision_to_dtype(self.trainer.precision)
            impl = functools.partial(
                self.poptorch.optim.Adam,
                accum_type=dtype,
                first_order_momentum_accum_type=dtype,
                second_order_momentum_accum_type=torch.float,
            )
        return super().configure_optimizers(impl=impl)

    def squeeze_input_dims(self, features, labels):
        for key, tensor in features:
            if isinstance(tensor, torch.Tensor):
                features[key] = features[key].squeeze(0)

        for key in labels:
            labels[key] = labels[key].squeeze(0)

        return features, labels

    def convert_from_fp16(self, data: Any) -> Any:
        """
        Converts tensors from FP16 to FP32. Useful to convert the IPU program output data
        """
        if isinstance(data, collections.Sequence):
            for idx in range(len(data)):
                data[idx] = self.convert_from_fp16(data[idx])
        elif isinstance(data, collections.Mapping):
            for key in data:
                data[key] = self.convert_from_fp16(data[key])
        elif isinstance(data, torch.Tensor) and data.dtype == torch.float16:
            data = data.float()
        return data

    def _convert_features_dtype(self, feats):
        """
        Converts features to trainer precision rather than model precision.
        Necessary to run IPU on FP16.
        """
        dtype = self.precision_to_dtype(self.trainer.precision)

        # Convert features to dtype
        if isinstance(feats, torch.Tensor):
            feats = feats.to(dtype)
        elif isinstance(feats, (Data, Batch, dict)):
            for key, val in feats.items():
                if isinstance(val, torch.Tensor) and (val.is_floating_point()):
                    feats[key] = val.to(dtype=dtype)
        else:
            raise ValueError(f"Unsupported feats type `{type(feats)}` : {feats}")
        return feats

    def precision_to_dtype(self, precision):
        return torch.half if precision == "16-true" else torch.float

    def get_num_graphs(self, data: Batch):
        """
        IPU specific method to compute the number of graphs in a Batch,
        that considers gradient accumulation, multiple IPUs and multiple
        device iterations. Essential to estimate throughput in graphs/s.
        """
        num_graphs = torch.max(data.batch, dim=-1).values
        num_graphs = torch.sum(num_graphs)

        return num_graphs
