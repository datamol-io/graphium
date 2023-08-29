import torch

from collections import defaultdict
from typing import Literal, Union, List

import tqdm

from graphium.nn.architectures.global_architectures import FullGraphMultiTaskNetwork
from graphium.trainer.predictor import PredictorModule


class Fingerprinter:
    """Extract fingerprints from a [`FullGraphMultiTaskNetwork`][graphium.nn.architectures.global_architectures.FullGraphMultiTaskNetwork]

    Fingerprints are hidden representations of a pre-trained network.
    They can be used as an additional representation for task-specific ML models
    in downstream tasks.

    Info: Connection to linear probing.
        This two-stage process is similar in concept to linear-probing,
        but pre-computing the fingerprints further decouples the two stages
        and allows for more flexibility.

    Note: CLI support
        You can extract fingerprints easily with the CLI. For more info, see
        ```sh
        graphium finetune fingerprint --help
        ```

    To return intermediate representations, the network will be altered
    to save the readouts of the specified layers. This class is designed to be used
    as a context manager that automatically reverts the network to its original state.

    Examples:

        Basic usage:
        ```python
        # Return layer 1 of the gcn submodule.
        # For a full list of options, see the `_module_map` attribute of the FullGraphMultiTaskNetwork.
        fp_spec = "gcn:1"

        # Create the object
        fingerprinter = Fingerprinter(predictor, "gcn:1")

        # Setup, run and teardown the fingerprinting process
        fingerprinter.setup()
        fp = fp.get_fingerprints_for_batch(batch)
        fingerprinter.teardown()
        ```

        As context manager:
        ```python
        with Fingerprinter(predictor, "gcn:1") as fingerprinter:
            fp = fp.get_fingerprints_for_batch(batch)
        ```

        Concatenating multiple hidden representations:
        ```python
        with Fingerprinter(predictor, ["gcn:0", "gcn:1"]) as fingerprinter:
            fp = fp.get_fingerprints_for_batch(batch)
        ```

        For an entire dataset (expects a PyG dataloader):
        ```python
        with Fingerprinter(predictor, ["gcn:0", "gcn:1"]) as fingerprinter:
            fps = fp.get_fingerprints_for_dataset(dataloader)
        ```

        Returning numpy arrays instead of torch tensors:
        ```python
        with Fingerprinter(predictor, ["gcn:0", "gcn:1"], out_type="numpy") as fingerprinter:
            fps = fp.get_fingerprints_for_dataset(dataloader)
        ```
    """

    def __init__(
        self,
        model: Union[FullGraphMultiTaskNetwork, PredictorModule],
        fingerprint_spec: Union[str, List[str]],
        out_type: Literal["torch", "numpy"] = "torch",
    ):
        """
        Args:
            model: Either a `FullGraphMultiTaskNetwork` or a `PredictorModule` that contains one. The benefit
                of using a `PredictorModule` is that it will automatically convert the features to the correct dtype.
            fingerprint_spec: The fingerprint specification. Of the format `module:layer`. If specified as a list,
                the fingerprints from all the specified layers will be concatenated.
            out_type: The output type of the fingerprints. Either `torch` or `numpy`.
        """

        network = model
        predictor = None

        if isinstance(model, PredictorModule):
            predictor = model
            network = model.model

        if not isinstance(network, FullGraphMultiTaskNetwork):
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports fingerprints for the FullGraphMultiTaskNetwork"
            )

        if out_type not in ["torch", "numpy"]:
            raise ValueError(f"Invalid output type {out_type}, pick from 'torch' or 'numpy'")

        self.network = network
        self.predictor = predictor
        self._out_type = out_type

        if isinstance(fingerprint_spec, str):
            fingerprint_spec = [fingerprint_spec]

        self._spec = defaultdict(list)
        for spec_str in fingerprint_spec:
            module_name, layer = spec_str.split(":")
            self._spec[module_name].append(int(layer.strip()))

        self._module_map_backup = None

    def setup(self):
        """Prepare the network for fingerprinting"""
        if hasattr(self.network, "_module_map"):
            self._module_map_backup = self.network._module_map
        self.network._enable_readout_cache(list(self._spec.keys()))
        return self

    def get_fingerprints_for_batch(self, batch):
        """Get the fingerprints for a single batch"""

        if not self.network._cache_readouts:
            raise RuntimeError(
                f"To use {self.__class__.__name__}, you must enable readout caching in the network. "
                f"Alternatively, you can use the {self.__class__.__name__} as a context manager "
                "to automatically setup the network for fingerprinting and revert the changes afterwards"
            )

        # Run the batch through the model.
        with torch.inference_mode():
            if self.predictor is not None:
                batch["features"] = self.predictor._convert_features_dtype(batch["features"])
            self.network(batch["features"])

        readout_list = []
        for module_name, layers in self._spec.items():
            readout_list.extend(
                [self.network._module_map[module_name]._readout_cache[layer].cpu() for layer in layers]
            )

        feats = torch.cat(readout_list, dim=-1)
        return self._convert_output_type(feats)

    def get_fingerprints_for_dataset(self, dataloader):
        """Return the fingerprints for an entire dataset"""

        original_out_type = self._out_type
        self._out_type = "torch"

        fps = []
        for batch in tqdm.tqdm(dataloader, desc="Fingerprinting batches"):
            feats = self.get_fingerprints_for_batch(batch)
            fps.append(feats)

        fps = torch.cat(fps, dim=0)

        self._out_type = original_out_type
        return self._convert_output_type(fps)

    def teardown(self):
        """Restore the network to its original state"""
        self.network._disable_readout_cache()
        if self._module_map_backup is not None:
            self.network._module_map = self._module_map_backup
        else:
            del self.network._module_map
        return self

    def __enter__(self):
        """Setup the network for fingerprinting"""
        return self.setup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Revert the network to its original state"""
        if exc_type is not None:
            raise exc_type(exc_val)
        return self.teardown()

    def _convert_output_type(self, feats: torch.Tensor):
        """Small utility function to convert output types"""
        if self._out_type == "numpy":
            feats = feats.numpy()
        return feats
