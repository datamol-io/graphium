import torch

from collections import defaultdict
from typing import Union, List

from graphium.nn.architectures.global_architectures import FullGraphMultiTaskNetwork


class Fingerprinter:
    """Class to extract fingerprints from a FullGraphMultiTaskNetwork"""

    def __init__(
        self,
        network: FullGraphMultiTaskNetwork,
        fingerprint_spec: Union[str, List[str]],
    ):
        """
        Args:
            network: The network to extract fingerprints from
            fingerprint_spec: The fingerprint specification. Of the format "module:layer". If specified as a list,
                the fingerprints from all the specified layers will be concatenated.
        """

        if not isinstance(network, FullGraphMultiTaskNetwork):
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports fingerprints for the FullGraphMultiTaskNetwork"
            )
        self.network = network

        if isinstance(fingerprint_spec, str):
            fingerprint_spec = [fingerprint_spec]

        self._spec = defaultdict(list)
        for spec_str in fingerprint_spec:
            module_name, layer = spec_str.split(":")
            self._spec[module_name].append(int(layer.strip()))

        self._module_map_backup = None

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
            self.network(batch)

        readout_list = []
        for module_name, layers in self._spec.items():
            readout_list.extend(
                [self.network._module_map[module_name]._readout_cache[layer] for layer in layers]
            )
        return torch.cat(readout_list, dim=-1)

    def __enter__(self):
        """Setup the network for fingerprinting"""
        if hasattr(self.network, "_module_map"):
            self._module_map_backup = self.network._module_map
        self.network._enable_readout_cache(list(self._spec.keys()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Revert the network to its original state"""
        self.network._disable_readout_cache()
        if self._module_map_backup is not None:
            self.network._module_map = self._module_map_backup
        else:
            del self.network._module_map
