import pytest
from graphium.config._loader import _get_ipu_opts, load_ipu_options
from graphium.ipu.ipu_utils import ipu_options_list_to_file

import tempfile
from typing import Optional, List
import os

CONFIG_EXTRACT = {
    "trainer": {"trainer": {"accumulate_grad_batches": 10}},
    "accelerator": {
        "type": "ipu",
        "config_override": {
            "datamodule": {
                "args": {
                    "ipu_dataloader_training_opts": {
                        "mode": "async",
                        "max_num_nodes_per_graph": 44,
                        "max_num_edges_per_graph": 80,
                    },
                    "ipu_dataloader_inference_opts": {
                        "mode": "async",
                        "max_num_nodes_per_graph": 44,
                        "max_num_edges_per_graph": 80,
                    },
                    "batch_size_training": 50,
                    "batch_size_inference": 50,
                }
            },
            "predictor": {"optim_kwargs": {"loss_scaling": 1024}},
            "trainer": {"trainer": {"precision": 16, "accumulate_grad_batches": 4}},
        },
        "ipu_config": [
            "deviceIterations(5)",
            "replicationFactor(16)",
            "TensorLocations.numIOTiles(128)",
            '_Popart.set("defaultBufferingDepth", 128)',
            "Precision.enableStochasticRounding(True)",
        ],
        "ipu_inference_config": [
            "deviceIterations(1)",
            "replicationFactor(4)",
            "TensorLocations.numIOTiles(32)",
            '_Popart.set("defaultBufferingDepth", 16)',
            "Precision.enableStochasticRounding(True)",
        ],
    },
}


@pytest.mark.ipu
def test_ipu_options():
    try:
        import poptorch

        ipu_opts, ipu_inference_opts = _get_ipu_opts(CONFIG_EXTRACT)

        # Define the expected IPU options for comparison
        expected_ipu_opts = [
            "deviceIterations(5)",
            "replicationFactor(16)",
            "TensorLocations.numIOTiles(128)",
            '_Popart.set("defaultBufferingDepth", 128)',
            "Precision.enableStochasticRounding(True)",
        ]
        expected_ipu_inference_opts = [
            "deviceIterations(1)",
            "replicationFactor(4)",
            "TensorLocations.numIOTiles(32)",
            '_Popart.set("defaultBufferingDepth", 16)',
            "Precision.enableStochasticRounding(True)",
        ]

        # Test the _get_ipu_opts method
        ipu_opts, ipu_inference_opts = _get_ipu_opts(CONFIG_EXTRACT)
        assert ipu_opts == expected_ipu_opts, f"Expected {expected_ipu_opts}, but got {ipu_opts}"
        assert (
            ipu_inference_opts == expected_ipu_inference_opts
        ), f"Expected {expected_ipu_inference_opts}, but got {ipu_inference_opts}"

        # Test the load_ipu_options method
        ipu_training_opts, ipu_inference_opts = load_ipu_options(
            ipu_opts=ipu_opts,
            seed=42,
            model_name="test_model",
            gradient_accumulation=CONFIG_EXTRACT["trainer"]["trainer"].get("accumulate_grad_batches", None),
            ipu_inference_opts=ipu_inference_opts,
        )

        # Ensure that the options objects are not None
        assert ipu_training_opts is not None, "Expected ipu_training_opts not to be None"
        assert ipu_inference_opts is not None, "Expected ipu_inference_opts not to be None"

        # Test the properties of the options objects
        assert (
            ipu_training_opts.replication_factor == 16
        ), "Expected replication_factor of ipu_training_opts to be 16"
        assert (
            ipu_inference_opts.replication_factor == 4
        ), "Expected replication_factor of ipu_inference_opts to be 4"
        assert ipu_training_opts._popart, "Expected _popart of ipu_training_opts to be True"
        assert ipu_inference_opts._popart, "Expected _popart of ipu_inference_opts to be True"

    except ImportError:
        pytest.skip("Skipping this test because poptorch is not available")


@pytest.mark.ipu
def test_ipu_options_list_to_file():
    # Define a list of IPU options
    ipu_options = [
        "deviceIterations(5)",
        "replicationFactor(16)",
        "TensorLocations.numIOTiles(128)",
        '_Popart.set("defaultBufferingDepth", 128)',
        "Precision.enableStochasticRounding(True)",
    ]

    # Call the function with the list of IPU options
    tmp_file = ipu_options_list_to_file(ipu_options)

    # Check that the function returns a temporary file object
    assert isinstance(tmp_file, tempfile._TemporaryFileWrapper)

    # Check that the temporary file exists
    assert os.path.exists(tmp_file.name)

    # Check the contents of the temporary file
    with open(tmp_file.name, "r") as f:
        contents = f.read().splitlines()
    assert contents == ipu_options

    # Check the behavior when the input is None
    tmp_file = ipu_options_list_to_file(None)
    assert tmp_file is None
