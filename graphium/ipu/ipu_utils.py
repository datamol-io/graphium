import os
import tempfile
from datetime import datetime
from copy import deepcopy
from types import ModuleType
from typing import Optional, Tuple, List
import torch


def import_poptorch(raise_error=True) -> Optional[ModuleType]:
    """
    Import poptorch and returns it.
    It is wrapped in a function to avoid breaking the code
    for non-IPU devices which did not install poptorch.

    Parameters:
        raise_error: Whether to raise an error if poptorch is unavailable.
            If `False`, return `None`

    Returns:
        The poptorch module

    """
    try:
        import poptorch

        return poptorch
    except ImportError as e:
        if raise_error:
            raise e
        return


def is_running_on_ipu() -> bool:
    """
    Returns whether the current module is running on ipu.
    Needs to be used in the `forward` or `backward` pass.
    """
    poptorch = import_poptorch(raise_error=False)
    on_ipu = (poptorch is not None) and (poptorch.isRunningOnIpu())
    return on_ipu


def load_ipu_options(
    ipu_opts: List[str],
    seed: Optional[int] = None,
    model_name: Optional[str] = None,
    gradient_accumulation: Optional[int] = None,
    precision: Optional[int] = None,
    ipu_inference_opts: Optional[List[str]] = None,
) -> Tuple["poptorch.Options", "poptorch.Options"]:
    """
    Load the IPU options from the config file.

    Parameters:
        ipu_cfg: The list  configurations for the IPU, written as a list of strings to make use of `poptorch.Options.loadFromFile`

            write a temporary config gile, and read it. See `Options.loadFromFile`
            #? see the tutorial for IPU options here
            # https://github.com/graphcore/tutorials/tree/sdk-release-2.6/tutorials/pytorch/efficient_data_loading
            #? see the full documentation for ipu options here
            # https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html?highlight=options#poptorch.Options

            ***minibatch size***: The number of samples processed by one simple fwd/bwd pass.
            = # of samples in a minibatch

            ***device iterations***: A device iteration corresponds to one iteration of the training loop executed on the IPU, starting with data-loading and ending with a weight update.
            In this simple case, when we set n deviceIterations, the host will prepare n mini-batches in an infeed queue so the IPU can perform efficiently n iterations.
            = # of minibatches to be processed at a time
            = # of training / backward pass in this call

            ***gradient accumulation factor***: After each backward pass the gradients are accumulated together for K mini-batches. set K in the argument
            = # of minibatches to accumulate gradients from

            ***replication factor***: Replication describes the process of running multiple instances of the same model simultaneously on different IPUs to achieve data parallelism.
            If the model requires N IPUs and the replication factor is M, N x M IPUs will be necessary.
            = # of times the model is copied to speed up computation, each replica of the model is sent a different subset of the dataset

            ***global batch size***: In a single device iteration, many mini-batches may be processed and the resulting gradients accumulated.
            We call this total number of samples processed for one optimiser step the global batch size.
            = total number of samples processed for *one optimiser step*
            = (minibatch size x Gradient accumulation factor) x Number of replicas

        seed: random seed for the IPU
        model_name: Name of the model, to be used for ipu profiling
        ipu_inference_opts: optional IPU configuration overrides for inference.
            If this is provided, options in this file override those in `ipu_file` for inference.

    Returns:

        training_opts: IPU options for the training set.

        inference_opts: IPU options for inference.
            It differs from the `training_opts` by enforcing `gradientAccumulation` to 1

    """

    poptorch = import_poptorch()
    ipu_options = poptorch.Options()
    ipu_opts_file = ipu_options_list_to_file(ipu_opts)
    ipu_options.loadFromFile(ipu_opts_file.name)
    ipu_opts_file.close()

    ipu_options.outputMode(poptorch.OutputMode.All)
    if seed is not None:
        ipu_options.randomSeed(seed)
    if model_name is not None:
        ipu_options.modelName(f"{model_name}_train")
    if gradient_accumulation is not None:
        current = ipu_options.Training.gradient_accumulation
        assert (current == 1) or (
            current == gradient_accumulation
        ), f"Received inconsistent gradient accumulation `{current}` and `{gradient_accumulation}"
        ipu_options.Training.gradientAccumulation(gradient_accumulation)

    if precision == "16-true":
        # IPUOptions.loadFromFile currently doesn't support setting half partials, doing it here
        ipu_options.Precision.setPartialsType(torch.half)
    training_opts = ipu_options

    # Change the inference options to remove gradient accumulation
    inference_opts = deepcopy(ipu_options)
    inference_opts.Training.gradientAccumulation(1)
    if ipu_inference_opts is not None:
        ipu_inference_opts_file = ipu_options_list_to_file(ipu_inference_opts)
        inference_opts.loadFromFile(ipu_inference_opts_file.name)
        ipu_inference_opts_file.close()

    return training_opts, inference_opts


def ipu_options_list_to_file(ipu_opts: Optional[List[str]]) -> tempfile._TemporaryFileWrapper:
    """
    Create a temporary file from a list of ipu configs, such that it can be read by `poptorch.Options.loadFromFile`

    Parameters:
        ipu_opts: The list  configurations for the IPU, written as a list of strings to make use of `poptorch.Options.loadFromFile`
    Returns:
        tmp_file: The temporary file of ipu configs
    """
    if ipu_opts is None:
        return

    tmp_file = tempfile.NamedTemporaryFile("w", delete=True)
    for s in ipu_opts:
        tmp_file.write(s + "\n")
    tmp_file.flush()
    return tmp_file
