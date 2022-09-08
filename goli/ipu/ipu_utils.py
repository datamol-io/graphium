from copy import deepcopy
from types import ModuleType
from typing import Optional, Tuple

def import_poptorch() -> ModuleType:
    """
    Import poptorch and returns it.
    It is wrapped in a function to avoid breaking the code
    for non-IPU devices which did not install poptorch.

    Returns:
        The poptorch module

    """
    try:
        import poptorch
        return poptorch
    except ImportError:
        raise ImportError("You must install poptorch and have IPU hardware. Check the GraphCore support https://www.graphcore.ai/support")


def load_ipu_options(ipu_file: str, seed: Optional[int]=None) -> Tuple["poptorch.Options", "poptorch.Options"]:
    """
    Load the IPU options from the config file.

    Parameters:
        ipu_file: file path containing the IPU configurations. Example of options are:

            load the options from a config file. See `Options.loadFromFile`
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

    Returns:

        training_opts: IPU options for the training set.

        inference_opts: IPU options for inference.
            It differs from the `training_opts` by enforcing `gradientAccumulation` to 1

    """

    poptorch = import_poptorch()

    ipu_options = poptorch.Options()
    ipu_options.loadFromFile(ipu_file)
    ipu_options.outputMode(poptorch.OutputMode.All)
    if seed is not None:
        ipu_options.randomSeed(seed)

    #ipu_options.anchorTensor("grad_input", "Gradient___input")
    ipu_options.anchorTensor("input", "input")

    training_opts = ipu_options

    # Change the inference options to remove gradient accumulation
    inference_opts = deepcopy(ipu_options)
    inference_opts.Training.gradientAccumulation(1)

    return training_opts, inference_opts
