from typing import Dict

def import_poptorch():
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


def load_ipu_options(config: Dict) -> "poptorch.Options":
    """
    Load the IPU options from the config file.

    Parameters:
        config: The configuration as a Dictonary. Must contain the IPU options
            under `config["constant"]["accelerator"]["options"]`
    """

    poptorch = import_poptorch()

    #! # TODO: Actually load the options from a config file. See `Options.loadFromFile`
    #? see the tutorial for IPU options here
    # https://github.com/graphcore/tutorials/tree/sdk-release-2.6/tutorials/pytorch/efficient_data_loading
    '''
    options explanations
    ***device iterations***: A device iteration corresponds to one iteration of the training loop executed on the IPU, starting with data-loading and ending with a weight update. 
    In this simple case, when we set n deviceIterations, the host will prepare n mini-batches in an infeed queue so the IPU can perform efficiently n iterations.

    ***replication factor***: Replication describes the process of running multiple instances of the same model simultaneously on different IPUs to achieve data parallelism.

    ***global batch size***: In a single device iteration, many mini-batches may be processed and the resulting gradients accumulated. 
    We call this total number of samples processed for one optimiser step the global batch size.

    '''
    ipu_options = poptorch.Options()
    ipu_options.deviceIterations(1) # I think this is similar to gradient accumulation??
    ipu_options.replicationFactor(1)  # use 1 IPU for now in testing
    # ipu_options.Jit.traceModel(False) # Use the experimental compiler
    # ipu_options._jit._values["trace_model"] = False

    return ipu_options
