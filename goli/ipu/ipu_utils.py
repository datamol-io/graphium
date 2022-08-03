from ast import Str
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


def load_ipu_options(ipu_file: Str) -> "poptorch.Options":
    """
    Load the IPU options from the config file.

    Parameters:
        config: The configuration as a Dictonary. Must contain the IPU options
            under `config["constant"]["accelerator"]["options"]`
    """

    poptorch = import_poptorch()

    #done : load the options from a config file. See `Options.loadFromFile`
    #? see the tutorial for IPU options here
    # https://github.com/graphcore/tutorials/tree/sdk-release-2.6/tutorials/pytorch/efficient_data_loading
    #? see the full documentation for ipu options here
    # https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html?highlight=options#poptorch.Options
    '''
    options explanations

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
    '''
    # ipu_options = poptorch.Options()
    # ipu_options.Training.gradientAccumulation(1)  #this is the gradient accumulation argument
    # ipu_options.deviceIterations(1) # I think this is similar to gradient accumulation??
    # ipu_options.replicationFactor(1)  # use 1 IPU for now in testing

    ipu_options = poptorch.Options()
    ipu_options.loadFromFile(ipu_file)
   
    # ipu_options.Jit.traceModel(False) # Use the experimental compiler
    # ipu_options._jit._values["trace_model"] = False

    return ipu_options
