
def get_poptorch():
    try:
        import poptorch
        return poptorch
    except ImportError:
        raise ImportError("You must install poptorch and have IPU hardware. Check the GraphCore support https://www.graphcore.ai/support")

def load_ipu_options(config):
    poptorch = get_poptorch()

    #! # TODO: Actually load the options from a config file. See `Options.loadFromFile`
    ipu_options = poptorch.Options()
    ipu_options.deviceIterations(1) # I think this is similar to gradient accumulation??
    ipu_options.replicationFactor(1)  # use 1 IPU for now in testing
    ipu_options.replicationFactor(1)  # use 1 IPU for now in testing
    # ipu_options.Jit.traceModel(False) # Use the experimental compiler
    # ipu_options._jit._values["trace_model"] = False

    return ipu_options
