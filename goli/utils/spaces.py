from copy import deepcopy
from ray import tune

from goli.nn.base_layers import FCLayer

from goli.nn.dgl_layers import (
    BaseDGLLayer,
    GATLayer,
    GCNLayer,
    GINLayer,
    GatedGCNLayer,
    PNAConvolutionalLayer,
    PNAMessagePassingLayer,
)

from goli.nn.residual_connections import (
    ResidualConnectionBase,
    ResidualConnectionConcat,
    ResidualConnectionDenseNet,
    ResidualConnectionNone,
    ResidualConnectionSimple,
    ResidualConnectionWeighted,
)


FC_LAYERS_DICT = {
    "fc": FCLayer,
}

DGL_LAYERS_DICT = {
    "gcn": GCNLayer,
    "gin": GINLayer,
    "gat": GATLayer,
    "gated-gcn": GatedGCNLayer,
    "pna-conv": PNAConvolutionalLayer,
    "pna-msgpass": PNAMessagePassingLayer,
}

LAYERS_DICT = deepcopy(DGL_LAYERS_DICT)
LAYERS_DICT.update(deepcopy(FC_LAYERS_DICT))


RESIDUALS_DICT = {
    "none": ResidualConnectionNone,
    "simple": ResidualConnectionSimple,
    "weighted": ResidualConnectionWeighted,
    "concat": ResidualConnectionConcat,
    "densenet": ResidualConnectionDenseNet,
}


SEARCH_SPACE_DGL_BASE = {
    "hidden_dim": tune.qloguniform(lower=64, upper=1024, q=1, base=2),
    "num_layers": tune.randint(3, 8),
    "dropout": tune.uniform(0, 0.4),
    "batch_norm": tune.choice([True, False]),
    "residual_type": tune.choice(RESIDUALS_DICT.keys()),
    "residual_skip_steps": tune.randint(1, 2),
    "layer_type": tune.choice(DGL_LAYERS_DICT.keys()),
    "pooling": tune.choice(["sum", "mean", "max", "s2s", ["sum", "max"]]),
    "virtual_node": tune.choice([None, "mean", "sum"]),
}


SEARCH_SPACE_DGL_EDGE_BASE = deepcopy(SEARCH_SPACE_DGL_BASE)
SEARCH_SPACE_DGL_EDGE_BASE.update({"tune.choice": tune.choice([0, 12])})


SCALERS = [["identity"], ["identity", "amplification"], ["identity", "amplification", "attenuation"]]

AGGREGATORS = [
    "mean",
    "max",
    ["mean", "max"],
    ["sum", "max"],
    ["mean", "sum", "max"],
    ["mean", "min", "max"],
    ["mean", "min", "max", "std"],
]


SEARCH_SPACE_GRAPH_PNA_CONV = {
    "scalers": tune.choice(SCALERS),
    "aggregators": tune.choice(AGGREGATORS),
}

SEARCH_SPACE_GRAPH_PNA_MSGPASS = deepcopy(SEARCH_SPACE_GRAPH_PNA_CONV)
SEARCH_SPACE_GRAPH_PNA_MSGPASS.update(
    {
        "pretrans_layers": tune.choice([1, 2]),
        "posttrans_layers": tune.choice([1, 2]),
    }
)


SPACE_MAP = {
    "gcn": {},
    "gin": {},
    "gat": {},
    "gated-gcn": {},
    "pna-conv": SEARCH_SPACE_GRAPH_PNA_CONV,
    "pna-msgpass": SEARCH_SPACE_GRAPH_PNA_MSGPASS,
}
