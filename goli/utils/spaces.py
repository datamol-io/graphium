from copy import deepcopy

from goli.nn.base_layers import FCLayer

from goli.nn.dgl_layers import (
    GATLayer,
    GCNLayer,
    GINLayer,
    GatedGCNLayer,
    PNAConvolutionalLayer,
    PNAMessagePassingLayer,
    DGNConvolutionalLayer,
    DGNMessagePassingLayer,
)

from goli.nn.residual_connections import (
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
    "dgn-conv": DGNConvolutionalLayer,
    "dgn-msgpass": DGNMessagePassingLayer,
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

