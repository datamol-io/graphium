from copy import deepcopy
import torch
import torch.optim.lr_scheduler as sc
import torchmetrics.functional as met

from goli.nn.base_layers import FCLayer
from goli.data.datamodule import (
    GraphOGBDataModule,
    MultitaskFromSmilesDataModule,
    MultitaskIPUFromSmilesDataModule,
    MultitaskIPUFromSmilesDataModule,
)
from goli.ipu.ipu_metrics import BCELossIPU, MSELossIPU, L1LossIPU

from goli.nn.dgl_layers import (
    GATDgl,
    GCNDgl,
    GINDgl,
    GatedGCNDgl,
    PNAConvolutionalDgl,
    PNAMessagePassingDgl,
    DGNConvolutionalDgl,
    DGNMessagePassingDgl,
)

from goli.nn.pyg_layers import PNAMessagePassingPyg, GINConvPyg, GINEConvPyg, GatedGCNPyg, GPSLayerPyg

from goli.nn.residual_connections import (
    ResidualConnectionConcat,
    ResidualConnectionDenseNet,
    ResidualConnectionNone,
    ResidualConnectionSimple,
    ResidualConnectionWeighted,
    ResidualConnectionRandom,
)

# * Andy adding positional encoder names here
from goli.nn.encoders import laplace_pos_encoder, mlp_encoder, signnet_pos_encoder

PE_ENCODERS_DICT = {
    "laplacian_pe": laplace_pos_encoder.LapPENodeEncoder,
    "mlp": mlp_encoder.MLPEncoder,
    "signnet": signnet_pos_encoder.SignNetNodeEncoder,
}


FC_LAYERS_DICT = {
    "fc": FCLayer,
}

DGL_LAYERS_DICT = {
    "dgl:gcn": GCNDgl,
    "dgl:gin": GINDgl,
    "dgl:gat": GATDgl,
    "dgl:gated-gcn": GatedGCNDgl,
    "dgl:pna-conv": PNAConvolutionalDgl,
    "dgl:pna-msgpass": PNAMessagePassingDgl,
    "dgl:dgn-conv": DGNConvolutionalDgl,
    "dgl:dgn-msgpass": DGNMessagePassingDgl,
}

PYG_LAYERS_DICT = {
    "pyg:gin": GINConvPyg,
    "pyg:gine": GINEConvPyg,
    "pyg:gated-gcn": GatedGCNPyg,
    "pyg:pna-msgpass": PNAMessagePassingPyg,
    "pyg:gps": GPSLayerPyg,
}

LAYERS_DICT = deepcopy(DGL_LAYERS_DICT)
LAYERS_DICT.update(deepcopy(FC_LAYERS_DICT))
LAYERS_DICT.update(deepcopy(PYG_LAYERS_DICT))


RESIDUALS_DICT = {
    "none": ResidualConnectionNone,
    "simple": ResidualConnectionSimple,
    "weighted": ResidualConnectionWeighted,
    "concat": ResidualConnectionConcat,
    "densenet": ResidualConnectionDenseNet,
    "random": ResidualConnectionRandom,
}

LOSS_DICT = {
    "mse": torch.nn.MSELoss(),
    "bce": torch.nn.BCELoss(),
    "l1": torch.nn.L1Loss(),
    "mae": torch.nn.L1Loss(),
    "bce_ipu": BCELossIPU(),
    "mse_ipu": MSELossIPU(),
    "mae_ipu": L1LossIPU(),
    "l1_ipu": L1LossIPU(),
}


SCHEDULER_DICT = {
    "CosineAnnealingLR": sc.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": sc.CosineAnnealingWarmRestarts,
    "CyclicLR": sc.CyclicLR,
    "ExponentialLR": sc.ExponentialLR,
    "LambdaLR": sc.LambdaLR,
    "MultiStepLR": sc.MultiStepLR,
    "ReduceLROnPlateau": sc.ReduceLROnPlateau,
    "StepLR": sc.StepLR,
}

METRICS_CLASSIFICATION = {
    "accuracy": met.accuracy,
    "averageprecision": met.average_precision,
    "auroc": met.auroc,
    "confusionmatrix": met.confusion_matrix,
    "f1": met.f1_score,
    "fbeta": met.fbeta_score,
    "precisionrecallcurve": met.precision_recall_curve,
    "precision": met.precision,
    "recall": met.recall,
    "mcc": met.matthews_corrcoef,
}

METRICS_REGRESSION = {
    "mae": met.mean_absolute_error,
    "mape": met.mean_absolute_percentage_error,
    "mse": met.mean_squared_error,
    "msle": met.mean_squared_log_error,
    "pearsonr": met.pearson_corrcoef,
    "spearmanr": met.spearman_corrcoef,
    "r2": met.r2_score,
    "cosine": met.cosine_similarity,
}

METRICS_DICT = deepcopy(METRICS_CLASSIFICATION)
METRICS_DICT.update(METRICS_REGRESSION)


DATAMODULE_DICT = {
    "GraphOGBDataModule": GraphOGBDataModule,
    "MultitaskFromSmilesDataModule": MultitaskFromSmilesDataModule,
    "MultitaskIPUFromSmilesDataModule": MultitaskIPUFromSmilesDataModule,
}
