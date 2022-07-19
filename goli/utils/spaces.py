from copy import deepcopy
import torch
import torch.optim.lr_scheduler as sc
import torchmetrics.functional as met

from goli.nn.base_layers import FCLayer
from goli.data.datamodule import DGLFromSmilesDataModule, DGLOGBDataModule, MultitaskDGLFromSmilesDataModule

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
    ResidualConnectionRandom,
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
    "random": ResidualConnectionRandom,
}

LOSS_DICT = {
    "mse": torch.nn.MSELoss(),
    "bce": torch.nn.BCELoss(),
    "l1": torch.nn.L1Loss(),
    "mae": torch.nn.L1Loss(),
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
    "DGLFromSmilesDataModule": DGLFromSmilesDataModule,
    "DGLOGBDataModule": DGLOGBDataModule,
    "MultitaskDGLFromSmilesDataModule": MultitaskDGLFromSmilesDataModule
}
