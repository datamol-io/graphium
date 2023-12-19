from copy import deepcopy
import torch
import torch.optim.lr_scheduler as sc
import torchmetrics.functional as TorchMetrics

import graphium.nn.base_layers as BaseLayers
import graphium.nn.ensemble_layers as EnsembleLayers
import graphium.nn.architectures as Architectures
import graphium.utils.custom_lr as CustomLR
import graphium.data.datamodule as Datamodules
import graphium.ipu.ipu_losses as IPULosses
import graphium.ipu.ipu_metrics as Metrics
import graphium.nn.pyg_layers as PygLayers
import graphium.nn.residual_connections as Residuals
import graphium.nn.encoders as Encoders
import graphium.trainer.losses as Losses

PE_ENCODERS_DICT = {
    "laplacian_pe": Encoders.laplace_pos_encoder.LapPENodeEncoder,
    "mlp": Encoders.mlp_encoder.MLPEncoder,
    "signnet": Encoders.signnet_pos_encoder.SignNetNodeEncoder,
    "gaussian_kernel": Encoders.gaussian_kernel_pos_encoder.GaussianKernelPosEncoder,
    "bessel_kernel": Encoders.bessel_pos_encoder.BesselSphericalPosEncoder,
}


FC_LAYERS_DICT = {
    "fc": BaseLayers.FCLayer,
}

ENSEMBLE_FC_LAYERS_DICT = {
    "ens-fc": EnsembleLayers.EnsembleFCLayer,
}

PYG_LAYERS_DICT = {
    "pyg:gcn": PygLayers.GCNConvPyg,
    "pyg:gin": PygLayers.GINConvPyg,
    "pyg:gine": PygLayers.GINEConvPyg,
    "pyg:gated-gcn": PygLayers.GatedGCNPyg,
    "pyg:pna-msgpass": PygLayers.PNAMessagePassingPyg,
    "pyg:gps": PygLayers.GPSLayerPyg,
    "pyg:dimenet": PygLayers.DimeNetPyg,
    "pyg:mpnnplus": PygLayers.MPNNPlusPyg,
}

LAYERS_DICT = deepcopy(FC_LAYERS_DICT)
LAYERS_DICT.update(deepcopy(PYG_LAYERS_DICT))

ENSEMBLE_LAYERS_DICT = deepcopy(ENSEMBLE_FC_LAYERS_DICT)

RESIDUALS_DICT = {
    "none": Residuals.ResidualConnectionNone,
    "simple": Residuals.ResidualConnectionSimple,
    "weighted": Residuals.ResidualConnectionWeighted,
    "concat": Residuals.ResidualConnectionConcat,
    "densenet": Residuals.ResidualConnectionDenseNet,
    "random": Residuals.ResidualConnectionRandom,
}

LOSS_DICT = {
    "bce": torch.nn.BCELoss,
    "bce_logits": torch.nn.BCEWithLogitsLoss,
    "mse": torch.nn.MSELoss,
    "bce": torch.nn.BCELoss,
    "l1": torch.nn.L1Loss,
    "mae": torch.nn.L1Loss,
    "hybrid_ce": Losses.HybridCELoss,
    "bce_ipu": IPULosses.BCELossIPU,
    "bce_logits_ipu": IPULosses.BCEWithLogitsLossIPU,
    "mse_ipu": IPULosses.MSELossIPU,
    "mae_ipu": IPULosses.L1LossIPU,
    "l1_ipu": IPULosses.L1LossIPU,
    "hybrid_ce_ipu": IPULosses.HybridCELossIPU,
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
    "ConstantLR": sc.ConstantLR,
    "WarmUpLinearLR": CustomLR.WarmUpLinearLR,
}

METRICS_CLASSIFICATION = {
    "accuracy": TorchMetrics.accuracy,
    "averageprecision": TorchMetrics.average_precision,
    "auroc": TorchMetrics.auroc,
    "confusionmatrix": TorchMetrics.confusion_matrix,
    "f1": TorchMetrics.f1_score,
    "fbeta": TorchMetrics.fbeta_score,
    "precisionrecallcurve": TorchMetrics.precision_recall_curve,
    "precision": TorchMetrics.precision,
    "recall": TorchMetrics.recall,
    "mcc": TorchMetrics.matthews_corrcoef,
    "auroc_ipu": Metrics.auroc_ipu,
    "accuracy_ipu": Metrics.accuracy_ipu,
    "average_precision_ipu": Metrics.average_precision_ipu,
    "f1_ipu": Metrics.f1_score_ipu,
    "fbeta_ipu": Metrics.fbeta_score_ipu,
    "precision_ipu": Metrics.precision_ipu,
    "recall_ipu": Metrics.recall_ipu,
}

METRICS_REGRESSION = {
    "mae": TorchMetrics.mean_absolute_error,
    "mape": TorchMetrics.mean_absolute_percentage_error,
    "mse": TorchMetrics.mean_squared_error,
    "msle": TorchMetrics.mean_squared_log_error,
    "pearsonr": TorchMetrics.pearson_corrcoef,
    "spearmanr": TorchMetrics.spearman_corrcoef,
    "r2_score": TorchMetrics.r2_score,
    "cosine": TorchMetrics.cosine_similarity,
    "pearsonr_ipu": Metrics.pearson_ipu,
    "spearmanr_ipu": Metrics.spearman_ipu,
    "r2_score_ipu": Metrics.r2_score_ipu,
    "mae_ipu": Metrics.mean_absolute_error_ipu,
    "mse_ipu": Metrics.mean_squared_error_ipu,
}

METRICS_DICT = deepcopy(METRICS_CLASSIFICATION)
METRICS_DICT.update(METRICS_REGRESSION)


DATAMODULE_DICT = {
    "GraphOGBDataModule": Datamodules.GraphOGBDataModule,
    "MultitaskFromSmilesDataModule": Datamodules.MultitaskFromSmilesDataModule,
    "ADMETBenchmarkDataModule": Datamodules.ADMETBenchmarkDataModule,
    "FakeDataModule": Datamodules.FakeDataModule,
}

GRAPHIUM_PRETRAINED_MODELS_DICT = {
    "dummy-pretrained-model": "tests/dummy-pretrained-model.ckpt",  # dummy model used for testing purposes
}

FINETUNING_HEADS_DICT = {
    "mlp": Architectures.FeedForwardNN,
    "gnn": Architectures.FeedForwardPyg,
    "task_head": Architectures.TaskHeads,
    "ens-mlp": Architectures.EnsembleFeedForwardNN,
}
