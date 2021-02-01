# %%
# General imports
import os
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import operator as op
import yaml
from copy import deepcopy
import hydra
from omegaconf import DictConfig
from warnings import warn

# Pytorch lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

# Current project imports
from goli.dgl.dgl_graph_transformer import DGLGraphTransformer
from goli.trainer.metrics import MetricWithThreshold, Thresholder, METRICS_DICT
from goli.dgl.architectures import SiameseGraphNetwork, DGLGraphNetwork
from goli.dgl.datasets import load_csv_to_dgl_dataset
from goli.dgl.utils import DGLCollate
from goli.trainer.model_wrapper import ModelWrapper
from goli.trainer.logger import HyperparamsMetricsTensorBoardLogger
from goli.trainer.reporting import BestEpochFromSummary
import goli

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.neighbors import KNeighborsRegressor


class DimensionalityReduction:
    """
    Dimensionality reduction helper. It wraps around a standart reduction like pca, tsne, umap and adds an inverse
    transform either by pre-implemented inverse transform or a knn based inverse transform.

    @param type: type of the dimensionality reduction. Can be None, 'pca', 'tsne', 'umap'. If None is given then the
    identity is used as reduction
    @param y: The data to fit the the reduction
    @param inverse: The inverse transform. Can be 'buildin', 'knn'
    """

    def __init__(self, cfg):
        self._type = cfg["type"]
        self._inverse = cfg["inverse"]

        # dimensionalty is eigther int or a float between 0 and 1
        dim = cfg["dimensionality"]

        # Reduction
        if self._type == "pca":
            self._reduction = PCA(dim)
        if self._type == "tsne":
            self._reduction = TSNE(dim)
        if self._type == "umap":
            self._reduction = UMAP(n_components=dim)

        # initialize knn regressor if needed
        if self._inverse == "knn":
            self._knn = KNeighborsRegressor(5)

    def fit(self, X):
        y = self._reduction.fit_transform(X)
        if self._inverse == "knn":
            self._knn.fit(y, X)

    def fit_transform(self, X):
        if self._inverse == "buildin":
            return self._reduction.fit_transform(X)
        if self._inverse == "knn":
            y = self._reduction.fit_transform(X)
            self._knn.fit(y, X)
            return y
        raise NotImplementedError()

    def transform(self, X):
        return self._reduction.transform(X)

    def inverse_transform(self, X):
        if self._inverse == "buildin":
            return self.reduction.inverse_transform(X)
        if self._inverse == "knn":
            return self._knn.predict(X.detach().numpy())


def config_load_dimred(cfg_dimred):
    dimred = DimensionalityReduction(cfg_dimred)
    return dimred


def config_load_constants(cfg_const, main_dir):
    np.random.seed(cfg_const["seed"])
    torch.manual_seed(cfg_const["seed"])
    exp_name = cfg_const["exp_name"]

    # Get the cpu or cuda device
    dtype = torch.float32
    if (not torch.cuda.is_available()) and (cfg_const["device"].lower() != "cpu"):
        warn("GPU not available for device name `{}`, code will run on CPU".format(cfg_const["device"]))
        device = torch.device("cpu")
    elif cfg_const["device"].lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(cfg_const["device"])

    # Read the gnns configurations
    config_gnns_path = os.path.join(main_dir, cfg_const["config_gnns"])
    with open(config_gnns_path) as cfg_file:
        cfg_gnns = yaml.load(cfg_file, Loader=yaml.FullLoader)

    return device, dtype, exp_name, cfg_gnns


def config_load_datasets(cfg_data, main_dir, device, train_val_test=["train", "val"]):
    trans = DGLGraphTransformer()

    if isinstance(train_val_test, str):
        train_val_test = [train_val_test]

    datasets = []
    for dt_name in train_val_test:
        dt_name = dt_name.lower()
        this_dt = load_csv_to_dgl_dataset(
            data_dir=os.path.join(main_dir, cfg_data[dt_name]["data_dir"]),
            name=cfg_data[dt_name]["filename"],
            smiles_cols=cfg_data["smiles_cols"],
            y_col=cfg_data["y_col"],
            max_mols=cfg_data[dt_name]["nrows"],
            trans=trans,
            device="cpu",
        )
        datasets.append(this_dt)

    return trans, datasets


def config_load_siamese_gnn(cfg_model, cfg_gnns, in_dim, out_dim, device, dtype):
    layer_name = cfg_model["layer_name"]
    gnn_layer_kwargs = cfg_gnns[layer_name]

    gnn_kwargs = dict(
        in_dim=in_dim, out_dim=cfg_model["dist_vector_size"], **cfg_model["gnn_kwargs"], **gnn_layer_kwargs
    )
    hidden_dim = gnn_kwargs.pop("hidden_dim")
    hidden_depth = gnn_kwargs.pop("hidden_depth")
    gnn_kwargs["hidden_dims"] = [hidden_dim] * hidden_depth

    gnn = SiameseGraphNetwork(
        gnn_kwargs=gnn_kwargs,
        gnn_architecture=cfg_model["gnn_architecture"],
        dist_method=cfg_model["dist_method"],
    )
    gnn = gnn.to(device=device, dtype=dtype)

    return gnn, layer_name


def config_load_gnn(cfg_model, cfg_gnns, in_dim, out_dim, device, dtype):
    layer_name = cfg_model["layer_name"]
    gnn_layer_kwargs = cfg_gnns[layer_name]

    # Parse the GNN arguments
    gnn_kwargs = dict(in_dim=in_dim, **cfg_model["gnn_kwargs"], **gnn_layer_kwargs)
    gnn_hidden_dim = gnn_kwargs.pop("hidden_dim")
    gnn_hidden_depth = gnn_kwargs.pop("hidden_depth")
    gnn_kwargs["hidden_dims"] = [gnn_hidden_dim] * gnn_hidden_depth
    gnn_kwargs["out_dim"] = gnn_hidden_dim

    # Parse the LNN arguments
    lnn_kwargs = dict(in_dim=gnn_kwargs["out_dim"], out_dim=out_dim, **cfg_model["lnn_kwargs"])
    lnn_hidden_dim = lnn_kwargs.pop("hidden_dim")
    lnn_hidden_depth = lnn_kwargs.pop("hidden_depth")
    lnn_kwargs["hidden_dims"] = [lnn_hidden_dim] * lnn_hidden_depth

    # Build the graph network
    gnn = DGLGraphNetwork(
        gnn_kwargs=gnn_kwargs, lnn_kwargs=lnn_kwargs, gnn_architecture=cfg_model["gnn_architecture"]
    )
    gnn = gnn.to(device=device, dtype=dtype)

    return gnn, layer_name


def config_load_metrics(cfg_metrics):
    # Define the metrics to use
    selected_metrics = dict(deepcopy(cfg_metrics["metrics_dict"]))
    metrics = {}
    for metric_name, metric_kwargs in selected_metrics.items():
        metric_kwargs = dict(metric_kwargs) if metric_kwargs is not None else {}
        threshold = metric_kwargs.pop("threshold", None)
        metric_object = METRICS_DICT[metric_name](**metric_kwargs)

        if threshold is None:
            metrics[metric_name] = metric_object
        else:
            operator = "greater" if threshold["above_th"] else "lower"
            th_on_pred = threshold["th_on_pred"]
            th_on_target = threshold["th_on_target"]
            ths = threshold["ths"]

            for th in ths:
                metric_th_name = f"{metric_name}-{operator}-{th}"
                thresholder = Thresholder(
                    threshold=th, operator=operator, th_on_pred=th_on_pred, th_on_target=th_on_target
                )
                metrics[metric_th_name] = MetricWithThreshold(metric_object, thresholder)

    metrics_on_progress_bar = cfg_metrics["metrics_on_progress_bar"]

    return metrics, metrics_on_progress_bar


def config_load_model_wrapper(
    cfg_reg, metrics, metrics_on_progress_bar, model, layer_name, train_dt, val_dt, device, dtype
):
    # Defining the model_wrapper

    reg_kwargs = dict(deepcopy(cfg_reg))
    siamese = reg_kwargs.pop("siamese")
    model_wrapper = ModelWrapper(
        model=model,
        dataset=train_dt,
        validation_split=val_dt,
        collate_fn=DGLCollate(device=device, siamese=siamese),
        metrics=metrics,
        metrics_on_progress_bar=metrics_on_progress_bar,
        additional_hparams={"layer_fullname": layer_name},
        **reg_kwargs,
    )
    model_wrapper = model_wrapper.to(device=device, dtype=dtype)

    return model_wrapper


def config_load_training(cfg_train, model_wrapper):
    early_stopping = EarlyStopping(**cfg_train["early_stopping"])
    checkpoint_callback = ModelCheckpoint(**cfg_train["model_checkpoint"])
    logger = HyperparamsMetricsTensorBoardLogger(save_dir="lightning_logs")
    training_results = BestEpochFromSummary(model_wrapper.metrics)

    trainer = Trainer(
        logger=logger,
        callbacks=[early_stopping, training_results, checkpoint_callback],
        terminate_on_nan=True,
        **cfg_train["trainer"],
    )

    return trainer
