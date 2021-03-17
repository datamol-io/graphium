# General imports
import os
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import SequentialSampler, SubsetRandomSampler
import numpy as np
import pandas as pd
import operator as op
import yaml
from copy import deepcopy
import hydra
from omegaconf import DictConfig
from warnings import warn
from functools import partial

# Pytorch lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

# Current project imports
# from goli.dgl.dgl_graph_transformer import DGLGraphTransformer
from goli.trainer.metrics import MetricWithThreshold, Thresholder, METRICS_DICT
from goli.dgl.architectures import FullDGLSiameseNetwork, FullDGLNetwork

from goli.utils.utils import is_device_cuda
from goli.trainer.model_wrapper import PredictorModule
from goli.trainer.logger import HyperparamsMetricsTensorBoardLogger
from goli.trainer.reporting import BestEpochFromSummary
from goli.utils.read_file import read_file
from goli.features.featurizer import mol_to_dglgraph
from goli.data.datamodule import DGLFromSmilesDataModule
import goli

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.neighbors import KNeighborsRegressor


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class DimensionalityReduction:
    r"""
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


def _get_device(device):
    if device is None:
        pass
    elif (not torch.cuda.is_available()) and is_device_cuda(device):
        warn(f"GPU not available for device `{device}`, code will run on CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    return device


def config_load_constants(
    main_dir,
    exp_name,
    seed=42,
    data_device="cpu",
    model_device="cpu",
    dtype=torch.float32,
    raise_train_error=True,
):
    seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    exp_name = exp_name

    # Get the dtype
    dtype = DTYPES[dtype]

    # Get the cpu or cuda device
    data_device = _get_device(data_device)
    model_device = _get_device(model_device)

    raise_train_error = raise_train_error

    return data_device, model_device, dtype, exp_name, seed, raise_train_error


def config_load_dataset(
    main_dir,
    train,
    val,
    test,
    label_keys,
    smiles_transform_kwargs,
    smiles_key="SMILES",
    pickle_path=None,  # IGNORED FOR NOW
    subset_max_size=None,  # IGNORED FOR NOW
    data_device="cpu",
    model_device="cpu",
    dtype=torch.float32,
    n_jobs=-1,
    seed=42,
):

    np.random.seed(seed)

    # Load the training set from file
    df_train = read_file(os.path.join(main_dir, train["path"]), **train["kwargs"])
    df_full = df_train

    if val["path"] != train["path"]:
        # Load the validation set from file
        df_val = read_file(os.path.join(main_dir, val["path"]), **val["kwargs"])
        train_idx = np.arange(df_train.shape[0])
        df_full = df_full.append(df_val)
        val_idx = np.arange(df_train.shape[0], df_full.shape[0])

        # Load the test set from file
        if test["path"] != train["path"]:
            df_test = read_file(os.path.join(main_dir, test["path"]), **test["kwargs"])
            test_idx = np.arange(df_full.shape[0], df_full.shape[0] + df_test.shape[0])
            df_full = df_full.append(df_test)

        else:
            raise ValueError(
                "Cannot create a test set from the training set if a different validation set is provided"
            )

    else:
        if test["path"] != train["path"]:

            if (train["idxfile_or_split"] + val["idxfile_or_split"]) != 1:
                raise ValueError(
                    "When a different test path is provided, the train split and test split must sum to 1"
                )

            # Generate train/val datasets from random splits
            shuffle_idx = np.random.choice(df_full.shape[0], df_full.shape[0], replace=False)
            max_train_idx = int(np.floor(train["idxfile_or_split"] * df_full.shape[0]))
            train_idx = shuffle_idx[:max_train_idx]
            val_idx = shuffle_idx[max_train_idx:]

            # Load the test set from file
            df_test = read_file(os.path.join(main_dir, test["path"]), **test["kwargs"])
            test_idx = np.arange(df_full.shape[0], df_full.shape[0] + df_test.shape[0])
            df_full = df_full.append(df_test)

        else:
            if (train["idxfile_or_split"] + val["idxfile_or_split"] + test["idxfile_or_split"]) != 1:
                raise ValueError("All splits must sum to 1")

            # Generate train/val/test datasets from random splits
            shuffle_idx = np.random.choice(df_full.shape[0], df_full.shape[0], replace=False)
            max_train_idx = int(np.floor(train["idxfile_or_split"] * df_full.shape[0]))
            max_val_idx = max_train_idx + int(np.floor(train["idxfile_or_split"] * df_full.shape[0]))
            train_idx = shuffle_idx[:max_train_idx]
            val_idx = shuffle_idx[max_train_idx:max_val_idx]
            test_idx = shuffle_idx[max_val_idx:]

    smiles_transform = partial(mol_to_dglgraph, **smiles_transform_kwargs)

    datamodule = DGLFromSmilesDataModule(
        smiles=df_full[smiles_key],
        labels=df_full[label_keys],
        weights=None,
        seed=seed,
        train_batch_size=train["batch_size"],
        test_batch_size=test["batch_size"],
        train_sampler=SubsetRandomSampler(train_idx),
        val_sampler=SubsetRandomSampler(val_idx),
        test_sampler=SubsetRandomSampler(test_idx),
        model_device=model_device,
        data_device=data_device,
        dtype=dtype,
        n_jobs=n_jobs,
        smiles_transform=smiles_transform,
    )

    num_node_feats = datamodule.num_node_feats
    num_edge_feats = datamodule.num_edge_feats

    return datamodule, num_node_feats, num_edge_feats


def config_load_architecture(
    model_type,
    pre_nn_kwargs,
    post_nn_kwargs,
    gnn_kwargs,
    in_dim_nodes,
    in_dim_edges=0,
    model_device="cpu",
    dtype=torch.float32,
    **kwargs,
):

    # Select the right architecture
    if model_type.lower() == "fulldglnetwork":
        model_class = FullDGLNetwork
    elif model_type.lower() == "fulldglsiamesenetwork":
        model_class = FullDGLSiameseNetwork
    else:
        raise ValueError(f"Unsupported model_type=`{model_type}`")

    gnn_kwargs = dict(gnn_kwargs)
    post_nn_kwargs = dict(post_nn_kwargs) if post_nn_kwargs is not None else None
    # Define the input dimension
    if pre_nn_kwargs is not None:
        pre_nn_kwargs = dict(pre_nn_kwargs)
        pre_nn_kwargs.setdefault("in_dim", in_dim_nodes)
    else:
        gnn_kwargs.setdefault("in_dim", in_dim_nodes)

    gnn_kwargs.setdefault("in_dim_edges", in_dim_edges)

    # Build the graph network
    model = model_class(
        gnn_kwargs=gnn_kwargs,
        pre_nn_kwargs=pre_nn_kwargs,
        post_nn_kwargs=post_nn_kwargs,
        **kwargs,
    )
    model = model.to(device=model_device, dtype=dtype)

    # Make sure the input dimensions are consistent
    assert model.in_dim == in_dim_nodes
    assert model.in_dim_edges == in_dim_edges

    return model


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


def config_load_predictor(
    cfg_reg, metrics, metrics_on_progress_bar, model, layer_name, train_dt, val_dt, device, dtype
):
    # Defining the model_wrapper

    reg_kwargs = dict(deepcopy(cfg_reg))
    siamese = reg_kwargs.pop("siamese")
    model_wrapper = PredictorModule(
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


def config_load_training(cfg_trainer, model_wrapper):
    early_stopping = EarlyStopping(**cfg_trainer["early_stopping"])
    checkpoint_callback = ModelCheckpoint(**cfg_trainer["model_checkpoint"])
    logger = HyperparamsMetricsTensorBoardLogger(**cfg_trainer["tensorboard_logs"])
    training_results = BestEpochFromSummary(model_wrapper.metrics)

    trainer = Trainer(
        logger=logger,
        callbacks=[early_stopping, training_results, checkpoint_callback],
        terminate_on_nan=True,
        **cfg_trainer["trainer"],
    )

    return trainer
