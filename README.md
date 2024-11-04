<div align="center">
    <img src="docs/images/banner-tight.png" height="200px">
    <h3>Scaling molecular GNNs to infinity</h3>
</div>

---

[![PyPI](https://img.shields.io/pypi/v/graphium)](https://pypi.org/project/graphium/)
[![Conda](https://img.shields.io/conda/v/conda-forge/graphium?label=conda&color=success)](https://anaconda.org/conda-forge/graphium)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/graphium)](https://pypi.org/project/graphium/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/graphium)](https://anaconda.org/conda-forge/graphium)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/datamol-io/graphium/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/datamol-io/graphium)](https://github.com/datamol-io/graphium/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/datamol-io/graphium)](https://github.com/datamol-io/graphium/network/members)
[![test](https://github.com/datamol-io/graphium/actions/workflows/test.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/test.yml)
[![release](https://github.com/datamol-io/graphium/actions/workflows/release.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/release.yml)
[![code-check](https://github.com/datamol-io/graphium/actions/workflows/code-check.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/code-check.yml)
[![doc](https://github.com/datamol-io/graphium/actions/workflows/doc.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/doc.yml)
[![codecov](https://codecov.io/gh/datamol-io/graphium/branch/main/graph/badge.svg?token=bHOkKY5Fze)](https://codecov.io/gh/datamol-io/graphium)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

A deep learning library focused on graph representation learning for real-world chemical tasks.

- ✅ State-of-the-art GNN architectures.
- 🐍 Extensible API: build your own GNN model and train it with ease.
- ⚗️ Rich featurization: powerful and flexible built-in molecular featurization.
- 🧠 Pretrained models: for fast and easy inference or transfer learning.
- ⮔ Read-to-use training loop based on [Pytorch Lightning](https://www.pytorchlightning.ai/).
- 🔌 Have a new dataset? Graphium provides a simple plug-and-play interface. Change the path, the name of the columns to predict, the atomic featurization, and you’re ready to play!

## Documentation

Visit https://graphium-docs.datamol.io/.

## Installation for developers

### For CPU and GPU developers

Use [`mamba`](https://github.com/mamba-org/mamba), a faster and better alternative to `conda`.

If you are using a GPU, we recommend enforcing the CUDA version that you need with `CONDA_OVERRIDE_CUDA=XX.X`.

```bash
# Install Graphium's dependencies in a new environment named `graphium`
mamba env create -f env.yml -n graphium

# To force the CUDA version to 11.2, or any other version you prefer, use the following command:
# CONDA_OVERRIDE_CUDA=11.2 mamba env create -f env.yml -n graphium

# Install Graphium in dev mode
mamba activate graphium
pip install --no-deps -e .
```

## Training a model

To learn how to train a model, we invite you to look at the documentation, or the jupyter notebooks available [here](https://github.com/datamol-io/graphium/tree/master/docs/tutorials/model_training).

If you are not familiar with [PyTorch](https://pytorch.org/docs) or [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), we highly recommend going through their tutorial first.

## Running an experiment

### Datasets

Graphium provides configs for 2 datasets: `toymix` and `largemix`. 
`Toymix` uses 3 datasets, which are referenced in datamodule [here](https://github.com/datamol-io/graphium/blob/d12df7e06828fa7d7f8792141d058a60b2b2d258/expts/hydra-configs/tasks/loss_metrics_datamodule/toymix.yaml#L59-L102). Its datasets and their splits files can be downloaded from here:

```bash
# Change or make the directory to where the dataset is to be downloaded
cd expts/data/neurips2023/small-dataset

# QM9 
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/qm9.csv.gz
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/qm9_random_splits.pt

# Tox21
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/Tox21-7k-12-labels.csv.gz
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/Tox21_random_splits.p

# Zinc
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/ZINC12k.csv.gz
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/ZINC12k_random_splits.pt
```

`Largemix` uses datasets referenced in datamodule [here](https://github.com/datamol-io/graphium/blob/e887176f71ee95c3b82f8f6b56c706eaa9765bf1/expts/hydra-configs/tasks/loss_metrics_datamodule/largemix.yaml#L82C1-L155C37). Its datasets and their splits files can be downloaded from here:


```bash
# Change or make the directory to where the dataset is to be downloaded
cd ../data/graphium/large-dataset/

# L1000_VCAP
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/LINCS_L1000_VCAP_0-4.csv.gz
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/l1000_vcap_random_splits.pt

# L1000_MCF7
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/LINCS_L1000_MCF7_0-4.csv.gz
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/l1000_mcf7_random_splits.pt

# PCBA_1328
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/PCBA_1328_1564k.parquet
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/pcba_1328_random_splits.pt

# PCQM4M_G25
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/PCQM4M_G25_N4.parquet
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/pcqm4m_g25_n4_random_splits.pt

#PCQM4M_N4
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/PCQM4M_G25_N4.parquet
wget https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/pcqm4m_g25_n4_random_splits.pt
```
These datasets can be used further for pretraining.

### Pretraining 

We have setup Graphium with `hydra` for managing config files. To run an experiment go to the `expts/` folder. For example, to benchmark a GCN on the ToyMix dataset run
```bash
graphium-train architecture=toymix tasks=toymix training=toymix model=gcn
```
To change parameters specific to this experiment like switching from `fp16` to `fp32` precision, you can either override them directly in the CLI via
```bash
graphium-train architecture=toymix tasks=toymix training=toymix model=gcn trainer.trainer.precision=32
```
or change them permanently in the dedicated experiment config under `expts/hydra-configs/toymix_gcn.yaml`.
Integrating `hydra` also allows you to quickly switch between accelerators. E.g., running
```bash
graphium-train architecture=toymix tasks=toymix training=toymix model=gcn accelerator=gpu
```
automatically selects the correct configs to run the experiment on GPU.
To use Largemix dataset instead, replace `toymix` to `largemix` in the above commmands.

To use a config file you built from scratch you can run
```bash
graphium-train --config-path [PATH] --config-name [CONFIG]
```
Thanks to the modular nature of `hydra` you can reuse many of our config settings for your own experiments with Graphium.

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).

## Documentation

- Diagram for data processing in Graphium.

<img src="docs/images/datamodule.png" alt="Data Processing Chart" width="60%" height="60%">

- Diagram for Muti-task network in Graphium

<img src="docs/images/full_graph_network.png" alt="Full Graph Multi-task Network" width="80%" height="80%">
