<div align="center">
    <img src="docs/images/logo.png" height="200px">
    <h3>Scaling molecular GNNs to infinity</h3>
</div>

---

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/sdGggS)
[![PyPI](https://img.shields.io/pypi/v/graphium)](https://pypi.org/project/graphium/)
[![Conda](https://img.shields.io/conda/v/conda-forge/graphium?label=conda&color=success)](https://anaconda.org/conda-forge/graphium)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/graphium)](https://pypi.org/project/graphium/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/graphium)](https://anaconda.org/conda-forge/graphium)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/datamol-io/graphium/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/datamol-io/graphium)](https://github.com/datamol-io/graphium/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/datamol-io/graphium)](https://github.com/datamol-io/graphium/network/members)
[![test](https://github.com/datamol-io/graphium/actions/workflows/test.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/test.yml)
[![test-ipu](https://github.com/datamol-io/graphium/actions/workflows/test_ipu.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/test_ipu.yml)
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

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/sdGggS)

You can try running Graphium on Graphcore IPUs for free on Gradient by clicking on the button above.

## Installation for developers

### For CPU and GPU developers

Use [`mamba`](https://github.com/mamba-org/mamba):

```bash
# Install Graphium's dependencies in a new environment named `graphium`
mamba env create -f env.yml -n graphium

# Install Graphium in dev mode
mamba activate graphium
pip install --no-deps -e .
```

### For IPU developers
```bash
# Install Graphcore's SDK and Graphium dependencies in a new environment called `.graphium_ipu`
./install_ipu.sh .graphium_ipu
```

The above step needs to be done once. After that, enable the SDK and the environment as follows:

```bash
source enable_ipu.sh .graphium_ipu
```

## The Graphium CLI

Installing `graphium` makes two CLI tools available: `graphium` and `graphium-train`. These CLI tools make it easy to access advanced functionality, such as _training a model_,  _extracting fingerprints from a pre-trained model_ or _precomputing the dataset_. For more information, visit [the documentation](https://graphium-docs.datamol.io/stable/cli/reference.html).

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).

## Documentation

- Diagram for data processing in Graphium.

<img src="docs/images/datamodule.png" alt="Data Processing Chart" width="60%" height="60%">

- Diagram for Muti-task network in Graphium

<img src="docs/images/full_graph_network.png" alt="Full Graph Multi-task Network" width="80%" height="80%">
