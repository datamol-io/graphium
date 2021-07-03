# Overview

A deep learning library focused on graph representation learning for real-world chemical tasks.

- ✅ State-of-the-art GNN architectures.
- 🐍 Extensible API: build your own GNN model and train it with ease.
- ⚗️ Rich featurization: powerful and flexible built-in molecular featurization.
- 🧠 Pretrained models: for fast and easy inference or transfer learning.
- ⮔ Read-to-use training loop based on [Pytorch Lightning](https://www.pytorchlightning.ai/).
- 🔌 Have a new dataset? Goli provides a simple plug-and-play interface. Change the path, the name of the columns to predict, the atomic featurization, and you’re ready to play!

## Try Online

Visit [![Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/valence-discovery/goli/master?urlpath=lab/tree/docs/*tutorials*.ipynb) and try Goli online.

## Documentation

Visit https://valence-discovery.github.io/goli/.

## Installation

Use either [`mamba`](https://github.com/mamba-org/mamba) or [`conda`](https://docs.conda.io/en/latest/):

```bash
# Install DGL from https://github.com/dmlc/dgl/#installation
mamba install -c dglteam dgl

# Install Goli
mamba install -c conda-forge goli
```

or pip:

```bash
pip install goli-life
```
