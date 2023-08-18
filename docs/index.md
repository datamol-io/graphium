# Overview

A deep learning library focused on graph representation learning for real-world chemical tasks.

- ✅ State-of-the-art GNN architectures.
- 🐍 Extensible API: build your own GNN model and train it with ease.
- ⚗️ Rich featurization: powerful and flexible built-in molecular featurization.
- 🧠 Pretrained models: for fast and easy inference or transfer learning.
- ⮔ Read-to-use training loop based on [Pytorch Lightning](https://www.pytorchlightning.ai/).
- 🔌 Have a new dataset? Graphium provides a simple plug-and-play interface. Change the path, the name of the columns to predict, the atomic featurization, and you’re ready to play!

## Installation

### For CPU or GPU
Use [`mamba`](https://github.com/mamba-org/mamba):

```bash
# Install Graphium
mamba install -c conda-forge graphium
```

or pip:

```bash
pip install graphium
```

### For IPU
```bash
# Install Graphcore's SDK and Graphium dependencies in a new environment called `.graphium_ipu`
./install_ipu.sh .graphium_ipu
```

The above step needs to be done once. After that, enable the SDK and the environment as follows:

```bash
source enable_ipu.sh .graphium_ipu
```

Finally, you will need to install graphium with pip
```bash
pip install graphium
```
