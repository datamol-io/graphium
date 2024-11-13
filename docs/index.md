# Overview


A deep learning library focused on graph representation learning for real-world chemical tasks.

- ✅ State-of-the-art GNN architectures.
- 🐍 Extensible API: build your own GNN model and train it with ease.
- ⚗️ Rich featurization: powerful and flexible built-in molecular featurization.
- 🧠 Pretrained models: for fast and easy inference or transfer learning.
- ⮔ Read-to-use training loop based on [Pytorch Lightning](https://www.pytorchlightning.ai/).
- 🔌 Have a new dataset? Graphium provides a simple plug-and-play interface. Change the path, the name of the columns to predict, the atomic featurization, and you’re ready to play!

## Documentation

Visit https://graphium-docs.datamol.io/.

## Installation for users 
### conda-forge
conda-forge is the recommended method for installing Graphium. To install Graphium via conda-forge, run the following command:
```
mamba install graphium -c conda-forge
```

Note: we recommend using [`mamba`](https://github.com/mamba-org/mamba) instead of `conda`. It is a faster and better alternative.

### PyPi
To install Graphium via PyPi, run the following command:
```
pip install graphium
```
Note: the latest available version of Graphium on PyPi is `2.4.7`. This is due to the addition of C++ code in version `3.0.0` that depends on packages only available via conda-forge. There are plans to eventually support Graphium `>=3.0.0` on PyPi.

## Installation for developers

If you are using a GPU, we recommend enforcing the CUDA version that you need with `CONDA_OVERRIDE_CUDA=XX.X`.

```bash
# Install Graphium's dependencies in a new environment named `graphium`
mamba env create -f env.yml -n graphium

# To force the CUDA version to 11.2, or any other version you prefer, use the following command:
# CONDA_OVERRIDE_CUDA=11.2 mamba env create -f env.yml -n graphium

# Activate the mamba environment containing Graphium's dependencies
mamba activate graphium

# Install Graphium in dev mode
pip install --no-deps --no-build-isolation -e .
```

