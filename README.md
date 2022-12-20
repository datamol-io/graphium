<div align="center">
    <img src="docs/images/logo-title.png" height="80px">
    <h3>The Graph Of LIfe Library.</h3>
</div>

---

[![Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/valence-discovery/goli/master?urlpath=lab/tree/docs/tutorials/)
[![PyPI](https://img.shields.io/pypi/v/goli)](https://pypi.org/project/goli-life/)
[![Conda](https://img.shields.io/conda/v/conda-forge/goli?label=conda&color=success)](https://anaconda.org/conda-forge/goli)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/goli-life)](https://pypi.org/project/goli/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/goli)](https://anaconda.org/conda-forge/goli)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/goli-life)](https://pypi.org/project/goli-life/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/valence-discovery/goli/blob/master/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/valence-discovery/goli)](https://github.com/valence-discovery/goli/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/valence-discovery/goli)](https://github.com/valence-discovery/goli/network/members)

A deep learning library focused on graph representation learning for real-world chemical tasks.

- ✅ State-of-the-art GNN architectures.
- 🐍 Extensible API: build your own GNN model and train it with ease.
- ⚗️ Rich featurization: powerful and flexible built-in molecular featurization.
- 🧠 Pretrained models: for fast and easy inference or transfer learning.
- ⮔ Read-to-use training loop based on [Pytorch Lightning](https://www.pytorchlightning.ai/).
- 🔌 Have a new dataset? Goli provides a simple plug-and-play interface. Change the path, the name of the columns to predict, the atomic featurization, and you’re ready to play!

## Try Online

Visit [![Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/valence-discovery/goli/master?urlpath=lab/tree/docs/tutorials/) and try Goli online.

## Documentation

Visit https://valence-discovery.github.io/goli/.

## Installation for developers

### For CPU and GPU developers installs

Use either [`mamba`](https://github.com/mamba-org/mamba) or [`conda`](https://docs.conda.io/en/latest/):

```bash
# Install mamba if unavailable
conda install -c conda-forge mamba

# Install Goli's dependencies in a new environment named `goli_dev`
mamba env create -f env.yml -n goli_dev

# Install Goli in dev mode
conda activate goli_dev
pip install -e .
```

### For IPU developers installs

```bash
mkdir ~/.venv                           # Create the folder for the environment
python3 -m venv ~/.venv/goli_ipu        # Create the environment
source ~/.venv/goli_ipu/bin/activate    # Activate the environment

# Installing the dependencies for the IPU environment
pip install torch==1.10+cpu torchvision==0.11+cpu torchaudio==0.10 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html


# Installing the poptorch SDK. Make sure to change the path
pip install PATH_TO_SDK/poptorch-3.0.0+86945_163b7ce462_ubuntu_20_04-cp38-cp38-linux_x86_64.wh

# Install the remaining requirements
pip install -r requirements.txt

# Install Goli in dev mode
pip install -e .
```

## Training a model

To learn how to train a model, we invite you to look at the documentation, or the jupyter notebooks available [here](https://github.com/valence-discovery/goli/tree/master/docs/tutorials/model_training).

If you are not familiar with [PyTorch](https://pytorch.org/docs) or [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), we highly recommend going through their tutorial first.

## Changelogs

See the latest changelogs at [CHANGELOG.rst](./CHANGELOG.rst).

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).

## Authors

See [AUTHORS.rst](./AUTHORS.rst).
