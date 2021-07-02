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

Visit [![Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/valence-discovery/goli/master?urlpath=lab/tree/docs/tutorials/) and try goli online.

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

## Changelogs

See the latest changelogs at [CHANGELOG.rst](./CHANGELOG.rst).

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).

## Authors

See [AUTHORS.rst](./AUTHORS.rst).
