<div align="center">
    <img src="docs/images/logo-title.png" height="80px">
    <h3>The Graph Of LIfe Library.</h3>
</div>

---

[![Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/valence-discovery/goli/master?urlpath=lab/tree/docs/tutorials/)
[![PyPI](https://img.shields.io/pypi/v/goli)](https://pypi.org/project/goli/)
[![Conda](https://img.shields.io/conda/v/conda-forge/goli?label=conda&color=success)](https://anaconda.org/conda-forge/goli)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/goli)](https://pypi.org/project/goli/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/goli)](https://anaconda.org/conda-forge/goli)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/goli)](https://pypi.org/project/goli/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/valence-discovery/goli/blob/master/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/valence-discovery/goli)](https://github.com/valence-discovery/goli/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/valence-discovery/goli)](https://github.com/valence-discovery/goli/network/members)

goli is a python library to work with molecules. It's a layer built on top of [RDKit](https://www.rdkit.org/) and aims to be as light as possible.

- 🐍 Simple pythonic API
- ⚗️ RDKit first: all you manipulate are `rdkit.Chem.Mol` objects.
- ✅ Manipulating molecules often rely on many options; goli provides good defaults by design.
- 🧠 Performance matters: built-in efficient parallelization when possible with optional progress bar.
- 🕹️ Modern IO: out-of-the-box support for remote paths using `fsspec` to read and write multiple formats (sdf, xlsx, csv, etc).

## Try Online

Visit [![Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/valence-discovery/goli/master?urlpath=lab/tree/docs/*tutorials*.ipynb) and try goli online.

## Documentation

Visit https://valence-discovery.github.io/goli/.

## Installation

Use conda:

```bash
mamba install -c conda-forge goli
```

## Changelogs

See the latest changelogs at [CHANGELOG.rst](./CHANGELOG.rst).

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).

## Authors

See [AUTHORS.rst](./AUTHORS.rst).
