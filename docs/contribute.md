# Contribute

The below documents the development lifecycle of Datamol.

## Setup a dev environment

```bash
conda create -n goli
conda activate goli

mamba env update -f env.yml

conda deactivate && conda activate goli
pip install -e .
```

## Run tests

```bash
pytest
```

## Build the documentation

You can build and serve the documentation locally with:

```bash
# Build and serve the doc
mike serve
```

## Release a new version

- Run check: `rever check`.
- Bump and release new version: `rever VERSION_NUMBER`.
- Releasing a new version will do the following things in that order:
  - Update `AUTHORS.rst`.
  - Update `CHANGELOG.rst`.
  - Bump the version number in `setup.py` and `_version.py`.
  - Add a git tag.
  - Push the git tag.
  - Add a new release on the GH repo associated with the git tag.
  - Update the conda forge feedstock.
