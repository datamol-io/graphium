# Contribute

The below documents the development lifecycle of Graphium.

## Setup a dev environment

```bash
mamba env create -n graphium -f env.yml
mamba activate graphium

pip install --no-deps -e .
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
