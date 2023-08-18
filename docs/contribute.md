# Contribute

The below documents the development lifecycle of Graphium.

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

## Build the documentation

You can build and serve the documentation locally with:

```bash
# Build and serve the doc
make serve
```
