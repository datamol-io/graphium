# Contribute

We are happy to see that you want to contribute 🤗.
Feel free to open an issue or pull request at any time. But first, follow this page to install Graphium in dev mode.

## Installation for developers

### For CPU and GPU developers

Use [`mamba`](https://github.com/mamba-org/mamba), a preferred alternative to conda, to create your environment:

```bash
# Install Graphium's dependencies in a new environment named `graphium`
mamba env create -f env.yml -n graphium

# Install Graphium in dev mode
mamba activate graphium
pip install --no-deps -e .
```

## Build the documentation

You can build and serve the documentation locally with:

```bash
# Build and serve the doc
mkdocs serve
```
