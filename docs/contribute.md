# Contribute

We are happy to see that you want to contribute 🤗.
Feel free to open an issue or pull request at any time. But first, follow this page to install Graphium in dev mode.

## Installation for developers

### For CPU and GPU developers

Use [`mamba`](https://github.com/mamba-org/mamba), a preferred alternative to conda, to create your environment:

```bash
# Install Graphium's dependencies in a new environment named `graphium`
# Replace `ENV_FILE` with the path to the env file in the `requirements` folder
# which matches your machine's OS
mamba env create -f ENV_FILE -n graphium

# Activate the mamba environment containing graphium's dependencies
mamba activate graphium

# Install Graphium in dev mode
pip install --no-deps --no-build-isolation -e .
```

## Build the documentation

You can build and serve the documentation locally with:

```bash
# Build and serve the doc
doxygen Doxyfile
mkdocs serve
```
