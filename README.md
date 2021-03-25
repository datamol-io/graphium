# GOLI

GOLI (Graph Of LIfe) is a deep learning library focused on graph representation learning for real-world chemical tasks.

## Getting started

### Environment setup

Setup conda and mamba:

```bash
# Install conda from https://docs.conda.io/en/latest/miniconda.html

# Setup conda-forge on your system
conda config --add channels conda-forge
conda config --set channel_priority strict

# Install mamba: a fast alternative to conda
conda install mamba
```

Clone this repository and setup your working environment:

```bash
# Clone repo
git clone git@github.com:valence-discovery/goli.git
cd goli/

# Create conda env
conda create -n goli

# Activate conda env
conda activate goli

# Install goli deps
mamba env update -f env.yml

# Install goli in dev mode
pip install -e .
```

### Data setup

Then, you need to download the data needed to run the code. Right now, we have 2 sets of data folders, present in the link [here](https://drive.google.com/drive/folders/1RrbNZkEE2rf41_iroa1LbIyegW00h3Ql?usp=sharing).

- **micro_ZINC** (Synthetic dataset)
  - A small subset (1000 mols) of the ZINC dataset
  - The score is the subtraction of the computed LogP and the synthetic accessibility score SA
  - The data must be downloaded to the folder `./goli/data/micro_ZINC/`

- **ZINC_bench_gnn** (Synthetic dataset)
  - A subset (12000 mols) of the ZINC dataset
  - The score is the subtraction of the computed LogP and the synthetic accessibility score SA
  - These are the same 12k molecules provided by the [Benchmarking-gnn](https://github.com/graphdeeplearning/benchmarking-gnns) repository.
    - We provide the pre-processed graphs in `ZINC_bench_gnn/data_from_benchmark`
    - We provide the SMILES in `ZINC_bench_gnn/smiles_score.csv`, with the train-val-test indexes in the file `indexes_train_val_test.csv`.
      - The first 10k elements are the training set
      - The next 1k the valid set
      - The last 1k the test set.
  - The data must be downloaded to the folder `./goli/data/ZINC_bench_gnn/`

Then, you can run the main file to make sure that all the dependancies are correctly installed and that the code works as expected.

```bash
python expts/main_micro_zinc.py
```

## Documentation

Find the documentation at [https://valence-discovery--goli.github.privpage.net/](https://valence-discovery--goli.github.privpage.net/).

## Maintainers

- Dominique Beaini (dominique@valencediscovery.com)
- Hadrien Mary (hadrien@valencediscovery.com)

## Development Lifecycle

### Build the documentation

You can build and serve the documentation locally with:

```bash
# Build and serve the doc
mkdocs serve
```

### Release a new version

- Install [rever](https://regro.github.io/rever-docs): `conda install -y rever`.
- Run check: `rever check`.
- Bump and release a new version: `rever VERSION_NUMBER`.
- Releasing a new version will do the following things in that order:
  - Update [AUTHORS.rst](./AUTHORS.rst).
  - Update [CHANGELOG.rst](./CHANGELOG.rst).
  - Bump the version number in `setup.py` and `_version.py`.
  - Add a git tag.
  - Push the git tag.
  - Add a new release on the GH repo associated with the git tag.
  - Update the appropriate feedstock to build a new conda package.
