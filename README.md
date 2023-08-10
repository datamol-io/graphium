<div align="center">
    <img src="docs/images/logo.png" height="200px">
    <h3>Scaling molecular GNNs to infinity</h3>
</div>

---

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/sdGggS)
[![PyPI](https://img.shields.io/pypi/v/graphium)](https://pypi.org/project/graphium/)
[![Conda](https://img.shields.io/conda/v/conda-forge/graphium?label=conda&color=success)](https://anaconda.org/conda-forge/graphium)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/graphium)](https://pypi.org/project/graphium/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/graphium)](https://anaconda.org/conda-forge/graphium)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/datamol-io/graphium/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/datamol-io/graphium)](https://github.com/datamol-io/graphium/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/datamol-io/graphium)](https://github.com/datamol-io/graphium/network/members)
[![test](https://github.com/datamol-io/graphium/actions/workflows/test.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/test.yml)
[![test-ipu](https://github.com/datamol-io/graphium/actions/workflows/test_ipu.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/test_ipu.yml)
[![release](https://github.com/datamol-io/graphium/actions/workflows/release.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/release.yml)
[![code-check](https://github.com/datamol-io/graphium/actions/workflows/code-check.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/code-check.yml)
[![doc](https://github.com/datamol-io/graphium/actions/workflows/doc.yml/badge.svg)](https://github.com/datamol-io/graphium/actions/workflows/doc.yml)
[![codecov](https://codecov.io/gh/datamol-io/graphium/branch/main/graph/badge.svg?token=bHOkKY5Fze)](https://codecov.io/gh/datamol-io/graphium)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

A deep learning library focused on graph representation learning for real-world chemical tasks.

- ✅ State-of-the-art GNN architectures.
- 🐍 Extensible API: build your own GNN model and train it with ease.
- ⚗️ Rich featurization: powerful and flexible built-in molecular featurization.
- 🧠 Pretrained models: for fast and easy inference or transfer learning.
- ⮔ Read-to-use training loop based on [Pytorch Lightning](https://www.pytorchlightning.ai/).
- 🔌 Have a new dataset? Graphium provides a simple plug-and-play interface. Change the path, the name of the columns to predict, the atomic featurization, and you’re ready to play!

## Documentation

Visit https://graphium-docs.datamol.io/.

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/sdGggS)

You can try running Graphium on Graphcore IPUs for free on Gradient by clicking on the button above.

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
mkdir ~/.venv                               # Create the folder for the environment
python3 -m venv ~/.venv/graphium_ipu        # Create the environment
source ~/.venv/graphium_ipu/bin/activate    # Activate the environment

# Update pip to the latest version
python3 -m pip install --upgrade pip

# Install the PopTorch wheel
# Make sure this is the 3.3 SDK
# Change the link according to your operating system and the `PATH_TO_SDK`
pip install PATH_TO_SDK/poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl

# Enable Poplar SDK (including Poplar and PopART)
source PATH_TO_SDK/enable

# Install the IPU specific and graphium requirements
pip install -r requirements_ipu.txt

# Install Graphium in dev mode
pip install --no-deps -e .
```
If you are new to Graphcore IPUs, you can find more details in the section below: `First Time Running On IPUs`. 

## Training a model

To learn how to train a model, we invite you to look at the documentation, or the jupyter notebooks available [here](https://github.com/datamol-io/graphium/tree/master/docs/tutorials/model_training).

If you are not familiar with [PyTorch](https://pytorch.org/docs) or [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), we highly recommend going through their tutorial first.

## Running an experiment
We have setup Graphium with `hydra` for managing config files. To run an experiment go to the `expts/` folder. For example, to benchmark a GCN on the ToyMix dataset run
```bash
graphium-train dataset=toymix model=gcn
```
To change parameters specific to this experiment like switching from `fp16` to `fp32` precision, you can either override them directly in the CLI via
```bash
graphium-train dataset=toymix model=gcn trainer.trainer.precision=32
```
or change them permamently in the dedicated experiment config under `expts/hydra-configs/toymix_gcn.yaml`.
Integrating `hydra` also allows you to quickly switch between accelerators. E.g., running
```bash
graphium-train dataset=toymix model=gcn accelerator=gpu
```
automatically selects the correct configs to run the experiment on GPU.
Finally, you can also run a fine-tuning loop: 
```bash
graphium-train +finetuning=admet
```

To use a config file you built from scratch you can run
```bash
graphium-train --config-path [PATH] --config-name [CONFIG]
```
Thanks to the modular nature of `hydra` you can reuse many of our config settings for your own experiments with Graphium.


## First Time Running on IPUs
For new IPU developers this section helps provide some more explanation on how to set up an environment to use Graphcore IPUs with Graphium. 

```bash
# Set up a virtual environment as normal
mkdir ~/.venv                               # Create the folder for the environment
python3 -m venv ~/.venv/graphium_ipu        # Create the environment
source ~/.venv/graphium_ipu/bin/activate    # Activate the environment

python3 -m pip install --upgrade pip
# We can download the Poplar SDK directly using `wget` - more details on the various Graphcore downloads can be found here `https://www.graphcore.ai/downloads`

# NOTE: For simplicity this will download the SDK directly where you run this command, we recommend doing this outside the Graphium directory. 
# Make sure to download the right file according to your operating system
wget -q -O 'poplar_sdk-ubuntu_20_04-3.3.0-208993bbb7.tar.gz' 'https://downloads.graphcore.ai/direct?package=poplar-poplar_sdk_ubuntu_20_04_3.3.0_208993bbb7-3.3.0&file=poplar_sdk-ubuntu_20_04-3.3.0-208993bbb7.tar.gz'

# Unzip the SDK file
tar -xzf poplar_sdk-ubuntu_20_04-3.3.0-208993bbb7.tar.gz
# Then use pip to install the wheel 
python3 -m pip install poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
# Enable Poplar SDK (including Poplar and PopART)
source poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/enable 

# Then as a quick test make sure poptorch is correctly installed
# If it is, this will not execute properly. 
python3 -c "import poptorch;print('poptorch installed correctly')"

# Install the IPU specific and graphium requirements
pip install -r requirements_ipu.txt
# Install Graphium in dev mode
python -m pip install --no-deps -e .

```



## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).

## Documentation

- Diagram for data processing in molGPS.

<img src="docs/images/datamodule.png" alt="Data Processing Chart" width="60%" height="60%">

- Diagram for Muti-task network in molGPS

<img src="docs/images/full_graph_network.png" alt="Full Graph Multi-task Network" width="80%" height="80%">
