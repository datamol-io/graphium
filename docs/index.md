# Overview

A deep learning library focused on graph representation learning for real-world chemical tasks.

- ✅ State-of-the-art GNN architectures.
- 🐍 Extensible API: build your own GNN model and train it with ease.
- ⚗️ Rich featurization: powerful and flexible built-in molecular featurization.
- 🧠 Pretrained models: for fast and easy inference or transfer learning.
- ⮔ Read-to-use training loop based on [Pytorch Lightning](https://www.pytorchlightning.ai/).
- 🔌 Have a new dataset? Graphium provides a simple plug-and-play interface. Change the path, the name of the columns to predict, the atomic featurization, and you’re ready to play!

## Installation

Use [`mamba`](https://github.com/mamba-org/mamba):

```bash
# Install Graphium
mamba install -c conda-forge graphium
```

or pip:

```bash
pip install graphium
```

### IPU installation

```bash
mkdir ~/.venv                               # Create the folder for the environment
python3 -m venv ~/.venv/graphium_ipu        # Create the environment
source ~/.venv/graphium_ipu/bin/activate    # Activate the environment

# Install the PopTorch wheel
pip install PATH_TO_SDK/poptorch-3.2.0+109946_bb50ce43ab_ubuntu_20_04-cp38-cp38-linux_x86_64.whl

# Enable Poplar SDK (including Poplar and PopART)
source PATH_TO_SDK/enable

# Install the IPU specific and graphium requirements
PACKAGE_NAME=pytorch pip install -r requirements_ipu.txt
pip install -r lightning.txt

# Install Graphium in dev mode
pip install --no-deps -e .
```
