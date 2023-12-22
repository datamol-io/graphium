"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


"""
Create the pip environment for IPU
"""

## Uncomment this to create the folder for the environment
# mkdir ~/.venv                              # Create the folder for the environment
# python3 -m venv ~/.venv/graphium_ipu           # Create the environment
# source ~/.venv/graphium_ipu/bin/activate       # Activate the environment

# Installing the dependencies for the IPU environment
pip install torch==1.10+cpu torchvision==0.11+cpu torchaudio==0.10 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
pip install /opt/gc/sdk-3.0.0+1128/poptorch-3.0.0+84519_672c9cbc7f_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
pip install -e .
