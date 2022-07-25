################ Preparation #################################################
# mkdir ~/.venv                              # Create the folder for the environment
# python3 -m venv ~/.venv/goli_ipu           # Create the environment
# source ~/.venv/goli_ipu/bin/activate       # Activate the environment
##############################################################################

# Installing the dependencies for the IPU environment
pip install torch==1.10.0 torchaudio==0.10.0
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
pip install /opt/gc/sdk-2.5.1/poptorch-2.5.0+62285_0f4af0bf32_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
pip install -e .