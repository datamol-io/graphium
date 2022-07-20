pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt 
pip install -e .