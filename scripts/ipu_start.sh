"""
Start the ipu environment and SDK
"""

source /opt/gc/sdk-3.0.0+1128/poplar-ubuntu_20_04-3.0.0+5468-0379b9a65d/enable.sh
source /opt/gc/sdk-3.0.0+1128/popart-ubuntu_20_04-3.0.0+5468-0379b9a65d/enable.sh

source ~/.venv/graphium_ipu/bin/activate # Change to your path

export VISUAL=vim
export EDITOR="$VISUAL"
