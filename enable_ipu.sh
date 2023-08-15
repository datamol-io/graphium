#!/bin/bash

# Constants
venv_name=".graphium_ipu"
sdk_path="${venv_name}/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7"

# Source the virtual environment
source ${venv_name}/bin/activate
source ${sdk_path}/enable