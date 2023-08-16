#!/bin/bash

# Default location for the virtual environment
default_venv_name=".graphium_ipu"

# Allow the user to specify the location of their virtual environment
# If not specified, use the default location
venv_name=${1:-$default_venv_name}

# Constants
sdk_path="${venv_name}/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7"

# Source the virtual environment
source ${venv_name}/bin/activate
source ${sdk_path}/enable