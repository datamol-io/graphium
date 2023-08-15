#!/bin/bash

# Constants
venv_name=".graphium_ipu"
sdk_compressed_file="poplar_sdk-ubuntu_20_04-3.3.0-208993bbb7.tar.gz"
sdk_wheel_file="poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl"
sdk_url="https://downloads.graphcore.ai/direct?package=poplar-poplar_sdk_ubuntu_20_04_3.3.0_208993bbb7-3.3.0&file=${sdk_compressed_file}"
sdk_path="${venv_name}/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7"

# Check for Python3 and pip
if ! command -v python3 &>/dev/null; then
    echo "Python3 is required but it's not installed. Exiting."
    exit 1
fi

if ! command -v pip3 &>/dev/null; then
    echo "pip3 is required but it's not installed. Exiting."
    exit 1
fi

# Remove existing venv directory if it exists
if [[ -d $venv_name ]]; then
    echo "Removing existing virtual environment directory..."
    rm -rf $venv_name
fi

# Create the virtual environment
echo "Creating virtual environment..."
mkdir -p $venv_name
python3 -m venv $venv_name
source $venv_name/bin/activate

# Update pip to the latest version
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Download the Poplar SDK
echo "Downloading Poplar SDK..."
wget -q -O "${venv_name}/${sdk_compressed_file}" "$sdk_url"

# Check the wget exit status
if [ $? -ne 0 ]; then
    echo "Failed to download Poplar SDK. Exiting."
    exit 1
fi

# Unzip the SDK file
echo "Extracting Poplar SDK..."
tar -xzf "$venv_name/$sdk_compressed_file" -C $venv_name

# Install the PopTorch wheel
echo "Installing PopTorch..."
python3 -m pip install "${sdk_path}/${sdk_wheel_file}"

# Enable Poplar SDK (including Poplar and PopART)
echo "Enabling Poplar SDK..."
source ${sdk_path}/enable

# Install the IPU specific and Graphium requirements
echo "Installing IPU specific and Graphium requirements..."
python3 -m pip install -r requirements_ipu.txt

# Install Graphium in dev mode
echo "Installing Graphium in dev mode..."
python3 -m pip install --no-deps -e .

echo "Installation completed successfully."

# Download the datafiles (Total ~ 10Mb - nothing compared to the libraries)
echo "Downloading the sub-datasets consisting on the ToyMix dataset"
toymix_dir=expts/data/neurips2023/small-dataset/
mkdir -p $toymix_dir
wget -P "${toymix_dir}" "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/ZINC12k.csv.gz"
wget -P "${toymix_dir}" "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/Tox21-7k-12-labels.csv.gz"
wget -P "${toymix_dir}" "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/qm9.csv.gz"
wget -P "${toymix_dir}" "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/qm9_random_splits.pt"
wget -P "${toymix_dir}" "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/Tox21_random_splits.pt"
wget -P "${toymix_dir}" "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/ZINC12k_random_splits.pt"