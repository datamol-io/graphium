#!/bin/bash

# Function to download dataset
download_dataset() {
    local dataset_url=$1
    local dataset_path=$2

    # Create directory if it does not exist
    mkdir -p $(dirname "${dataset_path}")

    # Download the dataset
    wget -O "${dataset_path}" "${dataset_url}"
}

# L1000_VCAP
download_dataset "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/LINCS_L1000_VCAP_0-4.csv.gz" "graphium/data/neurips2023/large-dataset/LINCS_L1000_VCAP_0-2_th2.csv.gz"
download_dataset "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/l1000_vcap_random_splits.pt" "graphium/data/neurips2023/large-dataset/l1000_vcap_random_splits.pt"

# l1000_MCF7
download_dataset "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/LINCS_L1000_MCF7_0-4.csv.gz" "graphium/data/neurips2023/large-dataset/LINCS_L1000_MCF7_0-2_th2.csv.gz"
download_dataset "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/l1000_mcf7_random_splits.pt" "graphium/data/neurips2023/large-dataset/l1000_mcf7_random_splits.pt"

# PCBA_1328
download_dataset "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/PCBA_1328_1564k.parquet" "graphium/data/neurips2023/large-dataset/PCBA_1328_1564k.parquet"
download_dataset "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/pcba_1328_random_splits.pt" "graphium/data/neurips2023/large-dataset/pcba_1328_random_splits.pt"

# PCQM4M_G25
download_dataset "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/PCQM4M_G25_N4.parquet" "graphium/data/neurips2023/large-dataset/PCQM4M_G25_N4.parquet"
download_dataset "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/pcqm4m_g25_n4_random_splits.pt" "graphium/data/neurips2023/large-dataset/pcqm4m_g25_n4_random_splits.pt"

# PCQM4M_N4
download_dataset "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/PCQM4M_G25_N4.parquet" "graphium/data/neurips2023/large-dataset/PCQM4M_G25_N4.parquet"
download_dataset "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Large-dataset/pcqm4m_g25_n4_random_splits.pt" "graphium/data/neurips2023/large-dataset/pcqm4m_g25_n4_random_splits.pt"
