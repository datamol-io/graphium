// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

// Torch tensor headers
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <torch/extension.h>

// PyBind and Torch headers
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

// The following functions are in labels.cpp, and declared here so that
// graphium_cpp.cpp can expose them to Python via pybind.
std::tuple<
    std::vector<int64_t>,
    std::vector<int32_t>
> load_num_cols_and_dtypes(
    const std::string& processed_graph_data_path,
    const std::string& data_hash);

std::vector<at::Tensor> load_metadata_tensors(
    const std::string processed_graph_data_path,
    const std::string stage,
    const std::string data_hash);

std::vector<at::Tensor> load_stats(
    const std::string processed_graph_data_path,
    const std::string data_hash,
    const std::string task_name);

std::pair<at::Tensor, at::Tensor> concatenate_strings(pybind11::handle handle);

std::tuple<
    std::unordered_map<std::string, std::vector<at::Tensor>>,
    std::unordered_map<std::string, std::vector<at::Tensor>>,
    std::vector<int64_t>,
    std::vector<int32_t>
> prepare_and_save_data(
    const pybind11::list& task_names,
    pybind11::dict& task_dataset_args,
    const pybind11::dict& task_label_normalization,
    const std::string processed_graph_data_path,
    const std::string data_hash,
    const pybind11::dict& task_train_indices,
    const pybind11::dict& task_val_indices,
    const pybind11::dict& task_test_indices,
    bool add_self_loop = false,
    bool explicit_H = false,
    int max_threads = 0,
    bool merge_equivalent_mols = true);

void load_labels_from_index(
    const std::string stage_directory,
    const int64_t mol_index,
    const at::Tensor& mol_file_data_offsets,
    const pybind11::list& label_names,
    const pybind11::list& label_num_cols,
    const pybind11::list& label_data_types,
    pybind11::dict& labels);

std::string extract_string(
    const at::Tensor& concat_strings,
    const at::Tensor& string_offsets,
    const int64_t index);
