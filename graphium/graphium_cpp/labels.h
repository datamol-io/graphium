// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file This file declares functions for preprocessing and looking up label data,
//!       for exporting to Python, defined in labels.cpp

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

//! Reads the number of columns and data type for each task, from the common label metadata
//! file that was already saved by `prepare_and_save_data`, possibly on a previous run, in the
//! directory `processed_graph_data_path/data_hash`.  Returns empty lists on failure.
//!
//! This is implemented in labels.cpp, and declared here so that graphium_cpp.cpp
//! can expose it to Python via pybind.
std::tuple<
    std::vector<int64_t>,
    std::vector<int32_t>
> load_num_cols_and_dtypes(
    const std::string& processed_graph_data_path,
    const std::string& data_hash);

//! Reads data from the stage-specific label metadata files that were already saved by
//! `prepare_and_save_data`, possibly on a previous run, in the directory
//! `processed_graph_data_path/stage_data_hash`.  Returns an empty list on failure.
//!
//! On success, the returned tensors are:
//!     0) All SMILES strings concatenated,
//!     1) The beginning offsets of each SMILES string in the first tensor, and
//!        one extra at the end equal to the length of the first tensor
//!     2) The number of nodes (atoms) in each molecule
//!     3) The number of edges (bonds) in each molecule
//!     4) (Optional if only inference) The offset of each molecule's label data within the
//!        label data files, plus an extra for the end of each file
//! The first two tensors are used by `extract_string`.  The optional last tensor is
//! used by `load_labels_from_index`.
//!
//! This is implemented in labels.cpp, and declared here so that graphium_cpp.cpp
//! can expose it to Python via pybind.
std::vector<at::Tensor> load_metadata_tensors(
    const std::string processed_graph_data_path,
    const std::string stage,
    const std::string data_hash);

//! Reads data from the task-specific stats file that was already saved by
//! `prepare_and_save_data`, possibly on a previous run, in the directory
//! `processed_graph_data_path/data_hash`.  Returns an empty list on failure.
//!
//! Each tensor's length is the number of columns for this task, and there are 4
//! tensors total: minimum, maximum, mean, standard deviation.
//!
//! This is implemented in labels.cpp, and declared here so that graphium_cpp.cpp
//! can expose it to Python via pybind.
std::vector<at::Tensor> load_stats(
    const std::string processed_graph_data_path,
    const std::string data_hash,
    const std::string task_name);

//! Accepts a Numpy array of strings or Python list of strings, and returns a PyTorch tensor
//! of all of the characters and another tensor containing indices into the other tensor
//! indicating where each string begins, plus one extra index indicating the end.
//!
//! This is implemented in labels.cpp, and declared here so that graphium_cpp.cpp
//! can expose it to Python via pybind.
std::pair<at::Tensor, at::Tensor> concatenate_strings(pybind11::handle handle);

//! Merges label data for equivalent molecules from separate datasets,
//! computes statistics, and caches the label data to files for efficient loading later.
//!
//! This is implemented in labels.cpp, and declared here so that graphium_cpp.cpp
//! can expose it to Python via pybind.
//!
//! @param task_names Python list of the names of the datasets to process.  These are used for
//!                   looking up into the other parameters starting with `task_`, and the
//!                   beginning of each name must be `graph_`, `node_`, `edge_`, or `nodepair_`
//!                   to determine the level of the label data.
//! @param task_dataset_args Python dict mapping task names to Python dicts for each dataset.
//!                          Each task's dict must contain a mapping from `"smiles"` to a 1D
//!                          Numpy array of objects, each of which is a Python string with a
//!                          molecule's SMILES text.  If doing inference, each task's dict must
//!                          also map from `"labels"` to a 2D Numpy array of float16, float32,
//!                          or float64 type.  For node, edge, or node-pair level label data,
//!                          the dict must also map from `"label_offsets"` to a 1D Numpy array
//!                          of type int64, indicating the row in the `"labels"` array where
//!                          each molecule's data begins, plus an extra for the end.  If
//!                          `"label_offsets"` is not present, the `"labels"` array has one row
//!                          per molecule, and if it is present, the `"labels"` array has one
//!                          row per atom, bond, or pair of atoms, according to the label level.
//! @param task_label_normalization Python dict mapping task names to Python dicts for each
//!                                 dataset's normalization options.  Each task's dict must
//!                                 contain a mapping from `"method"` to either `"none"`,
//!                                 `"normal"`, or `"unit"`, and can optionally contain a
//!                                 mapping from `"min_clipping"` and/or `"max_clipping"` to a
//!                                 Python float or int to explicitly clip the range.
//! @param processed_graph_data_path String containing the base directory to create
//!                                  subdirectories for cached files in.  It can exist already,
//!                                  or will be created if it does not already exist.
//! @param data_hash String representing a hash of the label data options.  It will be used in
//!                  the names of all subdirectories created under `processed_graph_data_path`.
//! @param task_train_indices Python dict mapping task names to Python lists of ints, indicating
//!                           indices into `task_dataset_args[task_name]["smiles"]` and other
//!                           per-molecule-per-task arrays.  Only these molecules will be used
//!                           for the "train" stage.
//! @param task_val_indices Python dict mapping task names to Python lists of ints, indicating
//!                         indices into `task_dataset_args[task_name]["smiles"]` and other
//!                         per-molecule-per-task arrays.  Only these molecules will be used
//!                         for the "val" stage.
//! @param task_test_indices Python dict mapping task names to Python lists of ints, indicating
//!                          indices into `task_dataset_args[task_name]["smiles"]` and other
//!                          per-molecule-per-task arrays.  Only these molecules will be used
//!                          for the "test" stage.
//! @param add_self_loop If true (default is false), `num_atoms` is added to the number of
//!                      directed edges (twice the number of bonds).  This is for consistency
//!                      with `featurize_smiles` later.
//! @param explicit_H If true (default is false), any implicit hydrogens will be made explicit,
//!                   possibly increasing the number of atoms.
//! @param max_threads If greater than zero, at most this many threads will be created for
//!                    processing in parallel.  If zero (the default), at most one thread per
//!                    logical CPU core will be created.  If less than zero, the limit is
//!                    reduced by adding this negative amount to the number of logical CPU
//!                    cores.
//! @param merge_equivalent_mols If true (the default), label data for the same molecule in
//!                              different datasets are collected together, even if the atoms
//!                              or bonds are in a different order.  Duplicates of the same
//!                              molecule within a single dataset will be ignored.  This is very
//!                              slow, and changes the number and order of the molecules, so it
//!                              can be set to false for inference, where there is no label data
//!                              or only one dataset.
//! @return Four objects:
//!         - A dict mapping the stage names ("train", "val", "test") to a list of five 1D
//!           PyTorch tensors:
//!             0) SMILES strings all concatenated, one per unique molecule
//!             1) Offsets into the previous tensor where the strings begin, one per unique
//!                molecule, plus one extra for the end
//!             2) Number of nodes (atoms) in each unique molecule
//!             3) Number of edges (2*bonds) in each unique molecule
//!             4) (Only if there is label data) `mol_file_data_offsets` to be passed to calls
//!                to `load_labels_from_index`
//!         - A dict mapping task names to a list of four 1D PyTorch tensors for column
//!           normalization: minimum, maximum, mean, standard deviation
//!         - A list of the number of columns in each task, in the same order as `task_names`
//!         - A list of integers representing the Torch data type of each task, in the same
//!           order as `task_names`
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

//! Loads label data associated with the molecule with index `mol_index` from the corresponding
//! file in the directory `stage_directory`, and adds the data to `labels` dictionary using
//! the strings from `label_names` to map to tensors.  The label data must be previously saved
//! by `prepare_and_save_data`.  `mol_file_data_offsets` is used to determine how to find the
//! data in the file, `label_data_types` is used for the type and size of each float, and
//! `label_num_cols` is used to determine the layout of each output tensor, especially ones with
//! multiple rows, such as node-level, edge-level, or node-pair-level label data.
//!
//! This is implemented in labels.cpp, and declared here so that graphium_cpp.cpp
//! can expose it to Python via pybind.
void load_labels_from_index(
    const std::string stage_directory,
    const int64_t mol_index,
    const at::Tensor& mol_file_data_offsets,
    const pybind11::list& label_names,
    const pybind11::list& label_num_cols,
    const pybind11::list& label_data_types,
    pybind11::dict& labels);

//! Extracts a single string from `concat_strings`, a Tensor of contatenated strings,
//! using offsets at the specified `index` in `string_offsets`.
//!
//! The tensors can be returned by `load_metadata_tensors`, `concatenate_strings`, or
//! `prepare_and_save_data`.
//!
//! This is implemented in labels.cpp, and declared here so that graphium_cpp.cpp
//! can expose it to Python via pybind.
std::string extract_string(
    const at::Tensor& concat_strings,
    const at::Tensor& string_offsets,
    const int64_t index);
