// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "labels.h"

#include "features.h"

// C++ standard library headers
#include <filesystem>
#include <thread>
#include <unordered_map>

// RDKit headers
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/inchi.h>

// Numpy array headers
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#ifdef _WIN32
// Windows file handling wrappers
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

using FileType = HANDLE;
const auto INVALID_FILE = INVALID_HANDLE_VALUE;

static FileType fopen_read_wrapper(const std::filesystem::path& file_path) {
    return CreateFileW(
        file_path.wstring().c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
}

static FileType fopen_write_wrapper(const std::filesystem::path& file_path) {
    return CreateFileW(
        file_path.wstring().c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
}

static size_t fread_wrapper(void* buffer, size_t bytes, FileType file) {
    size_t total_bytes_read = 0;
    while (bytes > 0) {
        // NOTE: ReadFile should support reads up to (2^32 - 1) bytes,
        // but might as well limit it to 1GB (2^30 bytes) at a time,
        // just in case there are issues at or above 2GB.
        const DWORD max_read_size = 1024 * 1024 * 1024;
        const DWORD bytes_to_read = (bytes > max_read_size) ? max_read_size : (DWORD)bytes;
        DWORD bytes_read;
        BOOL success = ReadFile(file, buffer, bytes_to_read, &bytes_read, nullptr);
        total_bytes_read += (success ? bytes_read : 0);
        if (!success || bytes_read != bytes_to_read) {
            return total_bytes_read;
        }
        bytes -= bytes_read;
    }
    return total_bytes_read;
}

static size_t fwrite_wrapper(const void* buffer, size_t bytes, FileType file) {
    size_t total_bytes_written = 0;
    while (bytes > 0) {
        // NOTE: ReadFile should support reads up to (2^32 - 1) bytes,
        // but might as well limit it to 1GB (2^30 bytes) at a time,
        // just in case there are issues at or above 2GB.
        const DWORD max_write_size = 1024 * 1024 * 1024;
        const DWORD bytes_to_write = (bytes > max_write_size) ? max_write_size : (DWORD)bytes;
        DWORD bytes_written;
        BOOL success = WriteFile(file, buffer, bytes_to_write, &bytes_written, nullptr);
        total_bytes_written += (success ? bytes_written : 0);
        if (!success || bytes_written != bytes_to_write) {
            return total_bytes_written;
        }
        bytes -= bytes_written;
    }
    return total_bytes_written;
}

static int fseek_wrapper(FileType file, int64_t file_pointer) {
    LARGE_INTEGER file_pointer_union;
    file_pointer_union.QuadPart = (LONGLONG)file_pointer;
    BOOL success = SetFilePointerEx(file, file_pointer_union, nullptr, FILE_BEGIN);
    return (success == 0);
}

static void fclose_wrapper(FileType file) {
    CloseHandle(file);
}

#else
// Linux file handling wrappers
#include <stdio.h>

using FileType = FILE*;
const auto INVALID_FILE = (FILE*)nullptr;

static FileType fopen_read_wrapper(const std::filesystem::path& file_path) {
    return fopen(file_path.string().c_str(), "rb");
}

static FileType fopen_write_wrapper(const std::filesystem::path& file_path) {
    return fopen(file_path.string().c_str(), "wb");
}

static size_t fread_wrapper(void* buffer, size_t bytes, FileType file) {
    return fread(buffer, 1, bytes, file);
}

static size_t fwrite_wrapper(const void* buffer, size_t bytes, FileType file) {
    return fwrite(buffer, 1, bytes, file);
}

static int fseek_wrapper(FileType file, int64_t file_pointer) {
    // NOTE: If these files could ever be larger than 2GB each, fseek won't
    // work on platforms where "long" is a 32-bit type (e.g. 32-bit Linux)
    return fseek(file, (long)file_pointer, SEEK_SET);
}

static void fclose_wrapper(FileType file) {
    fclose(file);
}

#endif // End of file handling wrappers

struct InitNumpyArrayModule {
    InitNumpyArrayModule() {
        // This imports the numpy array module, and it must be
        // called exactly once before numpy array functions are used.
        if (_import_array() < 0) {
            printf("ERROR: Failed to import numpy.core.multiarray from C++ in graphium_cpp module\n");
        }
    }
};
static void ensure_numpy_array_module_initialized() {
    // Function scope static variables will be initialized upon the first call,
    // and only once, in a threadsafe manner.
    static InitNumpyArrayModule numpy_initializer;
}

struct MolBriefData {
    uint64_t unique_id[2];
    uint32_t num_nodes;
    uint32_t num_edges;
};

static MolBriefData smiles_to_brief_data(
    const std::string& smiles_string,
    bool add_self_loop,
    bool explicit_H,
    bool compute_inchi_key) {

    // Don't add explicit_H here, in case it affects MolToInchiKey (though it really shouldn't)
    std::unique_ptr<RDKit::RWMol> mol{ parse_mol(smiles_string, false) };
    if (!mol) {
        return MolBriefData{ {0,0}, 0, 0 };
    }

    uint64_t id0 = 0;
    uint64_t id1 = 0;
    if (compute_inchi_key) {
        const std::string inchiKeyString = MolToInchiKey(*mol, "/FixedH /SUU /RecMet /KET /15T");
        size_t n = inchiKeyString.size();
        // Format: AAAAAAAAAAAAAA-BBBBBBBBFV-P
        // According to https://www.inchi-trust.org/technical-faq/
        assert(n == 27 && inchiKeyString[14] == '-' && inchiKeyString[25] == '-');
        // Convert from capital letter characters to 64-bit integers:
        // 13 characters for first integer, 12 characters for 2nd integer.
        // Neither should overflow a 64-bit unsigned integer.
        id0 = (n > 0) ? (inchiKeyString[0] - 'A') : 0;
        for (size_t i = 1; i < 13 && i < n; ++i) {
            id0 = 26*id0 + (inchiKeyString[i] - 'A');
        }
        id1 = (13 < n) ? (inchiKeyString[13] - 'A') : 0;
        for (size_t i = 15; i < 25 && i < n; ++i) {
            id1 = 26*id1 + (inchiKeyString[i] - 'A');
        }
        if (26 < n) {
            id1 = 26*id1 + (inchiKeyString[26] - 'A');
        }
    }

    // Now handle explicit_H
    if (explicit_H) {
        RDKit::MolOps::addHs(*mol);
    }
    else {
        // Default params for SmilesToMol already calls removeHs,
        // and calling it again shouldn't have any net effect.
        //RDKit::MolOps::removeHs(*mol);
    }

    return MolBriefData{
        {id0, id1},
        mol->getNumAtoms(),
        2*mol->getNumBonds() + (add_self_loop ? mol->getNumAtoms() : 0)
    };
}

enum class NormalizationMethod {
    NONE,
    NORMAL,
    UNIT
};
struct NormalizationOptions {
    NormalizationMethod method = NormalizationMethod::NONE;
    double min_clipping = -std::numeric_limits<double>::infinity();
    double max_clipping = std::numeric_limits<double>::infinity();
};

constexpr size_t num_mols_per_file = 1024;

static void get_mol_label_filename(
    char filename[25],
    uint64_t file_num) {

    size_t filename_index = 0;
    while (file_num != 0) {
        filename[filename_index] = '0' + (file_num % 10);
        ++filename_index;
        file_num /= 10;
    }
    while (filename_index < 7) {
        filename[filename_index] = '0';
        ++filename_index;
    }
    std::reverse(filename, filename + filename_index);
    filename[filename_index] = '.';
    filename[filename_index+1] = 't';
    filename[filename_index+2] = 'm';
    filename[filename_index+3] = 'p';
    filename[filename_index+4] = 0;
}

struct Types {
    size_t size;
    int numpy_type;
    c10::ScalarType torch_type;
};
constexpr size_t num_supported_types = 3;
constexpr Types supported_types[num_supported_types] = {
    {2, NPY_FLOAT16, c10::ScalarType::Half},
    {4, NPY_FLOAT32, c10::ScalarType::Float},
    {8, NPY_FLOAT64, c10::ScalarType::Double}
};
static bool is_supported_numpy_type(int type) {
    return (type == supported_types[0].numpy_type) ||
        (type == supported_types[1].numpy_type) ||
        (type == supported_types[2].numpy_type);
};
static size_t numpy_type_index(int type) {
    if (type == supported_types[0].numpy_type) {
        return 0;
    }
    if (type == supported_types[1].numpy_type) {
        return 1;
    }
    if (type == supported_types[2].numpy_type) {
        return 2;
    }
    return num_supported_types;
};
static size_t torch_type_index(c10::ScalarType type) {
    if (type == supported_types[0].torch_type) {
        return 0;
    }
    if (type == supported_types[1].torch_type) {
        return 1;
    }
    if (type == supported_types[2].torch_type) {
        return 2;
    }
    return num_supported_types;
};


constexpr const char*const label_metadata_filename = "label_metadata.tmp";
constexpr const char*const file_data_offsets_filename = "file_data_offsets.tmp";
constexpr const char*const concat_smiles_filename = "concat_smiles.tmp";
constexpr const char*const smiles_offsets_filename = "smiles_offsets.tmp";
constexpr const char*const num_nodes_filename = "num_nodes.tmp";
constexpr const char*const num_edges_filename = "num_edges.tmp";

static bool save_num_cols_and_dtypes(
    const std::filesystem::path& common_path,
    const std::vector<int64_t>& label_num_cols,
    const std::vector<int32_t>& label_data_types) {

    const uint64_t num_labels = label_num_cols.size();
    if (num_labels != label_data_types.size()) {
        return false;
    }
    std::filesystem::path file_path(common_path / label_metadata_filename);
    FileType file = fopen_write_wrapper(file_path);
    if (file == INVALID_FILE) {
        return false;
    }
    size_t num_bytes_written = fwrite_wrapper(&num_labels, sizeof(num_labels), file);
    num_bytes_written += fwrite_wrapper(label_num_cols.data(), sizeof(label_num_cols[0])*num_labels, file);
    num_bytes_written += fwrite_wrapper(label_data_types.data(), sizeof(label_data_types[0])*num_labels, file);
    fclose_wrapper(file);
    if (num_bytes_written != sizeof(num_labels) + (sizeof(label_num_cols[0]) + sizeof(label_data_types[0]))*num_labels) {
        return false;
    }
    return true;
}

std::tuple<
    std::vector<int64_t>,
    std::vector<int32_t>
> load_num_cols_and_dtypes(
    const std::string& processed_graph_data_path,
    const std::string& data_hash) {

    std::vector<int64_t> label_num_cols;
    std::vector<int32_t> label_data_types;
    std::filesystem::path file_path(
        std::filesystem::path(processed_graph_data_path) / data_hash / label_metadata_filename
    );
    FileType file = fopen_read_wrapper(file_path);
    if (file == INVALID_FILE) {
        return std::make_tuple(std::move(label_num_cols), std::move(label_data_types));
    }
    uint64_t num_labels = 0;
    size_t num_bytes_read = fread_wrapper(&num_labels, sizeof(num_labels), file);
    // Trying to allocate 2^60 would fail, unless it overflows and then crashes
    if (num_bytes_read != sizeof(num_labels) || num_labels == 0 || num_labels >= (uint64_t(1) << (64-4))) {
        fclose_wrapper(file);
        return std::make_tuple(std::move(label_num_cols), std::move(label_data_types));
    }
    label_num_cols.resize(num_labels, 0);
    num_bytes_read = fread_wrapper(label_num_cols.data(), sizeof(label_num_cols[0])*num_labels, file);
    if (num_bytes_read != sizeof(label_num_cols[0])*num_labels) {
        fclose_wrapper(file);
        label_num_cols.resize(0);
        return std::make_tuple(std::move(label_num_cols), std::move(label_data_types));
    }
    label_data_types.resize(num_labels, -1);
    num_bytes_read = fread_wrapper(label_data_types.data(), sizeof(label_data_types[0])*num_labels, file);
    fclose_wrapper(file);
    if (num_bytes_read != sizeof(label_data_types[0])*num_labels) {
        label_num_cols.resize(0);
        label_data_types.resize(0);
    }
    return std::make_tuple(std::move(label_num_cols), std::move(label_data_types));
}

template<typename T>
bool save_array_to_file(
    const std::filesystem::path& directory,
    const char*const filename,
    const T* data,
    const uint64_t n) {

    std::filesystem::path file_path(directory / filename);
    FileType file = fopen_write_wrapper(file_path);
    if (file == INVALID_FILE) {
        return false;
    }
    size_t num_bytes_written = fwrite_wrapper(&n, sizeof(n), file);
    num_bytes_written += fwrite_wrapper(data, sizeof(T)*n, file);
    fclose_wrapper(file);
    if (num_bytes_written != sizeof(n) + sizeof(T)*n) {
        return false;
    }
    return true;
}


template<typename T>
[[nodiscard]] uint64_t load_array_from_file(
    const std::filesystem::path& directory,
    const char*const filename,
    std::unique_ptr<T[]>& data) {

    data.reset(nullptr);

    std::filesystem::path file_path(directory / filename);
    FileType file = fopen_read_wrapper(file_path);
    if (file == INVALID_FILE) {
        return 0;
    }
    uint64_t n;
    size_t num_bytes_read = fread_wrapper(&n, sizeof(n), file);
    // Trying to allocate 2^60 would fail, unless it overflows and then crashes
    if (num_bytes_read != sizeof(n) || n == 0 || n >= (uint64_t(1) << (64-4))) {
        fclose_wrapper(file);
        return 0;
    }
    data.reset(new T[n]);
    num_bytes_read = fread_wrapper(data.get(), sizeof(T)*n, file);
    fclose_wrapper(file);
    if (num_bytes_read != sizeof(T)*n) {
        data.reset(nullptr);
        return 0;
    }
    return n;
}

std::vector<at::Tensor> load_metadata_tensors(
    const std::string processed_graph_data_path,
    const std::string stage,
    const std::string data_hash) {

    std::filesystem::path base_path{processed_graph_data_path};
    std::filesystem::path directory = base_path / (stage + "_" + data_hash);

    std::unique_ptr<char[]> concatenated_smiles;
    uint64_t concatenated_smiles_size =
        load_array_from_file(directory, concat_smiles_filename, concatenated_smiles);

    std::unique_ptr<int64_t[]> smiles_offsets;
    uint64_t num_smiles_offsets =
        load_array_from_file(directory, smiles_offsets_filename, smiles_offsets);

    std::unique_ptr<int32_t[]> num_nodes;
    uint64_t num_num_nodes =
        load_array_from_file(directory, num_nodes_filename, num_nodes);

    std::unique_ptr<int32_t[]> num_edges;
    uint64_t num_num_edges =
        load_array_from_file(directory, num_edges_filename, num_edges);

    std::unique_ptr<int64_t[]> mol_data_offsets;
    uint64_t num_mol_data_offsets =
        load_array_from_file(directory, file_data_offsets_filename, mol_data_offsets);

    if (num_num_nodes == 0 || num_num_edges != num_num_nodes || num_smiles_offsets != (num_num_nodes+1) ||
            concatenated_smiles_size == 0 || concatenated_smiles_size != uint64_t(smiles_offsets[num_num_edges]) ||
            (num_mol_data_offsets != num_num_nodes + (num_num_nodes + num_mols_per_file-1)/num_mols_per_file && num_mol_data_offsets != 0)) {
        printf("ERROR: graphium_cpp.load_metadata_tensors failed to load valid metadata files\n");
        printf("    len(concat_smiles) is %zu\n", size_t(concatenated_smiles_size));
        printf("    len(smiles_offsets) is %zu\n", size_t(num_smiles_offsets));
        printf("    len(num_nodes) is %zu\n", size_t(num_num_nodes));
        printf("    len(num_edges) is %zu\n", size_t(num_num_edges));
        printf("    len(file_data_offsets) is %zu\n", size_t(num_mol_data_offsets));
        return std::vector<at::Tensor>();
    }

    // The above conditions should ensure that none of these arrays are empty,
    // but assert in debug builds just in case.
    assert(concatenated_smiles && smiles_offsets && num_nodes && num_edges);
    
    const int64_t concatenated_smiles_dims[1] = { int64_t(concatenated_smiles_size) };
    at::Tensor smiles_tensor = torch_tensor_from_array(std::move(concatenated_smiles), concatenated_smiles_dims, 1, c10::ScalarType::Char);
    const int64_t smiles_offsets_dims[1] = { int64_t(num_num_nodes+1) };
    at::Tensor smiles_offsets_tensor = torch_tensor_from_array(std::move(smiles_offsets), smiles_offsets_dims, 1, c10::ScalarType::Long);
    const int64_t num_nodes_dims[1] = { int64_t(num_num_nodes) };
    at::Tensor num_nodes_tensor = torch_tensor_from_array(std::move(num_nodes), num_nodes_dims, 1, c10::ScalarType::Int);
    const int64_t num_edges_dims[1] = { int64_t(num_num_nodes) };
    at::Tensor num_edges_tensor = torch_tensor_from_array(std::move(num_edges), num_edges_dims, 1, c10::ScalarType::Int);

    std::vector<at::Tensor> stage_return_data;
    stage_return_data.reserve((num_mol_data_offsets > 0) ? 5 : 4);
    
    stage_return_data.push_back(std::move(smiles_tensor));
    stage_return_data.push_back(std::move(smiles_offsets_tensor));
    stage_return_data.push_back(std::move(num_nodes_tensor));
    stage_return_data.push_back(std::move(num_edges_tensor));

    if (num_mol_data_offsets > 0) {
        const int64_t data_offsets_dims[1] = { int64_t(num_mol_data_offsets) };
        at::Tensor data_offsets_tensor = torch_tensor_from_array(std::move(mol_data_offsets), data_offsets_dims, 1, c10::ScalarType::Long);

        stage_return_data.push_back(std::move(data_offsets_tensor));
    }

    return stage_return_data;
}

std::vector<at::Tensor> load_stats(
    const std::string processed_graph_data_path,
    const std::string data_hash,
    const std::string task_name) {

    std::filesystem::path base_path{processed_graph_data_path};
    std::filesystem::path directory = base_path / data_hash;
    const std::string filename(task_name + "_stats.tmp");

    std::unique_ptr<double[]> task_stats;
    uint64_t num_stat_floats =
        load_array_from_file(directory, filename.c_str(), task_stats);

    if (num_stat_floats == 0 || num_stat_floats % 4 != 0) {
        return std::vector<at::Tensor>();
    }
    
    const uint64_t num_cols = num_stat_floats / 4;
    std::vector<at::Tensor> return_stats(4);
    for (size_t stat_index = 0; stat_index < 4; ++stat_index) {
        std::unique_ptr<double[]> single_stat(new double[num_cols]);
        for (size_t i = 0; i < num_cols; ++i) {
            single_stat[i] = task_stats[4*i + stat_index];
        }
        const int64_t stat_dims[1] = { int64_t(num_cols) };
        at::Tensor stat_tensor = torch_tensor_from_array(std::move(single_stat), stat_dims, 1, c10::ScalarType::Double);
        return_stats.push_back(std::move(stat_tensor));
    }
    
    return return_stats;
}

std::pair<at::Tensor, at::Tensor> concatenate_strings(pybind11::handle handle) {
    using return_type = std::pair<at::Tensor, at::Tensor>;
    
    ensure_numpy_array_module_initialized();
    
    at::Tensor concatenated_strings;
    at::Tensor offsets;

    PyObject* obj_ptr = handle.ptr();
    if (PyArray_Check(obj_ptr)) {
        PyArrayObject* numpy_array = reinterpret_cast<PyArrayObject*>(obj_ptr);
        int type_num = PyArray_TYPE(numpy_array);
        int ndims = PyArray_NDIM(numpy_array);
        if (type_num != NPY_OBJECT || ndims != 1) {
            return return_type(std::move(concatenated_strings), std::move(offsets));
        }
        intptr_t n = PyArray_DIM(numpy_array, 0);
        if (n <= 0) {
            return return_type(std::move(concatenated_strings), std::move(offsets));
        }
        
        size_t total_characters = 0;
        for (intptr_t i = 0; i < n; ++i) {
            pybind11::handle string_handle(*(PyObject**)PyArray_GETPTR1(numpy_array, i));
            if (!pybind11::isinstance<pybind11::str>(string_handle)) {
                continue;
            }
            // TODO: Consider trying to avoid constructing std::string here
            std::string string{pybind11::str{string_handle}};
            // +1 is for null terminator
            total_characters += string.size() + 1;
        }
        std::unique_ptr<char[]> concatenated_chars(new char[total_characters]);
        std::unique_ptr<int64_t[]> offsets_buffer(new int64_t[n+1]);
        int64_t offset = 0;
        for (intptr_t i = 0; i < n; ++i) {
            offsets_buffer[i] = offset;
            pybind11::handle string_handle(*(PyObject**)PyArray_GETPTR1(numpy_array, i));
            if (!pybind11::isinstance<pybind11::str>(string_handle)) {
                continue;
            }
            // TODO: Consider trying to avoid constructing std::string here
            std::string string{pybind11::str{string_handle}};
            memcpy(concatenated_chars.get(), string.c_str(), string.size());
            offset += string.size();
            concatenated_chars[offset] = 0;
            ++offset;
        }
        offsets_buffer[n] = offset;

        const int64_t concatenated_strings_dims[1] = { int64_t(total_characters) };
        concatenated_strings = torch_tensor_from_array(std::move(concatenated_chars), concatenated_strings_dims, 1, c10::ScalarType::Char);
        const int64_t offsets_dims[1] = { int64_t(n+1) };
        offsets = torch_tensor_from_array(std::move(offsets_buffer), offsets_dims, 1, c10::ScalarType::Long);
    }
    if (pybind11::isinstance<pybind11::list>(handle)) {
        pybind11::list list = handle.cast<pybind11::list>();
        size_t n = list.size();
        
        size_t total_characters = 0;
        for (size_t i = 0; i < n; ++i) {
            pybind11::handle string_handle(list[i]);
            if (!pybind11::isinstance<pybind11::str>(string_handle)) {
                continue;
            }
            // TODO: Consider trying to avoid constructing std::string here
            std::string string{pybind11::str{string_handle}};
            // +1 is for null terminator
            total_characters += string.size() + 1;
        }
        std::unique_ptr<char[]> concatenated_chars(new char[total_characters]);
        std::unique_ptr<int64_t[]> offsets_buffer(new int64_t[n+1]);
        int64_t offset = 0;
        for (size_t i = 0; i < n; ++i) {
            offsets_buffer[i] = offset;
            pybind11::handle string_handle(list[i]);
            if (!pybind11::isinstance<pybind11::str>(string_handle)) {
                continue;
            }
            // TODO: Consider trying to avoid constructing std::string here
            std::string string{pybind11::str{string_handle}};
            memcpy(concatenated_chars.get(), string.c_str(), string.size());
            offset += string.size();
            concatenated_chars[offset] = 0;
            ++offset;
        }
        offsets_buffer[n] = offset;

        const int64_t concatenated_strings_dims[1] = { int64_t(total_characters) };
        concatenated_strings = torch_tensor_from_array(std::move(concatenated_chars), concatenated_strings_dims, 1, c10::ScalarType::Char);
        const int64_t offsets_dims[1] = { int64_t(n+1) };
        offsets = torch_tensor_from_array(std::move(offsets_buffer), offsets_dims, 1, c10::ScalarType::Long);
    }
    return return_type(std::move(concatenated_strings), std::move(offsets));
}

constexpr size_t num_stages = 3;
// NOTE: Computing stats below depends on that "train" is stage 0.
const std::string stages[num_stages] = {
    std::string("train"),
    std::string("val"),
    std::string("test")
};


static void get_task_data(
    const pybind11::list& task_names,
    pybind11::dict& task_dataset_args,
    const pybind11::dict& task_label_normalization,
    int64_t* return_label_num_cols,
    int32_t* return_label_data_types,
    size_t* task_col_starts,
    size_t* task_bytes_per_float,
    NormalizationOptions* task_normalization_options,
    PyArrayObject** smiles_numpy_arrays,
    PyArrayObject** labels_numpy_arrays,
    PyArrayObject** label_offsets_numpy_arrays,
    FeatureLevel* task_levels
) {
    size_t total_num_cols = 0;
    size_t task_index = 0;
    for (const auto& task : task_names) {
        const size_t current_task_index = task_index;
        task_col_starts[current_task_index] = total_num_cols;
        task_bytes_per_float[current_task_index] = 0;
        smiles_numpy_arrays[current_task_index] = nullptr;
        labels_numpy_arrays[current_task_index] = nullptr;
        label_offsets_numpy_arrays[current_task_index] = nullptr;
        ++task_index;
        if (!pybind11::isinstance<pybind11::str>(task)) {
            continue;
        }
        const std::string task_name{ pybind11::str(task) };
        pybind11::handle task_dataset_handle = pybind11::handle(PyDict_GetItemString(task_dataset_args.ptr(), task_name.c_str()));
        if (!task_dataset_handle || !pybind11::isinstance<pybind11::dict>(task_dataset_handle)) {
            continue;
        }
        pybind11::dict dataset_dict = task_dataset_handle.cast<pybind11::dict>();
        pybind11::handle smiles_handle = pybind11::handle(PyDict_GetItemString(dataset_dict.ptr(), "smiles"));
        if (!smiles_handle) {
            continue;
        }
        PyObject* smiles_obj_ptr = smiles_handle.ptr();
        if (!PyArray_Check(smiles_obj_ptr)) {
            continue;
        }
        PyArrayObject* smiles_numpy_array = reinterpret_cast<PyArrayObject*>(smiles_obj_ptr);
        int smiles_type_num = PyArray_TYPE(smiles_numpy_array);
        int smiles_ndims = PyArray_NDIM(smiles_numpy_array);
        if (smiles_type_num != NPY_OBJECT || smiles_ndims != 1) {
            continue;
        }
        intptr_t num_smiles = PyArray_DIM(smiles_numpy_array, 0);
        if (num_smiles <= 0) {
            continue;
        }

        // smiles array is okay
        smiles_numpy_arrays[current_task_index] = smiles_numpy_array;

        // Check for labels.  There might not be labels in inference case.
        pybind11::handle labels_handle = pybind11::handle(PyDict_GetItemString(dataset_dict.ptr(), "labels"));
        if (!labels_handle) {
            continue;
        }
        pybind11::handle label_offsets_handle = pybind11::handle(PyDict_GetItemString(dataset_dict.ptr(), "label_offsets"));
        PyObject* labels_obj_ptr = labels_handle.ptr();
        PyObject* label_offsets_obj_ptr = label_offsets_handle.ptr();
        const bool is_labels_numpy = PyArray_Check(labels_obj_ptr);
        const bool is_labels_multi_row = label_offsets_obj_ptr && PyArray_Check(label_offsets_obj_ptr);
        if (!is_labels_numpy) {
            continue;
        }
        PyArrayObject* labels_numpy_array = reinterpret_cast<PyArrayObject*>(labels_obj_ptr);
        PyArrayObject* label_offsets_numpy_array = is_labels_multi_row ? reinterpret_cast<PyArrayObject*>(label_offsets_obj_ptr) : nullptr;
        int labels_type_num = PyArray_TYPE(labels_numpy_array);
        int labels_ndims = PyArray_NDIM(labels_numpy_array);
#if GRAPHIUM_CPP_DEBUGGING
        printf("\"%s\" labels numpy type %d, %d dims\n", task_name.c_str(), labels_type_num, labels_ndims);
#endif
        if (!is_supported_numpy_type(labels_type_num) || labels_ndims != 2) {
            continue;
        }
        if (is_labels_multi_row) {
            int label_offsets_type_num = PyArray_TYPE(label_offsets_numpy_array);
            int label_offsets_ndims = PyArray_NDIM(label_offsets_numpy_array);
            // Only int64 is supported, for simplicity
            if (label_offsets_type_num != NPY_INT64 || label_offsets_ndims != 1) {
                continue;
            }
        }
        intptr_t num_label_rows = PyArray_DIM(labels_numpy_array, 0);
        intptr_t num_molecules = num_label_rows;
        if (is_labels_multi_row) {
            intptr_t num_offsets_rows = PyArray_DIM(label_offsets_numpy_array, 0);
            if (num_offsets_rows == 0) {
                continue;
            }
            // -1 is because last offset is the end offset
            num_molecules = num_offsets_rows - 1;

            // Verify that the first offset is zero
            if (*(const int64_t*)PyArray_GETPTR1(label_offsets_numpy_array, 0) != 0) {
                continue;
            }
            // Verify that the last offset is the end offset
            if (*(const int64_t*)PyArray_GETPTR1(label_offsets_numpy_array, num_molecules) != num_label_rows) {
                continue;
            }
        }
        intptr_t num_label_cols = PyArray_DIM(labels_numpy_array, 1);
#if GRAPHIUM_CPP_DEBUGGING
        printf("\"%s\" labels[%zd][%zd] (%zd molecules)\n", task_name.c_str(), num_label_rows, num_label_cols, num_molecules);
#endif
        if (num_smiles != num_molecules || num_label_cols <= 0) {
            continue;
        }

        const size_t supported_type_index = numpy_type_index(labels_type_num);
        const size_t bytes_per_float = supported_types[supported_type_index].size;
        labels_numpy_arrays[current_task_index] = labels_numpy_array;
        label_offsets_numpy_arrays[current_task_index] = is_labels_multi_row ? label_offsets_numpy_array : nullptr;
        return_label_num_cols[current_task_index] = num_label_cols;
        return_label_data_types[current_task_index] = int(supported_types[supported_type_index].torch_type);
        total_num_cols += size_t(num_label_cols);
        task_bytes_per_float[current_task_index] = bytes_per_float;

        pybind11::handle task_normalization_handle = pybind11::handle(PyDict_GetItemString(task_label_normalization.ptr(), task_name.c_str()));
        if (!task_normalization_handle || !pybind11::isinstance<pybind11::dict>(task_normalization_handle)) {
            continue;
        }
        pybind11::dict normalization_dict = task_normalization_handle.cast<pybind11::dict>();
        pybind11::handle method_handle = pybind11::handle(PyDict_GetItemString(normalization_dict.ptr(), "method"));
        pybind11::handle min_handle = pybind11::handle(PyDict_GetItemString(normalization_dict.ptr(), "min_clipping"));
        pybind11::handle max_handle = pybind11::handle(PyDict_GetItemString(normalization_dict.ptr(), "max_clipping"));
        if (method_handle && pybind11::isinstance<pybind11::str>(method_handle)) {
            std::string method{pybind11::str(method_handle)};
            if (strcmp(method.c_str(), "normal") == 0) {
                task_normalization_options[current_task_index].method = NormalizationMethod::NORMAL;
            }
            else if (strcmp(method.c_str(), "unit") == 0) {
                task_normalization_options[current_task_index].method = NormalizationMethod::UNIT;
            }
        }
        if (min_handle && pybind11::isinstance<pybind11::int_>(min_handle)) {
            task_normalization_options[current_task_index].min_clipping = double(int64_t(min_handle.cast<pybind11::int_>()));
        }
        else if (min_handle && pybind11::isinstance<pybind11::float_>(min_handle)) {
            task_normalization_options[current_task_index].min_clipping = double(min_handle.cast<pybind11::float_>());
        }
        if (max_handle && pybind11::isinstance<pybind11::int_>(max_handle)) {
            task_normalization_options[current_task_index].max_clipping = double(int64_t(max_handle.cast<pybind11::int_>()));
        }
        else if (max_handle && pybind11::isinstance<pybind11::float_>(max_handle)) {
            task_normalization_options[current_task_index].max_clipping = double(max_handle.cast<pybind11::float_>());
        }
    }
    const size_t num_tasks = task_names.size();
    assert(task_index == num_tasks);
    task_col_starts[num_tasks] = total_num_cols;

    // Determine the level of each task's data, for node reordering.
    for (size_t task_index = 0; task_index < num_tasks; ++task_index) {
        pybind11::handle task = task_names[task_index];
        if (!smiles_numpy_arrays[task_index]) {
            continue;
        }
        const std::string task_name{ pybind11::str(task) };

        constexpr const char* graph_prefix = "graph_";
        constexpr const char* node_prefix = "node_";
        constexpr const char* edge_prefix = "edge_";
        constexpr const char* nodepair_prefix = "nodepair_";
        constexpr size_t graph_prefix_length{ std::char_traits<char>::length(graph_prefix) };
        constexpr size_t node_prefix_length{ std::char_traits<char>::length(node_prefix) };
        constexpr size_t edge_prefix_length{ std::char_traits<char>::length(edge_prefix) };
        constexpr size_t nodepair_prefix_length{ std::char_traits<char>::length(nodepair_prefix) };

        if (std::strncmp(task_name.c_str(), graph_prefix, graph_prefix_length) == 0) {
            task_levels[task_index] = FeatureLevel::GRAPH;
        }
        else if (std::strncmp(task_name.c_str(), node_prefix, node_prefix_length) == 0) {
            task_levels[task_index] = FeatureLevel::NODE;
        }
        else if (std::strncmp(task_name.c_str(), edge_prefix, edge_prefix_length) == 0) {
            task_levels[task_index] = FeatureLevel::EDGE;
        }
        else if (std::strncmp(task_name.c_str(), nodepair_prefix, nodepair_prefix_length) == 0) {
            task_levels[task_index] = FeatureLevel::NODEPAIR;
        }
        else {
            // Invalid, but for now, just default to graph-level
            task_levels[task_index] = FeatureLevel::GRAPH;
            continue;
        }
    }
}

static void get_indices_and_strings(
    const pybind11::list& task_names,
    const pybind11::dict& task_train_indices,
    const pybind11::dict& task_val_indices,
    const pybind11::dict& task_test_indices,
    size_t* task_mol_start,
    std::vector<size_t>& task_mol_indices,
    PyArrayObject*const*const smiles_numpy_arrays,
    std::vector<std::string>& smiles_strings
) {
    const size_t num_tasks = task_names.size();

    const pybind11::dict* stage_task_indices[num_stages] = {
        &task_train_indices,
        &task_val_indices,
        &task_test_indices
    };
    
    // Get the total number of molecules, by stage and task
    size_t total_num_mols = 0;
    for (size_t stage_index = 0; stage_index < num_stages; ++stage_index) {
        const pybind11::dict& task_indices_dict = *stage_task_indices[stage_index];

        for (size_t task_index = 0; task_index < num_tasks; ++task_index) {
            pybind11::handle task = task_names[task_index];
            if (!smiles_numpy_arrays[task_index]) {
                continue;
            }
            const std::string task_name{ pybind11::str(task) };
            pybind11::handle task_indices_handle = pybind11::handle(PyDict_GetItemString(task_indices_dict.ptr(), task_name.c_str()));
            if (!task_indices_handle || !pybind11::isinstance<pybind11::list>(task_indices_handle)) {
                printf("Error: Task %s indices list isn't valid.\n", task_name.c_str());
                continue;
            }
            const pybind11::list task_indices_list = task_indices_handle.cast<pybind11::list>();
            const size_t current_num_mols = task_indices_list.size();
            if (current_num_mols == 0) {
                printf("Error: Task %s indices list is empty.\n", task_name.c_str());
            }
            total_num_mols += current_num_mols;
        }
    }
    
    // Get the mol indices for all stages and tasks
    task_mol_indices.reserve(total_num_mols);
    // Unfortunately, reading strings from a numpy array isn't threadsafe,
    // so we have to do that single-threaded first, too.
    smiles_strings.reserve(total_num_mols);
    for (size_t stage_index = 0; stage_index < num_stages; ++stage_index) {
        const pybind11::dict& task_indices_dict = *stage_task_indices[stage_index];

        for (size_t task_index = 0; task_index < num_tasks; ++task_index) {
            // Update task_mol_start here, in case any indices aren't integers
            // or any SMILES strings aren't strings below.
            task_mol_start[stage_index*num_tasks + task_index] = task_mol_indices.size();

            pybind11::handle task = task_names[task_index];
            if (!smiles_numpy_arrays[task_index]) {
                continue;
            }
            const std::string task_name{ pybind11::str(task) };
            pybind11::handle task_indices_handle = pybind11::handle(PyDict_GetItemString(task_indices_dict.ptr(), task_name.c_str()));
            if (!task_indices_handle || !pybind11::isinstance<pybind11::list>(task_indices_handle)) {
                continue;
            }
            
            const pybind11::list task_indices_list = task_indices_handle.cast<pybind11::list>();
            const size_t current_num_mols = task_indices_list.size();
            
            PyArrayObject*const smiles_numpy_array = smiles_numpy_arrays[task_index];
            const size_t smiles_array_size = PyArray_DIM(smiles_numpy_array, 0);
            
            for (size_t indices_index = 0; indices_index < current_num_mols; ++indices_index) {
                const auto list_item = task_indices_list[indices_index];
                if (!pybind11::isinstance<pybind11::int_>(list_item)) {
                    continue;
                }
                
                size_t task_mol_index = size_t(list_item.cast<pybind11::int_>());
                if (task_mol_index >= smiles_array_size) {
                    continue;
                }
                
                pybind11::handle single_smiles_handle(*(PyObject**)PyArray_GETPTR1(smiles_numpy_array, task_mol_index));
                if (!pybind11::isinstance<pybind11::str>(single_smiles_handle)) {
                    continue;
                }
                
                task_mol_indices.push_back(task_mol_index);
                smiles_strings.push_back(std::string(pybind11::str(single_smiles_handle)));
            }

        }
    }
    total_num_mols = task_mol_indices.size();
    task_mol_start[num_stages*num_tasks] = total_num_mols;
}

struct MolKey {
    uint64_t id0;
    uint64_t id1;
    uint32_t num_nodes;
    uint32_t num_edges;
    uint64_t task_index;
    uint64_t task_mol_index;
    uint64_t mol_index;

    bool operator<(const MolKey& other) const {
        if (id0 != other.id0) {
            return (id0 < other.id0);
        }
        if (id1 != other.id1) {
            return (id1 < other.id1);
        }
        if (num_nodes != other.num_nodes) {
            return (num_nodes < other.num_nodes);
        }
        if (num_edges != other.num_edges) {
            return (num_edges < other.num_edges);
        }
        if (task_index != other.task_index) {
            return (task_index < other.task_index);
        }
        return (task_mol_index < other.task_mol_index);
    }
    
    // This is used for identifying keys of molecules with invalid SMILES strings.
    // They show up as having no nodes, no edges, and ID 0.
    bool isInvalid() const {
        return id0 == 0 && id1 == 0 && num_nodes == 0 && num_edges == 0;
    }
};

static void compute_mol_keys(
    MolKey*const keys,
    const size_t total_num_mols,
    const size_t num_tasks,
    int max_threads,
    const size_t*const task_mol_start,
    const bool add_self_loop,
    const bool explicit_H,
    const bool merge_equivalent_mols,
    const size_t*const task_mol_indices,
    const std::vector<std::string>& smiles_strings) {

    // Determine the number of threads to use for computing MolKey values
    const size_t num_mols_per_block = 512;
    const size_t num_blocks = (total_num_mols + num_mols_per_block-1) / num_mols_per_block;
    const size_t num_processors = std::thread::hardware_concurrency();
    size_t num_threads = (num_processors == 1 || num_blocks <= 4) ? 1 : std::min(num_processors, num_blocks/2);
    // max_threads of -1 means n-1 threads, to avoid starving other processes
    if (max_threads < 0) {
        max_threads += num_processors;
        // Don't hit zero or remain negative, because that would skip applying the limit
        if (max_threads < 1) {
            max_threads = 1;
        }
    }
    // max_threads of 0 means to not limit the number of threads
    if (max_threads > 0 && num_threads > size_t(max_threads)) {
        num_threads = size_t(max_threads);
    }

    auto&& get_single_mol_key = [task_mol_start,add_self_loop,explicit_H,task_mol_indices,&smiles_strings,num_tasks,merge_equivalent_mols](size_t mol_index) -> MolKey {
        // Find which task this mol is in.  If there could be many tasks,
        // this could be a binary search, but for small numbers of tasks,
        // a linear search is fine.
        size_t task_index = 0;
        while (task_mol_start[task_index+1] <= mol_index) {
            ++task_index;
        }
        const size_t task_mol_index = task_mol_indices[mol_index];

        const std::string& smiles_str = smiles_strings[mol_index];
        MolBriefData mol_data = smiles_to_brief_data(smiles_str, add_self_loop, explicit_H, merge_equivalent_mols);

        if (!merge_equivalent_mols) {
            // mol_index is, by definition, distinct for each input index,
            // so no molecules will be identified as equivalent below.
            mol_data.unique_id[0] = mol_index;
            mol_data.unique_id[1] = 0;
        }

        return MolKey{mol_data.unique_id[0], mol_data.unique_id[1], mol_data.num_nodes, mol_data.num_edges, task_index % num_tasks, task_mol_index, mol_index};
    };
    if (num_threads == 1) {
        for (size_t mol_index = 0; mol_index < total_num_mols; ++mol_index) {
            keys[mol_index] = get_single_mol_key(mol_index);
        }
    }
    else {
        std::atomic<size_t> next_block_index(0);
        auto&& thread_functor = [keys,&next_block_index,num_blocks,num_mols_per_block,total_num_mols,&get_single_mol_key]() {
            while (true) {
                const size_t block_index = next_block_index.fetch_add(1);
                if (block_index >= num_blocks) {
                    return;
                }
                const size_t begin_index = block_index * num_mols_per_block;
                const size_t end_index = std::min((block_index+1) * num_mols_per_block, total_num_mols);
                for (size_t mol_index = begin_index; mol_index < end_index; ++mol_index) {
                    keys[mol_index] = get_single_mol_key(mol_index);
                }
            }
        };
        std::vector<std::thread> threads;
        for (size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
            threads.push_back(std::thread(thread_functor));
        }
        for (size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
            threads[thread_index].join();
        }
    }
}

constexpr size_t stat_min_offset = 0;
constexpr size_t stat_max_offset = 1;
constexpr size_t stat_mean_offset = 2;
constexpr size_t stat_std_offset = 3;
constexpr size_t num_stats = 4;

static auto compute_stats(
    const std::filesystem::path& common_path,
    const size_t total_num_cols,
    const pybind11::list& task_names,
    const size_t*const task_mol_start,
    const size_t*const task_col_starts,
    const size_t*const task_bytes_per_float,
    const NormalizationOptions*const task_normalization_options,
    PyArrayObject*const*const labels_numpy_arrays,
    PyArrayObject*const*const label_offsets_numpy_arrays,
    const MolKey*const keys,
    std::unique_ptr<double[]>& all_task_stats) {

    const size_t num_tasks = task_names.size();

    // Compute stats on the train stage only (stage 0), like how the python code did it.
    // Normalization will be applied to all stages later.
    // TODO: Does it matter that stats calculations will include all copies of molecules
    // that occur multiple times in the same dataset?
    size_t stats_floats = num_stats*total_num_cols;
    all_task_stats.reset((stats_floats > 0) ? new double[stats_floats] : nullptr);
    std::unordered_map<std::string, std::vector<at::Tensor>> all_stats_return_data;

    if (total_num_cols > 0) {
        std::unique_ptr<intptr_t[]> all_task_num_non_nan(new intptr_t[total_num_cols]);
        for (size_t task_index = 0; task_index < num_tasks; ++task_index) {
            const size_t task_num_mols = task_mol_start[task_index+1] - task_mol_start[task_index];
            const size_t task_first_col = task_col_starts[task_index];
            const size_t task_num_cols = task_col_starts[task_index+1] - task_first_col;
            if (task_num_mols == 0 || task_num_cols == 0) {
                continue;
            }
            // Initialize stats for accumulation
            double*const task_stats = all_task_stats.get() + num_stats*task_first_col;
            intptr_t*const task_num_non_nan = all_task_num_non_nan.get() + task_first_col;
            for (size_t task_col_index = 0; task_col_index < task_num_cols; ++task_col_index) {
                task_stats[num_stats*task_col_index + stat_min_offset] = std::numeric_limits<double>::infinity();
                task_stats[num_stats*task_col_index + stat_max_offset] = -std::numeric_limits<double>::infinity();
                task_stats[num_stats*task_col_index + stat_mean_offset] = 0.0;
                task_stats[num_stats*task_col_index + stat_std_offset] = 0.0;
                task_num_non_nan[task_col_index] = 0;
            }
            
            const size_t bytes_per_float = task_bytes_per_float[task_index];

            auto&& update_stats_single_row = [task_stats, task_num_non_nan](const char* col_data, const size_t task_num_cols, const size_t bytes_per_float, const intptr_t col_stride) {
                double* stats = task_stats;
                intptr_t* num_non_nan = task_num_non_nan;
                for (size_t task_col_index = 0; task_col_index < task_num_cols; ++task_col_index, col_data += col_stride, stats += num_stats, ++num_non_nan) {
                    // TODO: Move the type check outside the loop if it's a bottleneck
                    double value;
                    if (bytes_per_float == sizeof(double)) {
                        value = *(const double*)(col_data);
                    }
                    else if (bytes_per_float == sizeof(float)) {
                        value = *(const float*)(col_data);
                    }
                    else {
                        assert(bytes_per_float == sizeof(uint16_t));
                        value = c10::detail::fp16_ieee_to_fp32_value(*(const uint16_t*)(col_data));
                    }
                    if (value != value) {
                        // NaN value, so skip it
                        continue;
                    }
                    stats[stat_min_offset] = std::min(stats[stat_min_offset], value);
                    stats[stat_max_offset] = std::max(stats[stat_max_offset], value);
                    stats[stat_mean_offset] += value;
                    // TODO: If summing the squares isn't accurate enough for computing the variance,
                    // consider other approaches.
                    stats[stat_std_offset] += value*value;
                    ++(*num_non_nan);
                }
            };

            PyArrayObject*const labels_numpy_array = labels_numpy_arrays[task_index];
            if (labels_numpy_array != nullptr) {
                const char* raw_data = (const char*)PyArray_DATA(labels_numpy_array);
                const intptr_t* strides = PyArray_STRIDES(labels_numpy_array);
                const intptr_t num_label_rows = PyArray_DIM(labels_numpy_array, 0);
                PyArrayObject*const label_offsets_numpy_array = label_offsets_numpy_arrays[task_index];
                const char* offsets_raw_data = label_offsets_numpy_array ? (const char*)PyArray_DATA(label_offsets_numpy_array) : nullptr;
                const intptr_t offsets_stride = label_offsets_numpy_array ? PyArray_STRIDES(label_offsets_numpy_array)[0] : 0;
                // The -1 is because there's an extra entry at the end for the end offset.
                const intptr_t num_mols = label_offsets_numpy_array ? PyArray_DIM(label_offsets_numpy_array, 0) - 1 : num_label_rows;
                // The normalization is computed on the subsample being kept
                for (size_t task_key_index = 0; task_key_index < task_num_mols; ++task_key_index) {
                    const size_t task_mol_index = keys[task_mol_start[task_index] + task_key_index].task_mol_index;
                    if (task_mol_index >= size_t(num_mols)) {
                        printf("Error: In task %zu, mol index %zu is past limit of %zu\n", size_t(task_index), task_mol_index, size_t(num_mols));
                        continue;
                    }
                    if (offsets_raw_data == nullptr) {
                        const char* row_data = raw_data + strides[0]*task_mol_index;
                        update_stats_single_row(row_data, task_num_cols, bytes_per_float, strides[1]);
                    }
                    else {
                        size_t begin_offset = *reinterpret_cast<const int64_t*>(offsets_raw_data + offsets_stride*task_mol_index);
                        size_t end_offset = *reinterpret_cast<const int64_t*>(offsets_raw_data + offsets_stride*(task_mol_index+1));
                        const char* row_data = raw_data + strides[0]*begin_offset;
                        for (size_t row = begin_offset; row < end_offset; ++row, row_data += strides[0]) {
                            update_stats_single_row(row_data, task_num_cols, bytes_per_float, strides[1]);
                        }
                    }
                }
            }

#if GRAPHIUM_CPP_DEBUGGING
            printf("Task %zu normalization method %zu\n", size_t(task_index), size_t(task_normalization_options[task_index].method));
            for (size_t task_col_index = 0; task_col_index < task_num_cols; ++task_col_index) {
                printf("Task %zu col %zu, num non-nan = %zu, min = %e, max = %e\n",
                       size_t(task_index), task_col_index,
                       size_t(task_num_non_nan[task_col_index]),
                       task_stats[num_stats*task_col_index + stat_min_offset],
                       task_stats[num_stats*task_col_index + stat_max_offset]);
            }
#endif
        }

        for (size_t task_index = 0; task_index < num_tasks; ++task_index) {
            const size_t task_first_col = task_col_starts[task_index];
            const size_t task_num_cols = task_col_starts[task_index+1] - task_first_col;
            if (task_num_cols == 0) {
                continue;
            }

            // Finish accumulation
            double*const task_stats = all_task_stats.get() + num_stats*task_first_col;
            intptr_t*const task_num_non_nan = all_task_num_non_nan.get() + task_first_col;
            for (size_t task_col_index = 0; task_col_index < task_num_cols; ++task_col_index) {
                if (task_num_non_nan[task_col_index] == 0) {
                    task_stats[num_stats*task_col_index + stat_min_offset] = std::numeric_limits<double>::quiet_NaN();
                    task_stats[num_stats*task_col_index + stat_max_offset] = std::numeric_limits<double>::quiet_NaN();
                    task_stats[num_stats*task_col_index + stat_mean_offset] = std::numeric_limits<double>::quiet_NaN();
                    task_stats[num_stats*task_col_index + stat_std_offset] = std::numeric_limits<double>::quiet_NaN();
                }
                else {
                    if (task_normalization_options[task_index].min_clipping > task_stats[num_stats*task_col_index + stat_min_offset]) {
                        task_stats[num_stats*task_col_index + stat_min_offset] = task_normalization_options[task_index].min_clipping;
                    }
                    if (task_normalization_options[task_index].max_clipping < task_stats[num_stats*task_col_index + stat_max_offset]) {
                        task_stats[num_stats*task_col_index + stat_max_offset] = task_normalization_options[task_index].max_clipping;
                    }
                    const double n = double(task_num_non_nan[task_col_index]);
                    const double mean = task_stats[num_stats*task_col_index + stat_mean_offset] / n;
                    task_stats[num_stats*task_col_index + stat_mean_offset] = mean;
                    //   sum((x[i] - m)^2)/(n-1)
                    // = sum(x[i]^2 -2mx[i] + m^2)/(n-1)
                    // = (sum(x[i]^2) - 2nm^2 + nm^2)/(n-1)
                    // = (sum(x[i]^2) - nm^2)/(n-1)
                    // except, for compatibility with numpy.nanstd, use n instead of n-1
                    const double sum_sqaures = task_stats[num_stats*task_col_index + stat_std_offset];
                    const double stdev = std::sqrt((sum_sqaures - n*mean*mean)/n);
                    task_stats[num_stats*task_col_index + stat_std_offset] = stdev;
                }
            }

            const std::string task_name{ pybind11::str(task_names[task_index]) };
#if GRAPHIUM_CPP_DEBUGGING
            for (size_t task_col_index = 0; task_col_index < task_num_cols; ++task_col_index) {
                printf("%s %zu %lld %e %e %e %e\n",
                    task_name.c_str(), task_col_index, (long long)task_num_non_nan[task_col_index],
                    task_stats[num_stats*task_col_index + stat_min_offset],
                    task_stats[num_stats*task_col_index + stat_max_offset],
                    task_stats[num_stats*task_col_index + stat_mean_offset],
                    task_stats[num_stats*task_col_index + stat_std_offset]);
            }
#endif
            const std::string stats_filename = task_name + "_stats.tmp";
            save_array_to_file(common_path, stats_filename.c_str(), task_stats, num_stats*task_num_cols);

            // Make copies for returning in a format similar to the load_stats function.
            std::vector<at::Tensor> task_stats_out;
            for (size_t stat_index = 0; stat_index < num_stats; ++stat_index) {
                const int64_t task_stats_dims[1] = { int64_t(task_num_cols) };
                std::unique_ptr<double[]> task_stats_copy(new double[task_num_cols]);
                for (size_t task_col_index = 0; task_col_index < task_num_cols; ++task_col_index) {
                    task_stats_copy[task_col_index] = task_stats[num_stats*task_col_index + stat_index];
                }
                at::Tensor task_stats_tensor = torch_tensor_from_array(std::move(task_stats_copy), task_stats_dims, 1, c10::ScalarType::Double);
                task_stats_out.push_back(std::move(task_stats_tensor));
            }
            all_stats_return_data.insert(std::make_pair(std::move(task_name), std::move(task_stats_out)));
        }
    }
    
    return all_stats_return_data;
}

static auto save_non_label_data(
    const std::filesystem::path* stage_paths,
    const size_t num_tasks,
    const size_t*const task_mol_start,
    const MolKey*const keys,
    const std::vector<std::string>& smiles_strings,
    const size_t total_num_cols) {

    std::unordered_map<std::string, std::vector<at::Tensor>> per_stage_return_data;

    for (size_t stage_index = 0; stage_index < num_stages; ++stage_index) {
        size_t concatenated_smiles_size = 0;
        uint64_t num_unique_mols = 0;
        const size_t stage_begin_index = task_mol_start[stage_index*num_tasks];
        const size_t stage_end_index = task_mol_start[(stage_index+1)*num_tasks];
        for (size_t sorted_index = stage_begin_index; sorted_index < stage_end_index; ) {
            if (keys[sorted_index].isInvalid()) {
                ++sorted_index;
                continue;
            }
            ++num_unique_mols;

            // Add the length of the smiles string to the total length,
            // and include the terminating zero
            const size_t smiles_length = smiles_strings[keys[sorted_index].mol_index].size();
            concatenated_smiles_size += (smiles_length+1);
            
            const uint64_t id0 = keys[sorted_index].id0;
            const uint64_t id1 = keys[sorted_index].id1;
            ++sorted_index;
            while (sorted_index < stage_end_index && keys[sorted_index].id0 == id0 && keys[sorted_index].id1 == id1) {
                ++sorted_index;
            }
        }

        std::unique_ptr<char[]> concatenated_smiles(new char[concatenated_smiles_size]);
        std::unique_ptr<int64_t[]> smiles_offsets(new int64_t[num_unique_mols+1]);
        std::unique_ptr<int32_t[]> num_nodes(new int32_t[num_unique_mols]);
        std::unique_ptr<int32_t[]> num_edges(new int32_t[num_unique_mols]);
        size_t unique_index = 0;
        int64_t smiles_offset = 0;
        for (size_t sorted_index = stage_begin_index; sorted_index < stage_end_index; ) {
            if (keys[sorted_index].isInvalid()) {
                ++sorted_index;
                continue;
            }
            smiles_offsets[unique_index] = smiles_offset;
            
            const uint64_t id0 = keys[sorted_index].id0;
            const uint64_t id1 = keys[sorted_index].id1;
            num_nodes[unique_index] = keys[sorted_index].num_nodes;
            num_edges[unique_index] = keys[sorted_index].num_edges;
            
            // Copy the string
            const std::string& smiles_string = smiles_strings[keys[sorted_index].mol_index];
            const size_t smiles_length = smiles_string.size();
            memcpy(concatenated_smiles.get() + smiles_offset, smiles_string.c_str(), smiles_length);
            smiles_offset += smiles_length;
            // Don't forget the terminating zero
            concatenated_smiles[smiles_offset] = 0;
            ++smiles_offset;
            
            ++unique_index;
            ++sorted_index;
            while (sorted_index < stage_end_index && keys[sorted_index].id0 == id0 && keys[sorted_index].id1 == id1) {
                ++sorted_index;
            }
        }
        smiles_offsets[unique_index] = smiles_offset;
        
        save_array_to_file(stage_paths[stage_index], concat_smiles_filename, concatenated_smiles.get(), concatenated_smiles_size);
        save_array_to_file(stage_paths[stage_index], smiles_offsets_filename, smiles_offsets.get(), num_unique_mols+1);
        save_array_to_file(stage_paths[stage_index], num_nodes_filename, num_nodes.get(), num_unique_mols);
        save_array_to_file(stage_paths[stage_index], num_edges_filename, num_edges.get(), num_unique_mols);
        
        const int64_t concatenated_smiles_dims[1] = { int64_t(concatenated_smiles_size) };
        at::Tensor smiles_tensor = torch_tensor_from_array(std::move(concatenated_smiles), concatenated_smiles_dims, 1, c10::ScalarType::Char);
        const int64_t smiles_offsets_dims[1] = { int64_t(num_unique_mols+1) };
        at::Tensor smiles_offsets_tensor = torch_tensor_from_array(std::move(smiles_offsets), smiles_offsets_dims, 1, c10::ScalarType::Long);
        const int64_t num_nodes_dims[1] = { int64_t(num_unique_mols) };
        at::Tensor num_nodes_tensor = torch_tensor_from_array(std::move(num_nodes), num_nodes_dims, 1, c10::ScalarType::Int);
        const int64_t num_edges_dims[1] = { int64_t(num_unique_mols) };
        at::Tensor num_edges_tensor = torch_tensor_from_array(std::move(num_edges), num_edges_dims, 1, c10::ScalarType::Int);

        std::vector<at::Tensor> stage_return_data;
        // Reserve space for one extra, for the data offsets tensor later
        stage_return_data.reserve((total_num_cols > 0) ? 5 : 4);
        stage_return_data.push_back(std::move(smiles_tensor));
        stage_return_data.push_back(std::move(smiles_offsets_tensor));
        stage_return_data.push_back(std::move(num_nodes_tensor));
        stage_return_data.push_back(std::move(num_edges_tensor));
        per_stage_return_data.insert(std::make_pair(stages[stage_index], std::move(stage_return_data)));
    }

    return per_stage_return_data;
}

static void save_label_data(
    std::unordered_map<std::string, std::vector<at::Tensor>>& per_stage_return_data,
    const std::filesystem::path* stage_paths,
    const size_t num_tasks,
    const size_t*const task_mol_start,
    const size_t*const task_col_starts,
    const size_t total_num_cols,
    const MolKey*const keys,
    PyArrayObject*const*const labels_numpy_arrays,
    PyArrayObject*const*const label_offsets_numpy_arrays,
    const NormalizationOptions*const task_normalization_options,
    const double*const all_task_stats,
    const size_t*const task_bytes_per_float,
    const FeatureLevel*const task_levels,
    const std::vector<std::string>& smiles_strings,
    const bool explicit_H) {

    // mol_data_offsets will only need one entry for each unique molecule,
    // plus one per file, but we can preallocate an upper bound.
    std::vector<uint64_t> mol_data_offsets;
    size_t upper_bound_num_files = (task_mol_start[num_tasks] + num_mols_per_file-1) / num_mols_per_file;
    mol_data_offsets.reserve(task_mol_start[num_tasks] + upper_bound_num_files);

    // temp_data is used for normalization
    std::vector<char> temp_data;
    temp_data.reserve(total_num_cols*sizeof(double));

    std::vector<char> data;
    data.reserve(num_mols_per_file*(total_num_cols*sizeof(double) + (1+2*num_tasks)*sizeof(uint64_t)));

    // These are for reordering label data at node, edge, or nodepair level
    // when the same molecule may appear in multiple tasks with different
    // atom orders.
    std::vector<unsigned int> first_atom_order;
    std::vector<unsigned int> current_atom_order;
    std::vector<unsigned int> inverse_atom_order;

    // Now, deal with label data
    for (size_t stage_index = 0; stage_index < num_stages; ++stage_index) {
        mol_data_offsets.resize(0);
        assert(data.size() == 0);
        uint64_t num_unique_mols = 0;
        const size_t stage_begin_index = task_mol_start[stage_index*num_tasks];
        const size_t stage_end_index = task_mol_start[(stage_index+1)*num_tasks];
        for (size_t sorted_index = stage_begin_index; sorted_index < stage_end_index; ) {
            if (keys[sorted_index].isInvalid()) {
                ++sorted_index;
                continue;
            }
            size_t data_offset = data.size();
            mol_data_offsets.push_back(data_offset);

            const size_t first_sorted_index = sorted_index;
            const uint64_t id0 = keys[sorted_index].id0;
            const uint64_t id1 = keys[sorted_index].id1;
            
            uint64_t prev_task_index = keys[sorted_index].task_index;
            uint64_t mol_num_tasks = 1;
            ++sorted_index;
            while (sorted_index < stage_end_index && keys[sorted_index].id0 == id0 && keys[sorted_index].id1 == id1) {
                // The same molecule can occur multiple times in a single dataset,
                // but we only want to keep one copy for each task.
                if (keys[sorted_index].task_index != prev_task_index) {
                    ++mol_num_tasks;
                    prev_task_index = keys[sorted_index].task_index;
                }
                ++sorted_index;
            }
            assert(mol_num_tasks <= num_tasks);
            assert(!merge_equivalent_mols || mol_num_tasks == 1);

            // TODO: Double data capacity as needed if resizing is slow
            assert(data.size() == data_offset);
            data.resize(data_offset + sizeof(uint64_t)*(1+2*mol_num_tasks));

            // Copy in the number of tasks for this molecule, followed by a list of the task indices and their end offsets.
            memcpy(data.data() + data_offset, &mol_num_tasks, sizeof(uint64_t));
            data_offset += sizeof(uint64_t);
            uint64_t task_offset = 0;
            // Start with an invalid prev_task_index to pick up the first task
            prev_task_index = uint64_t(int64_t(-1));
            for (size_t i = first_sorted_index; i < sorted_index; ++i) {
                const uint64_t task_index = keys[i].task_index;
                // The same molecule can occur multiple times in a single dataset,
                // but we only want to keep one copy for each task.
                if (task_index == prev_task_index) {
                    continue;
                }
                prev_task_index = task_index;
                size_t num_cols = task_col_starts[task_index+1] - task_col_starts[task_index];
                PyArrayObject*const label_offsets_numpy_array = label_offsets_numpy_arrays[task_index];
                if (label_offsets_numpy_array != nullptr) {
                    const size_t task_mol_index = keys[i].task_mol_index;
                    const char* offsets_raw_data = (const char*)PyArray_DATA(label_offsets_numpy_array);
                    const intptr_t offsets_stride = PyArray_STRIDES(label_offsets_numpy_array)[0];
                    const int64_t begin_offset = *reinterpret_cast<const int64_t*>(offsets_raw_data + offsets_stride*task_mol_index);
                    const int64_t end_offset = *reinterpret_cast<const int64_t*>(offsets_raw_data + offsets_stride*(task_mol_index+1));
                    const size_t current_rows = size_t(end_offset - begin_offset);
                    num_cols *= current_rows;
                }
                task_offset += task_bytes_per_float[task_index]*num_cols;
                memcpy(data.data() + data_offset, &task_index, sizeof(uint64_t));
                data_offset += sizeof(uint64_t);
                memcpy(data.data() + data_offset, &task_offset, sizeof(uint64_t));
                data_offset += sizeof(uint64_t);
            }

            // TODO: Double data capacity as needed if resizing is slow
            assert(data.size() == data_offset);
            data.resize(data_offset + task_offset);

            auto&& store_single_row = [&data_offset, &data, &temp_data](
                const char* col_data,
                const size_t task_num_cols,
                const intptr_t col_stride,
                const size_t in_bytes_per_float,
                const size_t out_bytes_per_float,
                const NormalizationMethod normalization_method,
                const double* task_stats) {
                
                if (size_t(col_stride) == in_bytes_per_float) {
                    memcpy(temp_data.data(), col_data, in_bytes_per_float*task_num_cols);
                }
                else {
                    for (size_t col = 0; col < task_num_cols; ++col) {
                        memcpy(temp_data.data() + col*in_bytes_per_float, col_data, in_bytes_per_float);
                        col_data += col_stride;
                    }
                }
                for (size_t col = 0; col < task_num_cols; ++col) {
                    double value;
                    if (in_bytes_per_float == sizeof(double)) {
                        value = ((const double*)(temp_data.data()))[col];
                    }
                    else if (in_bytes_per_float == sizeof(float)) {
                        value = ((const float*)(temp_data.data()))[col];
                    }
                    else {
                        assert(in_bytes_per_float == sizeof(uint16_t));
                        value = c10::detail::fp16_ieee_to_fp32_value(((const uint16_t*)(temp_data.data()))[col]);
                    }
                    value = std::max(value, task_stats[stat_min_offset]);
                    value = std::min(value, task_stats[stat_max_offset]);
                    if (normalization_method == NormalizationMethod::NORMAL) {
                        if (task_stats[stat_std_offset] != 0) {
                            value = (value - task_stats[stat_mean_offset])/task_stats[stat_std_offset];
                        }
                        else {
                            value = 0;
                        }
                    }
                    else if (normalization_method == NormalizationMethod::UNIT) {
                        // TODO: Cache 1/(max-min) or 0 to avoid check
                        if (task_stats[stat_max_offset] - task_stats[stat_min_offset] != 0) {
                            value = (value - task_stats[stat_min_offset])/(task_stats[stat_max_offset] - task_stats[stat_min_offset]);
                        }
                        else {
                            value = 0;
                        }
                    }

                    // NOTE: The code below writes to temp_data, which is still being read from above,
                    // so this relies on that we're not writing to a larger data type than we're reading,
                    // else we'll overwrite data.
                    assert(out_bytes_per_float <= in_bytes_per_float);
                    if (out_bytes_per_float == sizeof(double)) {
                        ((double*)(temp_data.data()))[col] = value;
                    }
                    else if (out_bytes_per_float == sizeof(float)) {
                        ((float*)(temp_data.data()))[col] = float(value);
                    }
                    else {
                        assert(out_bytes_per_float == sizeof(uint16_t));
                        ((uint16_t*)(temp_data.data()))[col] = c10::detail::fp16_ieee_from_fp32_value(value);
                    }
                    task_stats += num_stats;
                }

                memcpy(data.data() + data_offset, temp_data.data(), out_bytes_per_float*task_num_cols);
                data_offset += out_bytes_per_float*task_num_cols;
            };

            // Copy in the task data, with optional normalization
            // Start with an invalid prev_task_index to pick up the first task
            prev_task_index = uint64_t(int64_t(-1));
            for (size_t i = first_sorted_index; i < sorted_index; ++i) {
                const uint64_t task_index = keys[i].task_index;
                // The same molecule can occur multiple times in a single dataset,
                // but we only want to keep one copy for each task.
                if (task_index == prev_task_index) {
                    continue;
                }
                prev_task_index = task_index;
                
                const uint64_t task_mol_index = keys[i].task_mol_index;
                
                const size_t task_first_col = task_col_starts[task_index];
                const size_t task_num_cols = task_col_starts[task_index+1] - task_first_col;
                const NormalizationOptions& normalization = task_normalization_options[task_index];
                const double* task_stats = all_task_stats + num_stats*task_first_col;

                const size_t bytes_per_float = task_bytes_per_float[task_index];
                
                // Before copying this task's label data, check whether the atom order
                // is different from the representative SMILES string's atom order.
                bool same_order_as_first = true;
                if (i != first_sorted_index && task_levels[task_index] != FeatureLevel::GRAPH) {
                    const std::string& first_string = smiles_strings[keys[first_sorted_index].mol_index];
                    const std::string& current_string = smiles_strings[keys[i].mol_index];
                    if (first_string != current_string) {
                        // Different string, so get first and current atom orders
                        if (first_atom_order.size() == 0) {
                            std::unique_ptr<RDKit::RWMol> mol = parse_mol(first_string, explicit_H);
                            get_canonical_atom_order(*mol, first_atom_order);
                        }
                        std::unique_ptr<RDKit::RWMol> mol = parse_mol(current_string, explicit_H);
                        get_canonical_atom_order(*mol, current_atom_order);
                        assert(first_atom_order.size() == current_atom_order.size());
                        
                        // first_atom_order maps from the first order to the canonical order.
                        // current_atom_order maps from the first order to the canonical order.
                        // We need the inverse current map, to go from the first order to the
                        // canonical order, and then from there to the current order.
                        inverse_atom_order.resize(first_atom_order.size());
                        for (unsigned int current_index = 0; current_index < current_atom_order.size(); ++current_index) {
                            unsigned int canon_index = current_atom_order[current_index];
                            assert(canon_index < inverse_atom_order.size());
                            inverse_atom_order[canon_index] = current_index;
                        }
                        for (unsigned int first_index = 0; first_index < first_atom_order.size(); ++first_index) {
                            unsigned int canon_index = first_atom_order[first_index];
                            assert(canon_index < inverse_atom_order.size());
                            unsigned int current_index = inverse_atom_order[canon_index];
                            assert(first_index < current_atom_order.size());
                            current_atom_order[first_index] = current_index;
                            if (current_index != first_index) {
                                same_order_as_first = false;
                            }
                        }
                    }
                }

                PyArrayObject*const labels_numpy_array = labels_numpy_arrays[task_index];
                if (labels_numpy_array != nullptr) {
                    const char* raw_data = (const char*)PyArray_DATA(labels_numpy_array);
                    const intptr_t* strides = PyArray_STRIDES(labels_numpy_array);
                    PyArrayObject*const label_offsets_numpy_array = label_offsets_numpy_arrays[task_index];
                    const char* offsets_raw_data = label_offsets_numpy_array ? (const char*)PyArray_DATA(label_offsets_numpy_array) : nullptr;
                    const intptr_t offsets_stride = label_offsets_numpy_array ? PyArray_STRIDES(label_offsets_numpy_array)[0] : 0;
                    if (offsets_raw_data == nullptr) {
                        const char* row_data = raw_data + strides[0]*task_mol_index;
                        store_single_row(row_data, task_num_cols, strides[1], bytes_per_float, bytes_per_float, normalization.method, task_stats);
                    }
                    else {
                        size_t begin_offset = *reinterpret_cast<const int64_t*>(offsets_raw_data + offsets_stride*task_mol_index);
                        size_t end_offset = *reinterpret_cast<const int64_t*>(offsets_raw_data + offsets_stride*(task_mol_index+1));
                        const char* row_data = raw_data + strides[0]*begin_offset;
                        if (same_order_as_first) {
                            for (size_t row = begin_offset; row < end_offset; ++row, row_data += strides[0]) {
                                store_single_row(row_data, task_num_cols, strides[1], bytes_per_float, bytes_per_float, normalization.method, task_stats);
                            }
                        }
                        else if (task_levels[task_index] == FeatureLevel::NODE) {
                            assert(end_offset - begin_offset == current_atom_order.size());
                            for (unsigned int current_index : current_atom_order) {
                                store_single_row(row_data + current_index*strides[0], task_num_cols, strides[1], bytes_per_float, bytes_per_float, normalization.method, task_stats);
                            }
                        }
                        else if (task_levels[task_index] == FeatureLevel::NODEPAIR) {
                            const size_t n = current_atom_order.size();
                            assert(end_offset - begin_offset == n*n);
                            for (unsigned int current_index0 : current_atom_order) {
                                for (unsigned int current_index1 : current_atom_order) {
                                    store_single_row(row_data + (current_index0*n + current_index1)*strides[0], task_num_cols, strides[1], bytes_per_float, bytes_per_float, normalization.method, task_stats);
                                }
                            }
                        }
                        else {
                            assert(task_levels[task_index] == FeatureLevel::EDGE);
                            // FIXME: Re-order edge-level data, too
                            for (size_t row = begin_offset; row < end_offset; ++row, row_data += strides[0]) {
                                store_single_row(row_data, task_num_cols, strides[1], bytes_per_float, bytes_per_float, normalization.method, task_stats);
                            }
                        }
                    }
                }
            }
            first_atom_order.resize(0);
            current_atom_order.resize(0);
            inverse_atom_order.resize(0);

            ++num_unique_mols;
            if (num_unique_mols % num_mols_per_file == 0 || sorted_index == stage_end_index) {
                // Write out the data to a file
                
                // First, construct the filename
                char filename[20+4+1];
                size_t file_num = ((num_unique_mols-1) / num_mols_per_file);
                get_mol_label_filename(filename, file_num);
                
                std::filesystem::path file_path(stage_paths[stage_index] / filename);
                FileType file = fopen_write_wrapper(file_path);
                if (file == INVALID_FILE) {
                    return;
                }
#if GRAPHIUM_CPP_DEBUGGING
                printf("Writing file %s\n", file_path.string().c_str());
#endif
                size_t num_bytes_written = fwrite_wrapper(data.data(), data_offset, file);
                fclose_wrapper(file);
                if (num_bytes_written != data_offset) {
                    return;
                }
                data.resize(0);
                
                // One extra data offset to mark the end of each file.
                // data_offset is automatically reset to 0 on the next iteration
                // due to data.size() being 0 now.
                mol_data_offsets.push_back(data_offset);
            }
        }
        
        // Write out the molecule data offsets to a separate file,
        // so that only one file read is needed per molecule when data loading
        // if the offsets are all loaded once and kept in memory.
        // Note the one extra entry per file.
#if GRAPHIUM_CPP_DEBUGGING
        printf("Stage %s has %zu unique mols from %zu original\n", stages[stage_index].c_str(), size_t(num_unique_mols), size_t(stage_end_index - stage_begin_index));
#endif
        assert(mol_data_offsets.size() == num_unique_mols + (num_unique_mols + num_mols_per_file-1)/num_mols_per_file);
        std::filesystem::path file_path(stage_paths[stage_index] / "mol_offsets.tmp");
        FileType file = fopen_write_wrapper(file_path);
        if (file == INVALID_FILE) {
            return;
        }
        size_t num_bytes_written = fwrite_wrapper(&num_unique_mols, sizeof(num_unique_mols), file);
        if (num_bytes_written != sizeof(num_unique_mols)) {
            fclose_wrapper(file);
            return;
        }
        size_t num_offsets = mol_data_offsets.size();
        size_t data_offsets_size = num_offsets*sizeof(mol_data_offsets[0]);
        num_bytes_written = fwrite_wrapper(mol_data_offsets.data(), data_offsets_size, file);
        fclose_wrapper(file);
        if (num_bytes_written != data_offsets_size) {
            return;
        }
        
        static_assert(sizeof(int64_t) == sizeof(mol_data_offsets[0]));
        save_array_to_file(stage_paths[stage_index], file_data_offsets_filename, mol_data_offsets.data(), num_offsets);
        std::unique_ptr<int64_t[]> temp_data_offsets(new int64_t[num_offsets]);
        memcpy(temp_data_offsets.get(), mol_data_offsets.data(), data_offsets_size);
        const int64_t data_offsets_dims[1] = { int64_t(num_offsets) };
        at::Tensor data_offsets_tensor = torch_tensor_from_array(std::move(temp_data_offsets), data_offsets_dims, 1, c10::ScalarType::Long);
        
        per_stage_return_data[stages[stage_index]].push_back(std::move(data_offsets_tensor));
        mol_data_offsets.resize(0);
    }
}

// Returns:
// stage -> [
//      unique mol smiles strings all concatenated,
//      unique mol smiles string offsets (including one extra for the end),
//      unique mol num_nodes,
//      unique mol num_edges,
//      mol_file_data_offsets
// ]
// task -> 4 stats tensors each
// task index -> label num columns
// task index -> label torch data type enum
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
    bool add_self_loop,
    bool explicit_H,
    int max_threads,
    bool merge_equivalent_mols) {

    ensure_numpy_array_module_initialized();

    const size_t num_tasks = task_names.size();
    std::vector<int64_t> return_label_num_cols(num_tasks, 0);
    std::vector<int32_t> return_label_data_types(num_tasks, -1);
    std::unique_ptr<size_t[]> task_col_starts(new size_t[num_tasks+1]);
    std::unique_ptr<size_t[]> task_bytes_per_float(new size_t[num_tasks]);
    std::unique_ptr<NormalizationOptions[]> task_normalization_options(new NormalizationOptions[num_tasks]);
    std::unique_ptr<PyArrayObject*[]> smiles_numpy_arrays(new PyArrayObject*[num_tasks]);
    std::unique_ptr<PyArrayObject*[]> labels_numpy_arrays(new PyArrayObject*[num_tasks]);
    std::unique_ptr<PyArrayObject*[]> label_offsets_numpy_arrays(new PyArrayObject*[num_tasks]);
    std::unique_ptr<FeatureLevel[]> task_levels(new FeatureLevel[num_tasks]);

    // Figure out the task bounds first, so that everything can be parallelized perfectly.
    get_task_data(
        task_names,
        task_dataset_args,
        task_label_normalization,
        return_label_num_cols.data(),
        return_label_data_types.data(),
        task_col_starts.get(),
        task_bytes_per_float.get(),
        task_normalization_options.get(),
        smiles_numpy_arrays.get(),
        labels_numpy_arrays.get(),
        label_offsets_numpy_arrays.get(),
        task_levels.get());

    const size_t total_num_cols = task_col_starts[num_tasks];

    std::filesystem::path base_path{processed_graph_data_path};
    std::filesystem::create_directories(base_path);
    std::filesystem::path common_path(base_path / data_hash);
    std::filesystem::create_directories(common_path);
    
    if (total_num_cols > 0) {
        save_num_cols_and_dtypes(common_path, return_label_num_cols, return_label_data_types);
    }

    std::unique_ptr<size_t[]> task_mol_start(new size_t[num_stages*num_tasks + 1]);
    std::vector<size_t> task_mol_indices;
    std::vector<std::string> smiles_strings;
    get_indices_and_strings(
        task_names,
        task_train_indices,
        task_val_indices,
        task_test_indices,
        task_mol_start.get(),
        task_mol_indices,
        smiles_numpy_arrays.get(),
        smiles_strings);
    const size_t total_num_mols = task_mol_indices.size();

    // Compute all InChI keys for all molecules, in parallel if applicable.
    std::unique_ptr<MolKey[]> keys(new MolKey[total_num_mols]);
    compute_mol_keys(
        keys.get(),
        total_num_mols,
        num_tasks,
        max_threads,
        task_mol_start.get(),
        add_self_loop,
        explicit_H,
        merge_equivalent_mols,
        task_mol_indices.data(),
        smiles_strings);

    std::unique_ptr<double[]> all_task_stats;
    auto all_stats_return_data = compute_stats(
        common_path,
        total_num_cols,
        task_names,
        task_mol_start.get(),
        task_col_starts.get(),
        task_bytes_per_float.get(),
        task_normalization_options.get(),
        labels_numpy_arrays.get(),
        label_offsets_numpy_arrays.get(),
        keys.get(),
        all_task_stats);

    if (merge_equivalent_mols) {
        // Sort train, val, and test separately, since they need to be stored separately.
        // Don't sort until after accumulating stats, because the code above currently assumes that the tasks
        // aren't interleaved.
        std::sort(keys.get(), keys.get() + task_mol_start[num_tasks]);
        std::sort(keys.get() + task_mol_start[num_tasks], keys.get() + task_mol_start[2*num_tasks]);
        std::sort(keys.get() + task_mol_start[2*num_tasks], keys.get() + total_num_mols);
    }

    std::filesystem::path stage_paths[num_stages] = {
        base_path / (stages[0] + "_" + data_hash),
        base_path / (stages[1] + "_" + data_hash),
        base_path / (stages[2] + "_" + data_hash)
    };
    std::filesystem::create_directories(stage_paths[0]);
    std::filesystem::create_directories(stage_paths[1]);
    std::filesystem::create_directories(stage_paths[2]);

    // Deal with non-label data first (smiles, num_nodes, num_edges)
    auto per_stage_return_data = save_non_label_data(
        stage_paths,
        num_tasks,
        task_mol_start.get(),
        keys.get(),
        smiles_strings,
        total_num_cols);

    if (total_num_cols == 0) {
        // No label data, so all done
        return std::make_tuple(
            std::move(per_stage_return_data),
            std::move(all_stats_return_data),
            std::move(return_label_num_cols),
            std::move(return_label_data_types));
    }

    save_label_data(
        per_stage_return_data,
        stage_paths,
        num_tasks,
        task_mol_start.get(),
        task_col_starts.get(),
        total_num_cols,
        keys.get(),
        labels_numpy_arrays.get(),
        label_offsets_numpy_arrays.get(),
        task_normalization_options.get(),
        all_task_stats.get(),
        task_bytes_per_float.get(),
        task_levels.get(),
        smiles_strings,
        explicit_H);

    return std::make_tuple(
        std::move(per_stage_return_data),
        std::move(all_stats_return_data),
        std::move(return_label_num_cols),
        std::move(return_label_data_types));
}

void load_labels_from_index(
    const std::string stage_directory,
    const int64_t mol_index,
    const at::Tensor& mol_file_data_offsets,
    const pybind11::list& label_names,
    const pybind11::list& label_num_cols,
    const pybind11::list& label_data_types,
    pybind11::dict& labels
) {
    const std::filesystem::path stage_path{stage_directory};
    if (mol_index < 0) {
        printf("Error: In load_labels_from_index, mol_index = %lld\n", (long long)mol_index);
        return;
    }
    const uint64_t file_num = uint64_t(mol_index) / num_mols_per_file;
    const size_t index_into_offsets = file_num*(num_mols_per_file+1) + (uint64_t(mol_index) % num_mols_per_file);

    const size_t num_data_offsets = (mol_file_data_offsets.scalar_type() == c10::ScalarType::Long && mol_file_data_offsets.ndimension() == 1) ? mol_file_data_offsets.size(0) : 0;
    if (index_into_offsets+1 >= num_data_offsets) {
        printf("Error: In load_labels_from_index, mol_index = %zu, index_into_offsets = %zu, num_data_offsets = %zu\n",
            size_t(mol_index), size_t(index_into_offsets), size_t(num_data_offsets));
        return;
    }
    // NOTE: If TensorBase::data_ptr is ever removed, change it to TensorBase::const_data_ptr.
    // Some torch version being used doesn't have const_data_ptr yet.
    const int64_t* const data_offsets = mol_file_data_offsets.data_ptr<int64_t>();
    const int64_t file_begin_offset = data_offsets[index_into_offsets];
    const int64_t file_end_offset = data_offsets[index_into_offsets+1];
    if (file_end_offset < 0 || file_end_offset-file_begin_offset < 8) {
        printf("Error: In load_labels_from_index, mol_index = %zu, file_begin_offset = %lld, file_end_offset = %lld\n",
            size_t(mol_index), (long long)(index_into_offsets), (long long)(num_data_offsets));
        return;
    }
    const size_t file_read_size = size_t(file_end_offset - file_begin_offset);
    
    std::unique_ptr<char[]> data(new char[file_read_size]);
    
    {
        char filename[25];
        get_mol_label_filename(filename, file_num);

        const std::filesystem::path file_path{stage_path / filename};
        FileType file = fopen_read_wrapper(file_path);
        if (file == INVALID_FILE) {
            printf("Error: In load_labels_from_index, failed to open \"%s\" for molecule %zu\n",
                file_path.string().c_str(), size_t(mol_index));
            return;
        }
        int seek_failed = fseek_wrapper(file, file_begin_offset);
        if (seek_failed) {
            printf("Error: In load_labels_from_index, failed to seek to offset %zu in \"%s\" for molecule %zu\n",
                size_t(file_begin_offset), file_path.string().c_str(), size_t(mol_index));
            fclose_wrapper(file);
            return;
        }
        size_t num_bytes_read = fread_wrapper(data.get(), file_read_size, file);
        fclose_wrapper(file);
        if (num_bytes_read != file_read_size) {
            printf("Error: In load_labels_from_index, read only %zu/%zu bytes from \"%s\" for molecule %zu\n",
                size_t(num_bytes_read), size_t(file_read_size), file_path.string().c_str(), size_t(mol_index));
            return;
        }
    }
    
    uint64_t mol_num_tasks = 0;
    memcpy(&mol_num_tasks, data.get(), sizeof(uint64_t));
    size_t data_offset = sizeof(uint64_t);
    if (mol_num_tasks == 0 || mol_num_tasks > label_names.size() || file_read_size < (1+2*mol_num_tasks)*sizeof(uint64_t)) {
        printf("Error: In load_labels_from_index, mol_index = %zu, mol_num_tasks = %zu, file_read_size = %zu\n",
            size_t(mol_index), size_t(mol_num_tasks), size_t(file_read_size));
        return;
    }
    const size_t base_offset = (1+2*mol_num_tasks)*sizeof(uint64_t);
    const char* base_task_data = data.get() + base_offset;
    uint64_t task_offset = 0;
    for (size_t data_task_index = 0; data_task_index < mol_num_tasks; ++data_task_index) {
        uint64_t task_index = 0;
        memcpy(&task_index, data.get() + data_offset, sizeof(uint64_t));
        data_offset += sizeof(uint64_t);
        if (task_index >= label_names.size() || task_index >= label_data_types.size() || task_index >= label_num_cols.size()) {
            printf("Error: In load_labels_from_index, mol_index = %zu, task_index = %zu\n",
                size_t(mol_index), size_t(task_index));
            return;
        }

        uint64_t task_end_offset = 0;
        memcpy(&task_end_offset, data.get() + data_offset, sizeof(uint64_t));
        data_offset += sizeof(uint64_t);
        if (task_end_offset < task_offset || task_end_offset > file_read_size-base_offset) {
            printf("Error: In load_labels_from_index, mol_index = %zu, task_offset = %zu, task_end_offset = %zu, file_read_size = %zu, base_offset = %zu\n",
                size_t(mol_index), size_t(task_offset), size_t(task_end_offset), size_t(file_read_size), size_t(base_offset));
            return;
        }
        
        const size_t task_num_bytes = task_end_offset - task_offset;
        if (!pybind11::isinstance<pybind11::int_>(label_data_types[task_index]) ||
            !pybind11::isinstance<pybind11::int_>(label_num_cols[task_index])) {
            printf("Error: In load_labels_from_index, mol_index = %zu, task_index = %zu, label_data_type = \"%s\", label_num_cols = \"%s\"\n",
                size_t(mol_index), size_t(task_index),
                std::string(pybind11::str(label_data_types[task_index])).c_str(),
                std::string(pybind11::str(label_num_cols[task_index])).c_str());
            return;
        }
        const c10::ScalarType torch_type = c10::ScalarType(size_t(label_data_types[task_index].cast<pybind11::int_>()));
        const size_t num_cols = size_t(label_num_cols[task_index].cast<pybind11::int_>());
        if (num_cols == 0) {
            printf("Error: In load_labels_from_index, mol_index = %zu, task_index = %zu, label_data_type = %zu, label_num_cols = %zu\n",
                size_t(mol_index), size_t(task_index),
                size_t(torch_type), num_cols);
            return;
        }
        const size_t supported_type_index = torch_type_index(torch_type);
        if (supported_type_index >= num_supported_types) {
            printf("Error: In load_labels_from_index, mol_index = %zu, task_index = %zu, label_data_type = %zu, label_num_cols = %zu\n",
                size_t(mol_index), size_t(task_index),
                size_t(torch_type), num_cols);
        }
        const size_t bytes_per_float = supported_types[supported_type_index].size;
        const size_t num_floats = task_num_bytes / bytes_per_float;
        const size_t num_rows = num_floats / num_cols;
        
        if (num_floats != num_rows*num_cols) {
            printf("Error: In load_labels_from_index, mol_index = %zu, task data bytes = %zu (not a multiple of %zu*%zu)\n",
                size_t(mol_index), size_t(task_num_bytes), bytes_per_float, num_cols);
            return;
        }

        const std::string label_name{pybind11::str(label_names[task_index])};
        const bool is_graph_level = (std::strncmp(label_name.c_str(), "graph", 5) == 0);
        if (is_graph_level && num_rows != 1) {
            printf("Error: In load_labels_from_index, mol_index = %zu, num_rows = %zu for task \"%s\"\n",
                size_t(mol_index), num_rows, label_name.c_str());
            return;
        }
        size_t num_label_dims = is_graph_level ? 1 : 2;
        const int64_t label_dims[2] = { (is_graph_level ? int64_t(num_floats) : int64_t(num_rows)), int64_t(num_cols) };
        at::Tensor label_tensor;
        
        if (bytes_per_float == 2) {
            std::unique_ptr<uint16_t[]> label_data(new uint16_t[num_floats]);
            memcpy(label_data.get(), base_task_data + task_offset, task_num_bytes);
            label_tensor = torch_tensor_from_array(std::move(label_data), label_dims, num_label_dims, torch_type);
        }
        else if (bytes_per_float == 4) {
            std::unique_ptr<float[]> label_data(new float[num_floats]);
            memcpy(label_data.get(), base_task_data + task_offset, task_num_bytes);
            label_tensor = torch_tensor_from_array(std::move(label_data), label_dims, num_label_dims, torch_type);
        }
        else if (bytes_per_float == 8) {
            std::unique_ptr<double[]> label_data(new double[num_floats]);
            memcpy(label_data.get(), base_task_data + task_offset, task_num_bytes);
            label_tensor = torch_tensor_from_array(std::move(label_data), label_dims, num_label_dims, torch_type);
        }
        
        PyDict_SetItem(labels.ptr(), label_names[task_index].ptr(), THPVariable_Wrap(std::move(label_tensor)));
        
        task_offset = task_end_offset;
    }
}

std::string extract_string(
    const at::Tensor& concat_strings,
    const at::Tensor& string_offsets,
    const int64_t index) {

    const size_t data_size = (concat_strings.scalar_type() == c10::ScalarType::Char && concat_strings.ndimension() == 1) ? concat_strings.size(0) : 0;
    const size_t num_data_offsets = (string_offsets.scalar_type() == c10::ScalarType::Long && string_offsets.ndimension() == 1) ? string_offsets.size(0) : 0;
    if (index < 0 || size_t(index) >= num_data_offsets) {
        return std::string();
    }
    const char* const data = reinterpret_cast<const char*>(concat_strings.data_ptr<int8_t>());
    const int64_t* const data_offsets = string_offsets.data_ptr<int64_t>();
    int64_t offset = data_offsets[index];
    int64_t end_offset = data_offsets[index+1];
    int64_t size = (end_offset - offset) - 1;
    if (offset < 0 || size < 0 || end_offset > int64_t(data_size)) {
        return std::string();
    }
    return std::string(data + offset, size_t(size));
}
