# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Setup script that builds graphium_cpp.

It requires that a conda environment be created according to the 
env.yml file. This file contains the build dependencies which
are necessary to compile and build graphium_cpp for use by the
graphium package.
"""

import platform
import sys
from distutils.core import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch, rdkit, os
import numpy

python_version = str(sys.version_info[0]) + str(sys.version_info[1])

# Base variables required for compilation
path_separator = "/"
lib_folder_name = "lib"
boost_include = "include/boost"
package_compile_args = []

# Updating variables used during compilation based on OS
system = platform.system()
if system == "Darwin" or system == "Linux":
    package_compile_args += ["-O3", "-Wall", "-Wmaybe-uninitialized", "-Wuninitialized"]

if system == "Darwin":
    package_compile_args += ["-mmacosx-version-min=10.15"]
elif system == "Windows":
    path_separator = "\\"
    lib_folder_name = "Lib"
    package_compile_args += [
        "/DWIN32",
        "/DRDKIT_DYN_LINK",
    ]

# Extracting paths to torch and rdkit dependencies
torch_dir = torch.__path__[0]
rdkit_lib_index = rdkit.__path__[0].split(path_separator).index(lib_folder_name)  # breaks on windows
rdkit_prefix = "/".join(rdkit.__path__[0].split(path_separator)[:rdkit_lib_index])

# Windows-specific changed to rdkit path
if system == "Windows":
    rdkit_prefix += "/Library"
    boost_include = "include"

ext_modules = [
    Pybind11Extension(
        "graphium_cpp",
        sources=[
            "./graphium/graphium_cpp/graphium_cpp.cpp",
            "./graphium/graphium_cpp/features.cpp",
            "./graphium/graphium_cpp/labels.cpp",
            "./graphium/graphium_cpp/commute.cpp",
            "./graphium/graphium_cpp/electrostatic.cpp",
            "./graphium/graphium_cpp/float_features.cpp",
            "./graphium/graphium_cpp/graphormer.cpp",
            "./graphium/graphium_cpp/one_hot.cpp",
            "./graphium/graphium_cpp/random_walk.cpp",
            "./graphium/graphium_cpp/spectral.cpp",
        ],
        language="c++",
        cxx_std=20,
        include_dirs=[
            os.path.join(torch_dir, "include"),
            os.path.join(torch_dir, "include/torch/csrc/api/include"),
            os.path.join(rdkit_prefix, "include/rdkit"),
            os.path.join(rdkit_prefix, boost_include),
            numpy.get_include(),
        ],
        libraries=[
            "RDKitAlignment",
            "RDKitDataStructs",
            "RDKitDistGeometry",
            "RDKitDistGeomHelpers",
            "RDKitEigenSolvers",
            "RDKitForceField",
            "RDKitForceFieldHelpers",
            "RDKitGenericGroups",
            "RDKitGraphMol",
            "RDKitInchi",
            "RDKitRDInchiLib",
            "RDKitRDBoost",
            "RDKitRDGeneral",
            "RDKitRDGeometryLib",
            "RDKitRingDecomposerLib",
            "RDKitSmilesParse",
            "RDKitSubstructMatch",
            "torch_cpu",
            "torch_python",
            "c10",
            f"boost_python{python_version}",
        ],
        library_dirs=[os.path.join(rdkit_prefix, "lib"), os.path.join(torch_dir, "lib")],
        extra_compile_args=package_compile_args,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
