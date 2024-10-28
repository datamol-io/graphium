# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Setup script that builds graphium_cpp.

It requires that a conda environment be created according to the 
build_env.yml file. This file contains the build dependencies which
are necessary to compile and build graphium_cpp for inclusion in the
graphium package.
"""

import platform
from distutils.core import setup
import site
import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

path_to_env = sys.prefix
path_to_site_packages = site.getsitepackages()[0]

numpy_include_dir = os.path.join(path_to_site_packages, "numpy/core/include")
if not os.path.isdir(numpy_include_dir):
    numpy_include_dir = os.path.join(path_to_site_packages, "numpy/_core/include")

system = platform.system()
package_compile_args = [
            "-O3",
            "-Wall",
            "-Wmissing-field-initializers",
            "-Wmaybe-uninitialized",
            "-Wuninitialized",
        ]
if system == "Darwin":
    package_compile_args.append("-mmacosx-version-min=10.15")
elif system == "Windows":
    pass

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
            os.path.join(path_to_site_packages, "torch/include"),
            os.path.join(path_to_site_packages, "torch/include/torch/csrc/api/include"),
            os.path.join(path_to_env, "include/rdkit"),
            os.path.join(path_to_env, "include/boost"),
            numpy_include_dir
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
        ],
        library_dirs=[os.path.join(path_to_env, "lib"), os.path.join(path_to_site_packages, "torch/lib")],
        extra_compile_args=package_compile_args
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
