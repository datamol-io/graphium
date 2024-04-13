# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Setup script that builds graphium_cpp.
At time of writing, this has only been tested with GCC 10.5.0.
To build, git clone pybind11 into this directory, then run:
rm -r build/*
export PYTHONPATH=$PYTHONPATH:./pybind11
python ./setup.py build
cp build/lib.linux-x86_64-cpython-311/graphium_cpp.cpython-311-x86_64-linux-gnu.so ~/mambaforge/envs/graphium/bin
"""

from distutils.core import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch, rdkit, os
import numpy

torch_dir = torch.__path__[0]
rdkit_lib_index = rdkit.__path__[0].split('/').index('lib')
rdkit_prefix = '/'.join(rdkit.__path__[0].split('/')[:rdkit_lib_index])

#print(rdkit_lib_index)
#print(rdkit_prefix)

ext_modules = [
        Pybind11Extension(
            'graphium_cpp',
            sources=[
                "graphium_cpp.cpp",
                "features.cpp",
                "labels.cpp",
                "commute.cpp",
                "electrostatic.cpp",
                "float_features.cpp",
                "graphormer.cpp",
                "one_hot.cpp",
                "random_walk.cpp",
                "spectral.cpp"
            ],
            language="c++",
            cxx_std=20,
            include_dirs = [os.path.join(torch_dir,"include"),
                os.path.join(torch_dir,"include/torch/csrc/api/include"),
                os.path.join(rdkit_prefix, "include/rdkit"),
                #"/opt/nvidia/nsight-systems/2023.2.3/target-linux-x64/nvtx/include",
                numpy.get_include()],
            libraries = [
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
                "torch_python"
            ],
            library_dirs = [os.path.join(rdkit_prefix,"lib"),
                os.path.join(torch_dir,"lib")],
            extra_compile_args=["-O3","-Wall", "-Wmissing-field-initializers", "-Wmaybe-uninitialized", "-Wuninitialized"]
        )
]

setup(name = "graphium_cpp",
    version = "0.1",
    author = "N. Dickson",
    author_email="ndickson@nvidia.com",
    license="NVIDIA Proprietary",
    description = "C++ extension for graphium",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext})
