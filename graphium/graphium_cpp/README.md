<div align="center">
    <img src="../../images/banner-tight.png" height="200px">
    <h3>C++ accelerations of `graphium`'s featurization and data loading processes</h3>
</div>

---

Welcome to the documentation for the C++ API of the `graphium` library. This API is used to:

- 🏗️ Compute positional and structural encodings
- 📁 Cache supervised labels into very light and optimized files

This significantly accelerates dataloading and removes the need for long pre-processing of large datasets.

## Python API Documentation

To view the documentation related to the Python API, visit https://graphium-docs.datamol.io/.

## Installation for users 
`graphium_cpp` is automatically included in the install of `graphium`. `graphium` can be installed via:
```
mamba install graphium -c conda-forge
```

## Installation for developers 
To install `graphium_cpp` (and `graphium`) locally, refer to these steps:
```
# Install Graphium's dependencies in a new environment named `graphium`
mamba env create -f env.yml -n graphium

# To force the CUDA version to 11.2, or any other version you prefer, use the following command:
# CONDA_OVERRIDE_CUDA=11.2 mamba env create -f env.yml -n graphium

# Activate the mamba environment containing Graphium's dependencies
mamba activate graphium

# Install Graphium in dev mode
pip install --no-deps --no-build-isolation -e .

# On every change to graphium_cpp code, reinstall the package
pip install --no-deps --no-build-isolation -e .
```

Please ensure you reinstall `graphium` when you make changes to the C++ code of `graphium_cpp`. This will automatically recompile the extension and make your changes available for use in `graphium`. 

If you are introducing new files or libraries to `graphium_cpp`, please ensure you update `graphium`'s `setup.py` file accordingly. 