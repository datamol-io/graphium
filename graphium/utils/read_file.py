""" Utiles for data parsing"""
import os
import warnings
import numpy as np
import pandas as pd
import datamol as dm
from functools import partial
from copy import copy
import fsspec

from loguru import logger
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from graphium.utils.tensor import parse_valid_args, arg_in_func


def read_file(filepath, as_ext=None, **kwargs):
    r"""
    Allow to read different file format and parse them into a MolecularDataFrame.
    Supported formats are:
    * csv (.csv, .smile, .smiles, .tsv)
    * txt (.txt)
    * xls (.xls, .xlsx, .xlsm, .xls*)
    * sdf (.sdf)
    * pkl (.pkl)

    Arguments
    -----------

        filepath: str
            The full path and name of the file to read.
            It also supports the s3 url path.
        as_ext: str, Optional
            The file extension used to read the file. If None, the extension is deduced
            from the extension of the file. Otherwise, no matter the file extension,
            the file will be read according to the specified ``as_ext``.
            (Default=None)
        **kwargs: All the optional parameters required for the desired file reader.

    TODO: unit test to make sure it works well with all extensions

    Returns
    ---------
        df: pandas.DataFrame
            The ``pandas.DataFrame`` containing the parsed data

    """

    # Get the file extension
    if as_ext is None:
        file_ext = os.path.splitext(filepath)[-1].lower()[1:]
    else:
        file_ext = as_ext
        if not isinstance(file_ext, str):
            raise TypeError("`file_type` must be a `str`. Provided: {}".format(file_ext))

    open_mode = "r"

    # Read the file according to the right extension
    if file_ext in ["csv", "smile", "smiles", "smi", "tsv"]:
        file_reader = pd.read_csv
    elif file_ext == "txt":
        file_reader = pd.read_table
    elif file_ext[0:3] == "xls":
        open_mode = "rb"
        file_reader = partial(pd.read_excel, engine="openpyxl")
    elif file_ext == "sdf":
        file_reader = parse_sdf_to_dataframe
    elif file_ext == "pkl":
        open_mode = "rb"
        file_reader = pd.read_pickle
    else:
        raise 'File extension "{}" not supported'.format(file_ext)

    kwargs = parse_valid_args(fn=file_reader, param_dict=kwargs)

    if file_ext[0:3] not in ["sdf", "xls"]:
        with file_opener(filepath, open_mode) as file_in:
            data = file_reader(file_in, **kwargs)
    else:
        data = file_reader(filepath, **kwargs)
    return data


def parse_sdf_to_dataframe(sdf_path, as_cxsmiles=True, skiprows=None):
    r"""
    Allows to read an SDF file containing molecular informations, convert
    it to a pandas DataFrame and convert the molecules to SMILES. It also
    lists a warning of all the molecules that couldn't be read.

    Arguments
    -----------

        sdf_path: str
            The full path and name of the sdf file to read
        as_cxsmiles: bool, optional
            Whether to use the CXSMILES notation, which preserves atomic coordinates,
            stereocenters, and much more.
            See `https://dl.chemaxon.com/marvin-archive/latest/help/formats/cxsmiles-doc.html`
            (Default = True)
        skiprows: int, list
            The rows to skip from dataset. The enumerate index starts from 1 insted of 0.
            (Default = None)

    """

    # read the SDF file
    # locally or from s3
    data = dm.read_sdf(sdf_path)

    # For each molecule in the SDF file, read all the properties and add it to a list of dict.
    # Also count the number of molecules that cannot be read.
    data_list = []
    count_none = 0
    if skiprows is not None:
        if isinstance(skiprows, int):
            skiprows = range(0, skiprows - 1)
        skiprows = np.array(skiprows) - 1

    for idx, mol in enumerate(data):
        if (skiprows is not None) and (idx in skiprows):
            continue

        if (mol is not None) and (ExactMolWt(mol) > 0):
            mol_dict = mol.GetPropsAsDict()
            data_list.append(mol_dict)
            if as_cxsmiles:
                smiles = Chem.rdmolfiles.MolToCXSmiles(mol, canonical=True)
            else:
                smiles = dm.to_smiles(mol, canonical=True)
            data_list[-1]["SMILES"] = smiles
        else:
            count_none += 1
            logger.info(f"Could not read molecule # {idx}")

    # Display a message or warning after the SDF is done parsing
    if count_none == 0:
        logger.info("Successfully read the SDF file without error: {}".format(sdf_path))
    else:
        warnings.warn(
            (
                'Error reading {} molecules from the "{}" file.\
         {} molecules read successfully.'
            ).format(count_none, sdf_path, len(data_list))
        )
    return pd.DataFrame(data_list)


def file_opener(filename, mode="r"):
    """File reader stream"""
    filename = str(filename)
    if "w" in mode:
        filename = "simplecache::" + filename
    if filename.endswith(".gz"):
        instream = fsspec.open(filename, mode=mode, compression="gzip")
    else:
        instream = fsspec.open(filename, mode=mode)
    return instream
