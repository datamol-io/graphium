""" Utiles for data parsing"""
import os
import warnings

import pandas as pd

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


def read_file(filepath, as_ext=None, **kwargs):
    r"""
    Allow to read different file format and parse them into a MolecularDataFrame.
    Supported formats are:
    * csv (.csv, .smile, .smiles)
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
            raise "`file_type` must be a `str`. Provided: {}".format(file_ext)

    # Read the file according to the right extension
    if file_ext in ["csv", "smile", "smiles", "smi"]:
        data = pd.read_csv(filepath, **kwargs)
    elif file_ext == "txt":
        data = pd.read_table(filepath, **kwargs)
    elif file_ext[0:3] == "xls":
        data = pd.read_excel(filepath, **kwargs)
    elif file_ext == "sdf":
        data = parse_sdf_to_dataframe(filepath, **kwargs)
    elif file_ext == "pkl":
        data = pd.read_pickle(filepath, **kwargs)
    else:
        raise 'File extension "{}" not supported'.format(file_ext)

    return data


def parse_sdf_to_dataframe(sdf_path, as_cxsmiles=True):
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
            (Default=True)

    TODO: Use pandas tools?
    TODO: Parallelize the loop to make it faster

    """

    # read the SDF file
    # locally or from s3
    data = load_sdf(sdf_path)

    # For each molecule in the SDF file, read all the properties and add it to a list of dict.
    # Also count the number of molecules that cannot be read.
    data_list = []
    count_none = 0
    for idx, mol in enumerate(data):
        if (mol is not None) and (ExactMolWt(mol) > 0):
            mol_dict = mol.GetPropsAsDict()
            data_list.append(mol_dict)
            if as_cxsmiles:
                smiles = Chem.rdmolfiles.MolToCXSmiles(mol, canonical=True)
            else:
                smiles = Chem.rdmolfiles.MolToSmiles(mol, canonical=True)
            data_list[-1]["SMILES"] = smiles
        else:
            count_none += 1
            print("Could not read molecule # {}".format(idx))

    # Display a message or warning after the SDF is done parsing
    if count_none == 0:
        print("Successfully read the SDF file without error: {}".format(sdf_path))
    else:
        warnings.warn(
            (
                'Error reading {} molecules from the "{}" file.\
         {} molecules read successfully.'
            ).format(count_none, sdf_path, len(data_list))
        )
    return pd.DataFrame(data_list)


def load_sdf(sdf_path):
    r""" Load sdf file from local or s3 path. """

    if "s3://" not in sdf_path:
        data = Chem.SDMolSupplier(sdf_path)
    else:
        sdf_path_temp = download_from_s3(url=sdf_path)
        data = Chem.SDMolSupplier(sdf_path_temp)

    return data
