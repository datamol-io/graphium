"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals, Graphcore Limited, and NVIDIA Corporation & Affiliates.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals, Graphcore Limited, and NVIDIA Corporation & Affiliates are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


from typing import Union, List, Callable, Dict, Tuple, Any, Optional

import inspect
from loguru import logger
import numpy as np
from scipy.sparse import issparse, coo_matrix
import torch
from torch import Tensor

from torch_geometric.data import Data

from rdkit import Chem
import datamol as dm

import graphium_cpp

def to_dense_array(array: np.ndarray, dtype: str = None) -> np.ndarray:
    r"""
    Assign the node data
    Parameters:
        array: The array to convert to dense
        dtype: The dtype of the array
    Returns:
        The dense array
    """
    if array is not None:
        if issparse(array):
            if array.dtype == np.float16:  # float16 doesn't support `todense`
                array = array.astype(np.float32)
            array = array.todense()

        if dtype is not None:
            array = array.astype(dtype)
    return array


def to_dense_tensor(tensor: Tensor, dtype: str = None) -> Tensor:
    r"""
    Assign the node data
    Parameters:
        array: The array to convert to dense
        dtype: The dtype of the array
    Returns:
        The dense array
    """
    if tensor is not None:
        if tensor.is_sparse:
            tensor = tensor.todense()
        if dtype is not None:
            tensor = tensor.to(dtype)
    return tensor


def get_mol_conformer_features(
    mol: dm.Mol,
    property_list: Union[List[str], List[Callable]],
    mask_nan: Optional[Union[float, str]] = None,
) -> Dict[str, np.ndarray]:
    r"""obtain the conformer features of a molecule
    Parameters:

        mol:
            molecule from which to extract the properties

        property_list:
            A list of conformer property to get from the molecule
            Accepted properties are:
            - "positions_3d"

    Returns:
        prop_dict: a dictionary where the element of ``property_list`` are the keys
    """
    prop_dict = {}
    has_conf = True

    try:
        mol.GetConformer()
    except:
        has_conf = False
    # * currently only accepts "positions_3d", raise errors otherwise
    for prop in property_list:
        if isinstance(prop, str):
            if prop in ["positions_3d"]:  # locating 3d conformer coordinates
                if not has_conf:
                    positions = np.full((mol.GetNumAtoms(), 3), float("nan"), dtype=np.float16)
                else:
                    positions = [[], [], []]
                    for i in range(mol.GetNumAtoms()):
                        pos = mol.GetConformer().GetAtomPosition(i)
                        positions[0].append(pos.x)
                        positions[1].append(pos.y)
                        positions[2].append(pos.z)
                    positions = np.asarray(positions, dtype=np.float16).T
                prop_dict[prop] = positions
            else:
                raise ValueError(
                    str(prop) + " is not currently supported as a conformer property in `property_list`"
                )
        else:
            raise ValueError(f"Elements in `property_list` must be str or callable, provided `{type(prop)}`")

        prop_dict[prop] = _mask_nans_inf(mask_nan, prop_dict[prop], prop)

    return prop_dict



def get_simple_mol_conformer(mol: dm.Mol) -> Union[Chem.rdchem.Conformer, None]:
    r"""
    If the molecule has a conformer, then it will return the conformer at idx `0`.
    Otherwise, it generates a simple molecule conformer using `rdkit.Chem.rdDistGeom.EmbedMolecule`
    and returns it. This is meant to be used in simple functions like `GetBondLength`,
    not in functions requiring complex 3D structure.

    Parameters:

        mol: Rdkit Molecule

    Returns:
        conf: A conformer of the molecule, or `None` if it fails
    """

    val = 0
    if mol.GetNumConformers() == 0:
        val = Chem.rdDistGeom.EmbedMolecule(mol)
    if val == -1:
        val = Chem.rdDistGeom.EmbedMolecule(
            mol,
            enforceChirality=False,
            ignoreSmoothingFailures=True,
            useBasicKnowledge=True,
            useExpTorsionAnglePrefs=True,
            forceTol=0.1,
        )

    if val == -1:
        conf = None
        logger.warn("Couldn't compute conformer for molecule `{}`".format(Chem.MolToSmiles(mol)))
    else:
        conf = mol.GetConformer(0)

    return conf


# These are the integers that correspond with the torch data types in C++
NP_DTYPE_TO_TORCH_INT = {np.float16: 5, np.float32: 6, np.float64: 7}

def mol_to_pyggraph(
    mol: str,
    atom_property_list_onehot: Union[List[str],torch.Tensor] = [],
    atom_property_list_float: Union[List[Union[str, Callable]],torch.Tensor] = [],
    conformer_property_list: List[str] = [],
    edge_property_list: Union[List[str],torch.Tensor] = [],
    add_self_loop: bool = False,
    explicit_H: bool = False,
    use_bonds_weights: bool = False,
    pos_encoding_as_features: Union[Dict[str, Any], Tuple[List[str],torch.Tensor]] = None,
    dtype: np.dtype = np.float16,
    on_error: str = "ignore",
    mask_nan: Union[str, float, type(None)] = "raise",
    max_num_atoms: Optional[int] = None,
) -> Union[Data, str]:
    r"""
    Transforms a molecule into an adjacency matrix representing the molecular graph
    and a set of atom and bond features, and re-organizes them into a dictionary
    that allows to build a `pyg.data.Data` object.

    Compared to `mol_to_pyggraph`, this function does not build the graph directly,
    and is thus faster, less memory heavy, and compatible with other frameworks.

    Parameters:

        mol:
            The molecule to be converted

        atom_property_list_onehot:
            List of the properties used to get one-hot encoding of the atom type,
            such as the atom index represented as a one-hot vector.

        atom_property_list_float:
            List of the properties used to get floating-point encoding of the atom type,
            such as the atomic mass or electronegativity.

        conformer_property_list:
            list of properties used to encode the conformer information, outside of atom properties, currently support "positions_3d"

        edge_property_list:
            List of the properties used to encode the edges, such as the edge type
            and the stereo type.

        add_self_loop:
            Whether to add a value of `1` on the diagonal of the adjacency matrix.

        explicit_H:
            Whether to consider the Hydrogens explicitely. If `False`, the hydrogens
            are implicit.

        use_bonds_weights:
            Whether to use the floating-point value of the bonds in the adjacency matrix,
            such that single bonds are represented by 1, double bonds 2, triple 3, aromatic 1.5

        pos_encoding_as_features: keyword arguments for function `graph_positional_encoder`
            to generate positional encoding for node features.

        dtype:
            The numpy data type used to build the graph

        on_error:
            What to do when the featurization fails. This can change the
            behavior of `mask_nan`.

            - "raise": Raise an error
            - "warn": Raise a warning and return a string of the error
            - "ignore": Ignore the error and return a string of the error

        mask_nan:
            Deal with molecules that fail a part of the featurization.
            NaNs can happen when taking the of a noble gas,
            or other properties that are not measured for specific atoms.

            - "raise": Raise an error when there is a nan or inf in the featurization
            - "warn": Raise a warning when there is a nan or inf in the featurization
            - "None": DEFAULT. Don't do anything
            - "Floating value": Replace nans or inf by the specified value

        max_num_atoms:
            Maximum number of atoms for a given molecule. If a molecule with more atoms
            is give, an error is raised, but catpured according to the rules of
            `on_error`.

    Returns:

        graph_dict:
            A dictionary `GraphDict` containing the keys required to build a graph,
            and which can be used to build a PyG graph. If it fails
            to featurize the molecule, it returns a string with the error.

            - "adj": A sparse int-array containing the adjacency matrix

            - "data": A dictionnary containing different keys and numpy
              arrays associated to the (node, edge & graph) features.

            - "dtype": The numpy dtype for the floating data.
    """

    if not isinstance(mol, str):
        raise ValueError(f"mol_to_pyggraph requires that molecule be received as a string, not type "+str(type(mol)))

    try:
        has_conformer = ('positions_3d' in conformer_property_list)
        pe_index = 4
        if has_conformer:
            pe_index = 5;
        mask_nan_value = 0.0
        if mask_nan is None:
            mask_nan_style_int = 0
        elif mask_nan == "raise" or mask_nan == "warn":
            mask_nan_style_int = 1
        else:
            mask_nan_style_int = 2
            mask_nan_value = float(mask_nan)
        tensors, num_nans, nan_tensor_index = graphium_cpp.featurize_smiles(
            mol,
            atom_property_list_onehot,
            atom_property_list_float,
            'positions_3d' in conformer_property_list,
            edge_property_list,
            pos_encoding_as_features[1],
            True, # duplicate_edges, so that we don't have to duplicate below
            add_self_loop,
            explicit_H,
            use_bonds_weights,
            True, #offset_carbon
            NP_DTYPE_TO_TORCH_INT[dtype],
            mask_nan_style_int,
            mask_nan_value
        )

        if num_nans > 0:
            if nan_tensor_index == 2:
                array_name = "atom featurization"
            elif nan_tensor_index == 3:
                array_name = "edge property"
            elif nan_tensor_index == 4 and has_conformer:
                array_name = 'positions_3d'
            else:
                array_name = pos_encoding_as_features[0][nan_tensor_index - pe_index]
            msg = f"There are {num_nans} NaNs in `{array_name}`"
            if mask_nan == "raise":
                raise ValueError(msg)
            elif mask_nan == "warn":
                logger.warning(msg)

        num_atoms = tensors[2].size(0)
        data_dict = {
            "feat": tensors[2],
            "edge_feat": tensors[3]
            }
        if has_conformer:
            data_dict['positions_3d'] = tensors[4]
        for i in range(len(tensors)-pe_index):
            data_dict[pos_encoding_as_features[0][i]] = tensors[i+pe_index]
        # Create the PyG graph object `Data`
        data = Data(edge_index=tensors[0], edge_weight=tensors[1], num_nodes=num_atoms, **data_dict)
        return data

    except Exception as e:
        if on_error.lower() == "raise":
            raise e
        elif on_error.lower() == "warn":
            msg = str(e) + "\nIgnoring following molecule:" + mol
            logger.warning(msg)
            return str(e)
        elif on_error.lower() == "ignore":
            return str(e)
        else:
            # Invalid on_error value, so default to raising an exception.
            raise e


def mol_to_graph_signature(featurizer_args: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get the default arguments of `mol_to_pyggraph` and update it
    with a provided dict of arguments in order to get a fulle signature
    of the featurizer args actually used for the features computation.

    Parameters:
        featurizer_args: A dictionary of featurizer arguments to update
    Returns:
        A dictionary of featurizer arguments
    """

    # Get the signature of `mol_to_pyggraph`
    signature = inspect.signature(mol_to_pyggraph)

    # Filter out empty arguments (without default value)
    parameters = list(filter(lambda param: param.default is not param.empty, signature.parameters.values()))

    # Convert to dict
    parameters = {param.name: param.default for param in parameters}

    # Update the parameters with the supplied ones
    if featurizer_args is not None:
        parameters.update(featurizer_args)

    return parameters
