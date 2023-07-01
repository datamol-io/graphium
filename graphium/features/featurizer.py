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

from graphium.features import nmp
from graphium.utils.tensor import one_of_k_encoding
from graphium.features.positional_encoding import get_all_positional_encodings


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


def _mask_nans_inf(mask_nan: Optional[str], array: np.ndarray, array_name: str) -> np.ndarray:
    r"""
    mask the NaNs in the array
    Parameters:
        mask_nan: How to mask the NaNs
        array: The array to mask
        array_name: The name of the array
    Returns:
        The masked array
    """
    if (mask_nan is None) or (array is None):
        return array

    new_array = array
    if issparse(new_array):
        new_array = new_array.data
    nans = ~np.isfinite(new_array)

    # Mask the NaNs
    if nans.any():
        msg = f"There are {np.sum(nans)} NaNs in `{array_name}`"
        if mask_nan == "raise":
            raise ValueError(msg)
        elif mask_nan == "warn":
            logger.warning(msg)
        else:
            new_array[nans] = mask_nan
            if issparse(array):
                array.data = new_array
                new_array = array
    return new_array


def get_mol_atomic_features_onehot(mol: dm.Mol, property_list: List[str]) -> Dict[str, Tensor]:
    r"""
    Get the following set of features for any given atom

    * One-hot representation of the atom
    * One-hot representation of the atom degree
    * One-hot representation of the atom implicit valence
    * One-hot representation of the the atom hybridization
    * Whether the atom is aromatic
    * The atom's formal charge
    * The atom's number of radical electrons

    Additionally, the following features can be set, depending on the value of input Parameters

    * One-hot representation of the number of hydrogen atom in the the current atom neighborhood if `explicit_H` is false
    * One-hot encoding of the atom chirality, and whether such configuration is even possible

    Parameters:

        mol:
            molecule from which to extract the properties

        property_list:
            A list of integer atomic properties to get from the molecule.
            The integer values are converted to a one-hot vector.
            Callables are not supported by this function.

            Accepted properties are:

            - "atomic-number"
            - "degree"
            - "valence", "total-valence"
            - "implicit-valence"
            - "hybridization"
            - "chirality"
            - "phase"
            - "type"
            - "group"
            - "period"

    Returns:
        prop_dict:
            A dictionnary where the element of ``property_list`` are the keys
            and the values are np.ndarray of shape (N, OH). N is the number of atoms
            in ``mol`` and OH the lenght of the one-hot encoding.

    """

    prop_dict = {}

    for prop in property_list:
        prop = prop.lower()
        prop_name = prop

        property_array = []
        for ii, atom in enumerate(mol.GetAtoms()):
            if prop in ["atomic-number"]:
                one_hot = one_of_k_encoding(atom.GetSymbol(), nmp.ATOM_LIST)
            elif prop in ["degree"]:
                one_hot = one_of_k_encoding(atom.GetDegree(), nmp.ATOM_DEGREE_LIST)
            elif prop in ["valence", "total-valence"]:
                prop_name = "valence"
                one_hot = one_of_k_encoding(atom.GetTotalValence(), nmp.VALENCE)
            elif prop in ["implicit-valence"]:
                one_hot = one_of_k_encoding(atom.GetImplicitValence(), nmp.VALENCE)
            elif prop in ["hybridization"]:
                one_hot = one_of_k_encoding(atom.GetHybridization(), nmp.HYBRIDIZATION_LIST)
            elif prop in ["chirality"]:
                try:
                    one_hot = one_of_k_encoding(atom.GetProp("_CIPCode"), nmp.CHIRALITY_LIST)
                    one_hot.append(int(atom.HasProp("_ChiralityPossible")))
                except:
                    one_hot = [0, 0, int(atom.HasProp("_ChiralityPossible"))]
            elif prop in "phase":
                one_hot = one_of_k_encoding(nmp.PHASE[atom.GetAtomicNum() - 1], nmp.PHASE_SET)
            elif prop in "type":
                one_hot = one_of_k_encoding(nmp.TYPE[atom.GetAtomicNum() - 1], nmp.TYPE_SET)
            elif prop in "group":
                one_hot = one_of_k_encoding(nmp.GROUP[atom.GetAtomicNum() - 1], nmp.GROUP_SET)
            elif prop in "period":
                one_hot = one_of_k_encoding(nmp.PERIOD[atom.GetAtomicNum() - 1], nmp.PERIOD_SET)
            else:
                raise ValueError(f"Unsupported property `{prop}`")

            property_array.append(np.asarray(one_hot, dtype=np.float16))

        prop_dict[prop_name] = np.stack(property_array, axis=0)

    return prop_dict


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


def get_mol_atomic_features_float(
    mol: dm.Mol,
    property_list: Union[List[str], List[Callable]],
    offset_carbon: bool = True,
    mask_nan: Union[str, float, type(None)] = "raise",
) -> Dict[str, np.ndarray]:
    r"""
    Get a dictionary of floating-point arrays of atomic properties.
    To ensure all properties are at a similar scale, some of the properties
    are divided by a constant.

    There is also the possibility of offseting by the carbon value using
    the `offset_carbon` parameter.

    Parameters:

        mol:
            molecule from which to extract the properties

        property_list:
            A list of atomic properties to get from the molecule, such as 'atomic-number',
            'mass', 'valence', 'degree', 'electronegativity'.
            Some elements are divided by a factor to avoid feature explosion.

            Accepted properties are:

            - "atomic-number"
            - "mass", "weight"
            - "valence", "total-valence"
            - "implicit-valence"
            - "hybridization"
            - "chirality"
            - "hybridization"
            - "aromatic"
            - "ring", "in-ring"
            - "min-ring"
            - "max-ring"
            - "num-ring"
            - "degree"
            - "radical-electron"
            - "formal-charge"
            - "vdw-radius"
            - "covalent-radius"
            - "electronegativity"
            - "ionization", "first-ionization"
            - "melting-point"
            - "metal"
            - "single-bond"
            - "aromatic-bond"
            - "double-bond"
            - "triple-bond"
            - "is-carbon"
            - "group"
            - "period"

        offset_carbon:
            Whether to subract the Carbon property from the desired atomic property.
            For example, if we want the mass of the Lithium (6.941), the mass of the
            Carbon (12.0107) will be subracted, resulting in a value of -5.0697

        mask_nan:
            Deal with molecules that fail a part of the featurization.
            NaNs can happen when taking the of a noble gas,
            or other properties that are not measured for specific atoms.

            - "raise": Raise an error when there is a nan or inf in the featurization
            - "warn": Raise a warning when there is a nan or inf in the featurization
            - "None": DEFAULT. Don't do anything
            - "Floating value": Replace nans or inf by the specified value

    Returns:

        prop_dict:
            A dictionnary where the element of ``property_list`` are the keys
            and the values are np.ndarray of shape (N,). N is the number of atoms
            in ``mol``.

    """

    periodic_table = Chem.GetPeriodicTable()
    prop_dict = {}
    C = Chem.Atom("C")
    C_num = C.GetAtomicNum()
    offC = bool(offset_carbon)
    atom_list = list(mol.GetAtoms())

    for prop in property_list:
        prop_name = None

        property_array = np.zeros(mol.GetNumAtoms(), dtype=np.float16)
        for ii, atom in enumerate(atom_list):
            val = None
            atomic_num = atom.GetAtomicNum()

            if isinstance(prop, str):
                prop = prop.lower()
                prop_name = prop

                if prop in ["atomic-number"]:
                    val = (atomic_num - (offC * C_num)) / 5
                elif prop in ["mass", "weight"]:
                    prop_name = "mass"
                    val = (atom.GetMass() - (offC * C.GetMass())) / 10
                elif prop in ["valence", "total-valence"]:
                    prop_name = "valence"
                    val = atom.GetTotalValence() - (offC * 4)
                elif prop in ["implicit-valence"]:
                    val = atom.GetImplicitValence()
                elif prop in ["hybridization"]:
                    val = atom.GetHybridization()
                elif prop in ["chirality"]:
                    val = (atom.GetProp("_CIPCode") == "R") if atom.HasProp("_CIPCode") else 2
                elif prop in ["hybridization"]:
                    val = atom.GetHybridization()
                elif prop in ["aromatic"]:
                    val = atom.GetIsAromatic()
                elif prop in ["ring", "in-ring"]:
                    prop_name = "in-ring"
                    val = atom.IsInRing()
                elif prop in ["min-ring"]:
                    ring_info = mol.GetRingInfo()
                    val = ring_info.MinAtomRingSize(atom.GetIdx())
                elif prop in ["max-ring"]:
                    rings = mol.GetRingInfo().AtomRings()
                    val = 0
                    for ring in rings:
                        if atom.GetIdx() in ring:
                            if len(ring) > val:
                                val = len(ring)
                elif prop in ["num-ring"]:
                    ring_info = mol.GetRingInfo()
                    val = ring_info.NumAtomRings(atom.GetIdx())
                elif prop in ["degree"]:
                    val = atom.GetTotalDegree() - (offC * 2)
                elif prop in ["radical-electron"]:
                    val = atom.GetNumRadicalElectrons()
                elif prop in ["formal-charge"]:
                    val = atom.GetFormalCharge()
                elif prop in ["vdw-radius"]:
                    val = periodic_table.GetRvdw(atom.GetAtomicNum()) - offC * periodic_table.GetRvdw(C_num)
                elif prop in ["covalent-radius"]:
                    val = periodic_table.GetRcovalent(atomic_num) - offC * periodic_table.GetRcovalent(C_num)
                elif prop in ["electronegativity"]:
                    val = (
                        nmp.ELECTRONEGATIVITY[atom.GetAtomicNum() - 1]
                        - offC * nmp.ELECTRONEGATIVITY[C_num - 1]
                    )
                elif prop in ["ionization", "first-ionization"]:
                    prop_name = "ionization"
                    val = (nmp.FIRST_IONIZATION[atomic_num - 1] - offC * nmp.FIRST_IONIZATION[C_num - 1]) / 5
                elif prop in ["melting-point"]:
                    val = (nmp.MELTING_POINT[atomic_num - 1] - offC * nmp.MELTING_POINT[C_num - 1]) / 200
                elif prop in ["metal"]:
                    val = nmp.METAL[atomic_num - 1]
                elif prop in "group":
                    val = float(nmp.GROUP[atomic_num - 1]) - offC * float(nmp.GROUP[C_num - 1])
                elif prop in "period":
                    val = float(nmp.PERIOD[atomic_num - 1]) - offC * float(nmp.PERIOD[C_num - 1])
                elif "-bond" in prop:
                    bonds = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
                    if prop in ["single-bond"]:
                        val = len([bond == 1 for bond in bonds])
                    elif prop in ["aromatic-bond"]:
                        val = len([bond == 1.5 for bond in bonds])
                    elif prop in ["double-bond"]:
                        val = len([bond == 2 for bond in bonds])
                    elif prop in ["triple-bond"]:
                        val = len([bond == 3 for bond in bonds])
                    else:
                        raise ValueError(f"{prop} is not a correct bond.")
                    val -= offC * 1
                elif prop in ["is-carbon"]:
                    val = atom.GetAtomicNum() == 6
                    val -= offC * 1
                else:
                    raise ValueError(f"Unsupported property `{prop}`")

            elif callable(prop):
                prop_name = str(prop)
                val = prop(atom)
            else:
                ValueError(f"Elements in `property_list` must be str or callable, provided `{type(prop)}`")

            if val is None:
                raise ValueError("val is undefined.")

            property_array[ii] = val

        if prop_name is None:
            raise ValueError("prop_name is undefined.")

        # Mask the NaNs
        prop_dict[prop_name] = _mask_nans_inf(mask_nan, property_array, "atom featurization")

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


def get_estimated_bond_length(bond: Chem.rdchem.Bond, mol: dm.Mol) -> float:
    r"""
    Estimate the bond length between atoms by looking at the estimated atomic radius
    that depends both on the atom type and the bond type. The resulting bond-length is
    then the sum of the radius.

    Keep in mind that this function only provides an estimate of the bond length and not
    the true one based on a conformer. The vast majority od estimated bond lengths will
    have an error below 5% while some bonds can have an error up to 20%. This function
    is mostly useful when conformer generation fails for some molecules, or for
    increased computation speed.

    Parameters:
        bond: The bond to measure its lenght
        mol: The molecule containing the bond (used to get neighbouring atoms)

    Returns:
        bond_length: The bond length in Angstrom, typically a value around 1-2.

    """

    # Get the atoms connected by the bond
    idx1 = bond.GetBeginAtomIdx()
    idx2 = bond.GetEndAtomIdx()
    atom1 = mol.GetAtomWithIdx(idx1).GetAtomicNum()
    atom2 = mol.GetAtomWithIdx(idx2).GetAtomicNum()
    bond_type = bond.GetBondType()

    # Get single bond atomic radius
    if bond_type == Chem.rdchem.BondType.SINGLE:
        rad1 = [nmp.BOND_RADIUS_SINGLE[atom1 - 1]]
        rad2 = [nmp.BOND_RADIUS_SINGLE[atom2 - 1]]
    # Get double bond atomic radius
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        rad1 = [nmp.BOND_RADIUS_DOUBLE[atom1 - 1]]
        rad2 = [nmp.BOND_RADIUS_DOUBLE[atom2 - 1]]
    # Get triple bond atomic radius
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        rad1 = [nmp.BOND_RADIUS_TRIPLE[atom1 - 1]]
        rad2 = [nmp.BOND_RADIUS_TRIPLE[atom2 - 1]]
    # Get average of single bond and double bond atomic radius
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        rad1 = [nmp.BOND_RADIUS_SINGLE[atom1 - 1], nmp.BOND_RADIUS_DOUBLE[atom1 - 1]]
        rad2 = [nmp.BOND_RADIUS_SINGLE[atom2 - 1], nmp.BOND_RADIUS_DOUBLE[atom2 - 1]]

    # Average the bond lengths, while ignoring nans in case some missing value
    rad1_float = [elem for elem in rad1 if elem is not None]
    rad2_float = [elem for elem in rad2 if elem is not None]

    if len(rad1_float) > 0:
        rad1_float = sum(rad1_float) / len(rad1_float)
    else:
        rad1_float = float(nmp.BOND_RADIUS_SINGLE[atom1 - 1])

    if len(rad2_float) > 0:
        rad2_float = sum(rad2_float) / len(rad2_float)
    else:
        rad2_float = float(nmp.BOND_RADIUS_SINGLE[atom2 - 1])

    bond_length = rad1_float + rad2_float
    return bond_length


def get_mol_edge_features(
    mol: dm.Mol, property_list: List[str], mask_nan: Union[str, float, type(None)] = "raise"
) -> Dict[str, np.ndarray]:
    r"""
    Get the following set of features for any given bond
    See `graphium.features.nmp` for allowed values in one hot encoding

    * One-hot representation of the bond type. Note that you should not kekulize your
        molecules, if you expect this to take aromatic bond into account.
    * Bond stereo type, following CIP classification
    * Whether the bond is conjugated
    * Whether the bond is in a ring

    Parameters:
        mol: rdkit.Chem.Molecule
            the molecule of interest

        property_list:
            A list of edge properties to return for the given molecule.
            Accepted properties are:

            - "bond-type-onehot"
            - "bond-type-float"
            - "stereo"
            - "in-ring"
            - "conjugated"
            - "conformer-bond-length" (might cause problems with complex molecules)
            - "estimated-bond-length"

    Returns:
        prop_dict:
            A dictionnary where the element of ``property_list`` are the keys
            and the values are np.ndarray of shape (N,). N is the number of atoms
            in ``mol``.

    """

    prop_dict = {}

    # Compute features for each bond
    num_bonds = mol.GetNumBonds()
    for prop in property_list:
        property_array = []
        for ii in range(num_bonds):
            prop = prop.lower()
            bond = mol.GetBondWithIdx(ii)

            if prop in ["bond-type-onehot"]:
                encoding = one_of_k_encoding(bond.GetBondType(), nmp.BOND_TYPES)
            elif prop in ["bond-type-float"]:
                encoding = [bond.GetBondTypeAsDouble()]
            elif prop in ["stereo"]:
                encoding = one_of_k_encoding(bond.GetStereo(), nmp.BOND_STEREO)
            elif prop in ["in-ring"]:
                encoding = [bond.IsInRing()]
            elif prop in ["conjugated"]:
                encoding = [bond.GetIsConjugated()]
            elif prop in ["conformer-bond-length"]:
                conf = get_simple_mol_conformer(mol)
                if conf is not None:
                    idx1 = bond.GetBeginAtomIdx()
                    idx2 = bond.GetEndAtomIdx()
                    encoding = [Chem.rdMolTransforms.GetBondLength(conf, idx1, idx2)]
                else:
                    encoding = [0]
            elif prop in ["estimated-bond-length"]:
                encoding = [get_estimated_bond_length(bond, mol)]

            else:
                raise ValueError(f"Unsupported property `{prop}`")

            property_array.append(np.asarray(encoding, dtype=np.float16))

        if num_bonds > 0:
            property_array = np.stack(property_array, axis=0)
            # Mask the NaNs
            prop_dict[prop] = _mask_nans_inf(mask_nan, property_array, "edge property")
        else:
            # Add an empty vector with the right shape
            arr_len = 1
            if prop in ["bond-type-onehot"]:
                arr_len = len(nmp.BOND_TYPES) + 1
            elif prop in ["stereo"]:
                arr_len = len(nmp.BOND_STEREO) + 1

            prop_dict[prop] = np.zeros((0, arr_len))

    return prop_dict


def mol_to_adj_and_features(
    mol: Union[str, dm.Mol],
    atom_property_list_onehot: List[str] = [],
    atom_property_list_float: List[Union[str, Callable]] = [],
    conformer_property_list: List[str] = [],
    edge_property_list: List[str] = [],
    add_self_loop: bool = False,
    explicit_H: bool = False,
    use_bonds_weights: bool = False,
    pos_encoding_as_features: Dict[str, Any] = None,
    dtype: np.dtype = np.float16,
    mask_nan: Union[str, float, type(None)] = "raise",
) -> Union[
    coo_matrix,
    Union[Tensor, None],
    Union[Tensor, None],
    Dict[str, Tensor],
    Union[Tensor, None],
    Dict[str, Tensor],
]:
    r"""
    Transforms a molecule into an adjacency matrix representing the molecular graph
    and a set of atom and bond features.

    It also returns the positional encodings associated to the graph.

    Parameters:

        mol:
            The molecule to be converted

        atom_property_list_onehot:
            List of the properties used to get one-hot encoding of the atom type,
            such as the atom index represented as a one-hot vector.
            See function `get_mol_atomic_features_onehot`

        atom_property_list_float:
            List of the properties used to get floating-point encoding of the atom type,
            such as the atomic mass or electronegativity.
            See function `get_mol_atomic_features_float`

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
            The torch data type used to build the graph

        mask_nan:
            Deal with molecules that fail a part of the featurization.
            NaNs can happen when taking the of a noble gas,
            or other properties that are not measured for specific atoms.

            - "raise": Raise an error when there is a nan or inf in the featurization
            - "warn": Raise a warning when there is a nan or inf in the featurization
            - "None": DEFAULT. Don't do anything
            - "Floating value": Replace nans or inf by the specified value
    Returns:

        adj:
            torch coo sparse adjacency matrix of the molecule

        ndata:
            Concatenated node data of the atoms, based on the properties from
            `atom_property_list_onehot` and `atom_property_list_float`.
            If no properties are given, it returns `None`

        edata:
            Concatenated node edge of the molecule, based on the properties from
            `edge_property_list`.
            If no properties are given, it returns `None`

        pe_dict:
            Dictionary of all positional encodings. Current supported keys:

            - "pos_enc_feats_sign_flip":
                Node positional encoding that requires augmentation via sign-flip.
                For example, eigenvectors of the Laplacian are ambiguous to the
                sign and are returned here.

            - "pos_enc_feats_no_flip":
                Node positional encoding that requires does not use sign-flip.
                For example, distance from centroid are returned here.

            - "rwse":
                Node structural encoding corresponding to the diagonal of the random
                walk matrix

        conf_dict:
            contains the 3d positions of a conformer of the molecule or 0s if none is found

    """

    if isinstance(mol, str):
        mol = dm.to_mol(mol)

    # Add or remove explicit hydrogens
    if explicit_H:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    num_nodes = mol.GetNumAtoms()

    adj = mol_to_adjacency_matrix(
        mol, use_bonds_weights=use_bonds_weights, add_self_loop=add_self_loop, dtype=dtype
    )

    # Get the node features
    atom_features_onehot = get_mol_atomic_features_onehot(mol, atom_property_list_onehot)
    atom_features_float = get_mol_atomic_features_float(mol, atom_property_list_float, mask_nan=mask_nan)
    conf_dict = get_mol_conformer_features(mol, conformer_property_list, mask_nan=mask_nan)
    ndata = list(atom_features_float.values()) + list(atom_features_onehot.values())
    ndata = [d[:, np.newaxis] if d.ndim == 1 else d for d in ndata]

    if len(ndata) > 0:
        ndata = np.concatenate(ndata, axis=1).astype(dtype=dtype)
    else:
        ndata = None

    # Get the edge features
    edge_features = get_mol_edge_features(mol, edge_property_list, mask_nan=mask_nan)
    edata = list(edge_features.values())
    edata = [np.expand_dims(d, axis=1) if d.ndim == 1 else d for d in edata]
    if len(edata) > 0:
        edata = np.concatenate(edata, axis=1).astype(dtype=dtype)
    else:
        edata = None

    # Get all positional encodings
    pe_dict = get_all_positional_encodings(adj, num_nodes, pos_encoding_as_features)

    # Mask the NaNs
    for pe_key, pe_val in pe_dict.items():
        pe_val = np.asarray(pe_val, dtype=dtype)
        pe_dict[pe_key] = _mask_nans_inf(mask_nan, pe_val, pe_key)

    return adj, ndata, edata, pe_dict, conf_dict


def mol_to_adjacency_matrix(
    mol: dm.Mol,
    use_bonds_weights: bool = False,
    add_self_loop: bool = False,
    dtype: np.dtype = np.float32,
) -> coo_matrix:
    r"""
    Convert a molecule to a sparse adjacency matrix, as a torch Tensor.
    Instead of using the Rdkit `GetAdjacencyMatrix()` method, this method
    uses the bond ordering from the molecule object, which is the same as
    the bond ordering in the bond features.

    Warning:
        Do not use `Tensor.coalesce()` on the returned adjacency matrix, as it
        will change the ordering of the bonds.

    Args:
        mol: A molecule in the form of a SMILES string or an RDKit molecule object.

        use_bonds_weights:
            If `True`, the adjacency matrix will contain the bond type as the
            value of the edge. If `False`, the adjacency matrix will contain
            `1` as the value of the edge.

        add_self_loop:
            If `True`, the adjacency matrix will contain a self-loop for each
            node.

        dtype:
            The data type used to build the graph

    Returns:
        adj:
            coo sparse adjacency matrix of the molecule
    """

    # Get the indices for the adjacency matrix, and the bond value
    adj_idx, adj_val = [], []
    for bond in mol.GetBonds():
        adj_idx.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        adj_idx.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        if use_bonds_weights:
            val = nmp.BOND_TYPES[bond.GetBondType()]
        else:
            val = 1
        adj_val.extend([val, val])

    # Convert to torch coo sparse tensor
    if len(adj_val) > 0:  # ensure tensor is not empty
        adj = coo_matrix(
            (torch.as_tensor(adj_val), torch.as_tensor(adj_idx).T.reshape(2, -1)),
            shape=(mol.GetNumAtoms(), mol.GetNumAtoms()),
            dtype=dtype,
        )
    else:
        # Special case for molecules with one atom
        adj = coo_matrix(([], np.array([[], []])), shape=(mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=dtype)

    # Add self loops
    if add_self_loop:
        arange = np.arange(adj.shape[0], dtype=int)
        adj[arange, arange] = 1
    return adj


class GraphDict(dict):
    def __init__(
        self,
        dic: Dict,
    ):
        r"""
        Store the parameters required to initialize a `pyg.data.Data`, but
        as a dictionary to reduce memory consumption.

        Possible keys for the dictionary:

        - adj: A sparse Tensor containing the adjacency matrix

        - ndata: A dictionnary containing different keys and Tensors
            associated to the node features.

        - edata: A dictionnary containing different keys and Tensors
            associated to the edge features.

        - dtype: The dtype for the floating data.

        - mask_nan:
            Deal with molecules that fail a part of the featurization.
            NaNs can happen when taking the of a noble gas,
            or other properties that are not measured for specific atoms.

            - "raise": Raise an error when there is a nan or inf in the featurization
            - "warn": Raise a warning when there is a nan or inf in the featurization
            - "None": DEFAULT. Don't do anything
            - "Floating value": Replace nans or inf by the specified value
        """
        default_dic = {
            "dtype": np.float16,
            "mask_nan": "raise",
        }
        data = dic.pop("data", {})
        # ndata = dic.pop("ndata", {})
        # edata = dic.pop("edata", {})
        # for key in edata.keys():
        #     assert key.startswith("edge_"), f"Edge features must start with 'edge_' but got {key}"
        default_dic.update(dic)
        default_dic.update(data)
        # default_dic.update(ndata)
        # default_dic.update(edata)
        super().__init__(default_dic)

    @property
    def keys(self):
        return list(super().keys())

    @property
    def values(self):
        return list(super().self.values())

    def make_pyg_graph(self, **kwargs) -> Data:
        """
        Convert the current dictionary of parameters, containing an adjacency matrix with node/edge data
        into a `pyg.data.Data` of torch Tensors.

        `**kwargs` can be used to overwrite any parameter from the current dictionary. See `GraphDict.__init__`
        for a list of parameters
        """

        num_nodes = self.adj.shape[0]
        data_dict = {}

        # Convert the numpy and numpy sparse data to torch
        for key, val in self.items():
            if key in ["adj", "dtype", "mask_nan"]:  # Skip the parameters
                continue
            elif isinstance(val, np.ndarray):
                # Convert the data to the specified dtype in torch format
                val = val.astype(self.dtype)
                data_dict[key] = torch.as_tensor(val)
            elif issparse(val):
                data_dict[key] = torch.as_tensor(val.astype(np.float32).todense())
                # `torch.sparse_coo_tensor` is too slow. Slows down the multiprocessing of features by >3x on 32 cores.
                # indices = torch.from_numpy(np.vstack((val.row, val.col)).astype(np.int64))
                # data_dict[key] = torch.sparse_coo_tensor(indices=indices, values=val.data, size=val.shape)
            elif isinstance(val, torch.Tensor):
                data_dict[key] = val
            else:
                pass  # Skip the other parameters

        # Create the PyG graph object `Data`
        edge_index = torch.as_tensor(np.vstack((self.adj.row, self.adj.col)))
        edge_weight = torch.as_tensor(self.adj.data)
        data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes, **data_dict)
        return data

    @property
    def adj(self):
        return self["adj"]

    @property
    def dtype(self):
        return self["dtype"]

    @property
    def mask_nan(self):
        return self["mask_nan"]

    @property
    def num_nodes(self) -> int:
        return self.adj.shape[0]

    @property
    def num_edges(self) -> int:
        if issparse(self.adj):
            return self.adj.nnz
        else:
            return np.count_nonzero(self.adj)  # No division by 2 because edges are counted twice


def mol_to_graph_dict(
    mol: dm.Mol,
    atom_property_list_onehot: List[str] = [],
    atom_property_list_float: List[Union[str, Callable]] = [],
    conformer_property_list: List[str] = [],
    edge_property_list: List[str] = [],
    add_self_loop: bool = False,
    explicit_H: bool = False,
    use_bonds_weights: bool = False,
    pos_encoding_as_features: Dict[str, Any] = None,
    dtype: np.dtype = np.float16,
    on_error: str = "ignore",
    mask_nan: Union[str, float, type(None)] = "raise",
    max_num_atoms: Optional[int] = None,
) -> Union[GraphDict, str]:
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
            See function `get_mol_atomic_features_onehot`

        atom_property_list_float:
            List of the properties used to get floating-point encoding of the atom type,
            such as the atomic mass or electronegativity.
            See function `get_mol_atomic_features_float`

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

    input_mol = mol
    try:
        if isinstance(mol, str):
            mol = dm.to_mol(mol)
        if explicit_H:
            mol = Chem.AddHs(mol)
        else:
            mol = Chem.RemoveHs(mol)

        num_atoms = mol.GetNumAtoms()
        if (max_num_atoms is not None) and (num_atoms > max_num_atoms):
            raise ValueError(f"Maximum number of atoms greater than permitted {num_atoms}>{max_num_atoms}")

        (
            adj,
            ndata,
            edata,
            pe_dict,
            conf_dict,
        ) = mol_to_adj_and_features(
            mol=mol,
            atom_property_list_onehot=atom_property_list_onehot,
            atom_property_list_float=atom_property_list_float,
            conformer_property_list=conformer_property_list,
            edge_property_list=edge_property_list,
            add_self_loop=add_self_loop,
            explicit_H=explicit_H,
            use_bonds_weights=use_bonds_weights,
            pos_encoding_as_features=pos_encoding_as_features,
            mask_nan=mask_nan,
        )
    except Exception as e:
        if on_error.lower() == "raise":
            raise e
        elif on_error.lower() == "warn":
            smiles = input_mol
            if isinstance(smiles, dm.Mol):
                smiles = Chem.MolToSmiles(input_mol)
            msg = str(e) + "\nIgnoring following molecule:" + smiles
            logger.warning(msg)
            return str(e)
        elif on_error.lower() == "ignore":
            return str(e)

    graph_dict = {"adj": adj, "data": {}, "dtype": dtype}

    # Assign the node data
    if ndata is not None:
        graph_dict["data"]["feat"] = ndata

    # Assign the edge data
    if edata is not None:
        if issparse(edata):
            edata = to_dense_array(edata, dtype=dtype)
        hetero_edata = edata.repeat(2, axis=0)
        graph_dict["data"]["edge_feat"] = hetero_edata

    # Put the positional encodings as node features
    # TODO: add support for PE on edges
    for key, pe in pe_dict.items():
        graph_dict["data"][key] = pe

    # put the conformer positions here
    for key, val in conf_dict.items():
        graph_dict["data"][key] = val

    graph_dict = GraphDict(graph_dict)
    return graph_dict


def mol_to_pyggraph(
    mol: dm.Mol,
    atom_property_list_onehot: List[str] = [],
    atom_property_list_float: List[Union[str, Callable]] = [],
    conformer_property_list: List[str] = [],
    edge_property_list: List[str] = [],
    add_self_loop: bool = False,
    explicit_H: bool = False,
    use_bonds_weights: bool = False,
    pos_encoding_as_features: Dict[str, Any] = None,
    dtype: np.dtype = np.float16,
    on_error: str = "ignore",
    mask_nan: Union[str, float, type(None)] = "raise",
    max_num_atoms: Optional[int] = None,
) -> Union[Data, str]:
    r"""
    Transforms a molecule into an adjacency matrix representing the molecular graph
    and a set of atom and bond features.

    Then, the adjacency matrix and node/edge features are used to build a
    `pyg.data.Data` with pytorch Tensors.

    Parameters:

        mol:
            The molecule to be converted

        atom_property_list_onehot:
            List of the properties used to get one-hot encoding of the atom type,
            such as the atom index represented as a one-hot vector.
            See function `get_mol_atomic_features_onehot`

        atom_property_list_float:
            List of the properties used to get floating-point encoding of the atom type,
            such as the atomic mass or electronegativity.
            See function `get_mol_atomic_features_float`

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

            - "raise": Raise an error when there is a nan in the featurization
            - "warn": Raise a warning when there is a nan in the featurization
            - "None": DEFAULT. Don't do anything
            - "Floating value": Replace nans by the specified value

        max_num_atoms:
            Maximum number of atoms for a given molecule. If a molecule with more atoms
            is give, an error is raised, but catpured according to the rules of
            `on_error`.
    Returns:

        graph:
            Pyg graph, with `graph['feat']` corresponding to the concatenated
            node data from `atom_property_list_onehot` and `atom_property_list_float`,
            `graph['edge_feat']` corresponding to the concatenated edge data from `edge_property_list`.
            There are also additional entries for the positional encodings.

    """
    graph_dict = mol_to_graph_dict(
        mol=mol,
        atom_property_list_onehot=atom_property_list_onehot,
        atom_property_list_float=atom_property_list_float,
        conformer_property_list=conformer_property_list,
        edge_property_list=edge_property_list,
        add_self_loop=add_self_loop,
        explicit_H=explicit_H,
        use_bonds_weights=use_bonds_weights,
        pos_encoding_as_features=pos_encoding_as_features,
        dtype=dtype,
        on_error=on_error,
        mask_nan=mask_nan,
        max_num_atoms=max_num_atoms,
    )

    if (graph_dict is not None) and not isinstance(graph_dict, str):
        return graph_dict.make_pyg_graph()
    else:
        return graph_dict


def mol_to_graph_signature(featurizer_args: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get the default arguments of `mol_to_graph_dict` and update it
    with a provided dict of arguments in order to get a fulle signature
    of the featurizer args actually used for the features computation.

    Parameters:
        featurizer_args: A dictionary of featurizer arguments to update
    Returns:
        A dictionary of featurizer arguments
    """

    # Get the signature of `mol_to_graph_dict`
    signature = inspect.signature(mol_to_graph_dict)

    # Filter out empty arguments (without default value)
    parameters = list(filter(lambda param: param.default is not param.empty, signature.parameters.values()))

    # Convert to dict
    parameters = {param.name: param.default for param in parameters}

    # Update the parameters with the supplied ones
    if featurizer_args is not None:
        parameters.update(featurizer_args)

    return parameters
