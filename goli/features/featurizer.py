from typing import Union, List, Callable, Dict, Tuple, Any, Optional

import inspect
from loguru import logger

import numpy as np
from scipy.sparse import csr_matrix, issparse, coo_matrix
import torch
import dgl

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from datamol import to_mol

from goli.features import nmp
from goli.utils.tensor import one_of_k_encoding
from goli.features.positional_encoding import get_all_positional_encoding


def _to_dense_array(array, dtype=None):
    # Assign the node data
    if array is not None:
        if issparse(array):
            if array.dtype == np.float16:  # float16 doesn't support `todense`
                array = array.astype(np.float32)
            array = array.todense()

        if dtype is not None:
            array = array.astype(dtype)
    return array


def _mask_nans_inf(mask_nan, array, array_name):

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


def get_mol_atomic_features_onehot(mol: Chem.rdchem.Mol, property_list: List[str]) -> Dict[str, np.ndarray]:
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
            else:
                raise ValueError(f"Unsupported property `{prop}`")

            property_array.append(np.asarray(one_hot, dtype=np.float16))

        prop_dict[prop_name] = np.stack(property_array, axis=0)

    return prop_dict


def get_mol_atomic_features_float(
    mol: Chem.rdchem.Mol,
    property_list: Union[List[str], List[Callable]],
    offset_carbon: bool = True,
    mask_nan: Union[str, float, type(None)] = "raise",
) -> Dict[str, np.ndarray]:
    """
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
    offC = bool(offset_carbon)

    for prop in property_list:

        prop_name = None

        property_array = np.zeros(mol.GetNumAtoms(), dtype=np.float16)
        for ii, atom in enumerate(mol.GetAtoms()):

            val = None

            if isinstance(prop, str):

                prop = prop.lower()
                prop_name = prop

                if prop in ["atomic-number"]:
                    val = (atom.GetAtomicNum() - (offC * C.GetAtomicNum())) / 5
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
                elif prop in ["degree"]:
                    val = atom.GetTotalDegree() - (offC * 2)
                elif prop in ["radical-electron"]:
                    val = atom.GetNumRadicalElectrons()
                elif prop in ["formal-charge"]:
                    val = atom.GetFormalCharge()
                elif prop in ["vdw-radius"]:
                    val = periodic_table.GetRvdw(atom.GetAtomicNum()) - offC * periodic_table.GetRvdw(
                        C.GetAtomicNum()
                    )
                elif prop in ["covalent-radius"]:
                    val = periodic_table.GetRcovalent(
                        atom.GetAtomicNum()
                    ) - offC * periodic_table.GetRcovalent(C.GetAtomicNum())
                elif prop in ["electronegativity"]:
                    val = (
                        nmp.PERIODIC_TABLE["Electronegativity"][atom.GetAtomicNum()]
                        - offC * nmp.PERIODIC_TABLE["Electronegativity"][C.GetAtomicNum()]
                    )
                elif prop in ["ionization", "first-ionization"]:
                    prop_name = "ionization"
                    val = (
                        nmp.PERIODIC_TABLE["FirstIonization"][atom.GetAtomicNum()]
                        - offC * nmp.PERIODIC_TABLE["FirstIonization"][C.GetAtomicNum()]
                    ) / 5
                elif prop in ["melting-point"]:
                    val = (
                        nmp.PERIODIC_TABLE["MeltingPoint"][atom.GetAtomicNum()]
                        - offC * nmp.PERIODIC_TABLE["MeltingPoint"][C.GetAtomicNum()]
                    ) / 200
                elif prop in ["metal"]:
                    if nmp.PERIODIC_TABLE["Metal"][atom.GetAtomicNum()] == "yes":
                        val = 2
                    elif nmp.PERIODIC_TABLE["Metalloid"][atom.GetAtomicNum()] == "yes":
                        val = 1
                    else:
                        val = 0
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


def get_simple_mol_conformer(mol: Chem.rdchem.Mol) -> Union[Chem.rdchem.Conformer, None]:
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


def get_estimated_bond_length(bond: Chem.rdchem.Bond, mol: Chem.rdchem.Mol) -> float:
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

    # Small function to convert strings to floats
    def float_or_nan(string):
        try:
            val = float(string)
        except:
            val = float("nan")
        return val

    # Get the atoms connected by the bond
    idx1 = bond.GetBeginAtomIdx()
    idx2 = bond.GetEndAtomIdx()
    atom1 = mol.GetAtomWithIdx(idx1).GetAtomicNum()
    atom2 = mol.GetAtomWithIdx(idx2).GetAtomicNum()
    bond_type = bond.GetBondType()

    # Get single bond atomic radius
    if bond_type == Chem.rdchem.BondType.SINGLE:
        rad1 = [nmp.PERIODIC_TABLE["SingleBondRadius"][atom1]]
        rad2 = [nmp.PERIODIC_TABLE["SingleBondRadius"][atom2]]
    # Get double bond atomic radius
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        rad1 = [nmp.PERIODIC_TABLE["DoubleBondRadius"][atom1]]
        rad2 = [nmp.PERIODIC_TABLE["DoubleBondRadius"][atom2]]
    # Get triple bond atomic radius
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        rad1 = [nmp.PERIODIC_TABLE["TripleBondRadius"][atom1]]
        rad2 = [nmp.PERIODIC_TABLE["TripleBondRadius"][atom2]]
    # Get average of single bond and double bond atomic radius
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        rad1 = [nmp.PERIODIC_TABLE["SingleBondRadius"][atom1], nmp.PERIODIC_TABLE["DoubleBondRadius"][atom1]]
        rad2 = [nmp.PERIODIC_TABLE["SingleBondRadius"][atom2], nmp.PERIODIC_TABLE["DoubleBondRadius"][atom2]]

    # Average the bond lengths, while ignoring nans in case some missing value
    rad1_float = np.nanmean(np.array([float_or_nan(elem) for elem in rad1]))
    rad2_float = np.nanmean(np.array([float_or_nan(elem) for elem in rad2]))

    # If the bond radius is still nan (this shouldn't happen), take the single bond radius
    if np.isnan(rad1_float):
        rad1_float = float_or_nan(nmp.PERIODIC_TABLE["SingleBondRadius"][atom1])
    if np.isnan(rad2_float):
        rad2_float = float_or_nan(nmp.PERIODIC_TABLE["SingleBondRadius"][atom2])

    bond_length = rad1_float + rad2_float

    return bond_length


def get_mol_edge_features(
    mol: Chem.rdchem.Mol, property_list: List[str], mask_nan: Union[str, float, type(None)] = "raise"
):
    r"""
    Get the following set of features for any given bond
    See `goli.features.nmp` for allowed values in one hot encoding

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
            prop_dict[prop] = np.array([])

    return prop_dict


def mol_to_adj_and_features(
    mol: Union[str, Chem.rdchem.Mol],
    atom_property_list_onehot: List[str] = [],
    atom_property_list_float: List[Union[str, Callable]] = [],
    edge_property_list: List[str] = [],
    add_self_loop: bool = False,
    explicit_H: bool = False,
    use_bonds_weights: bool = False,
    pos_encoding_as_features: Dict[str, Any] = None,
    pos_encoding_as_directions: Dict[str, Any] = None,
    dtype: np.dtype = np.float16,
    mask_nan: Union[str, float, type(None)] = "raise",
) -> Union[csr_matrix, Union[np.ndarray, None], Union[np.ndarray, None]]:
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

        pos_encoding_as_directions: keyword arguments for function `graph_positional_encoder`
            to generate positional encoding for directional features.

        dtype:
            The numpy data type used to build the graph

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
            Scipy sparse adjacency matrix of the molecule

        ndata:
            Concatenated node data of the atoms, based on the properties from
            `atom_property_list_onehot` and `atom_property_list_float`.
            If no properties are given, it returns `None`

        edata:
            Concatenated node edge of the molecule, based on the properties from
            `edge_property_list`.
            If no properties are given, it returns `None`

        pos_enc_feats_sign_flip:
            Node positional encoding that requires augmentation via sign-flip.
            For example, eigenvectors of the Laplacian are ambiguous to the
            sign and are returned here.

        pos_enc_feats_no_flip:
            Node positional encoding that requires does not use sign-flip.
            For example, distance from centroid are returned here.

        pos_enc_dir:
            Node positional encoding used to define directions. This can thus
            be used for relative position inference.
            This is used, for example, by `DGNConvolutionalLayer` to define
            the direction of the messages.

    """

    if isinstance(mol, str):
        mol = to_mol(mol)

    # Add or remove explicit hydrogens
    if explicit_H:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    # Get the adjacency matrix
    adj = GetAdjacencyMatrix(mol, useBO=use_bonds_weights, force=True)
    if add_self_loop:
        adj = adj + np.eye(adj.shape[0])
    adj = csr_matrix(adj, dtype=np.int8)

    # Get the node features
    atom_features_onehot = get_mol_atomic_features_onehot(mol, atom_property_list_onehot)
    atom_features_float = get_mol_atomic_features_float(mol, atom_property_list_float, mask_nan=mask_nan)
    ndata = list(atom_features_float.values()) + list(atom_features_onehot.values())
    ndata = [np.expand_dims(d, axis=1) if d.ndim == 1 else d for d in ndata]

    if len(ndata) > 0:
        ndata = np.concatenate(ndata, axis=1).astype(dtype)
        ndata = coo_matrix(ndata)
    else:
        ndata = None

    # Get the edge features
    edge_features = get_mol_edge_features(mol, edge_property_list, mask_nan=mask_nan)
    edata = list(edge_features.values())
    edata = [np.expand_dims(d, axis=1) if d.ndim == 1 else d for d in edata]
    if len(edata) > 0:
        edata = np.concatenate(edata, axis=1).astype(dtype)
        edata = coo_matrix(edata)
    else:
        edata = None

    pos_enc_feats_sign_flip, pos_enc_feats_no_flip, pos_enc_dir = get_all_positional_encoding(
        adj, pos_encoding_as_features, pos_encoding_as_directions
    )

    # Mask the NaNs
    pos_enc_feats_sign_flip = _mask_nans_inf(mask_nan, pos_enc_feats_sign_flip, "pos_enc_feats_sign_flip")
    pos_enc_feats_no_flip = _mask_nans_inf(mask_nan, pos_enc_feats_no_flip, "pos_enc_feats_no_flip")
    pos_enc_dir = _mask_nans_inf(mask_nan, pos_enc_dir, "pos_enc_dir")

    return adj, ndata, edata, pos_enc_feats_sign_flip, pos_enc_feats_no_flip, pos_enc_dir


def mol_to_dglgraph_dict(
    mol: Chem.rdchem.Mol,
    atom_property_list_onehot: List[str] = [],
    atom_property_list_float: List[Union[str, Callable]] = [],
    edge_property_list: List[str] = [],
    add_self_loop: bool = False,
    explicit_H: bool = False,
    use_bonds_weights: bool = False,
    pos_encoding_as_features: Dict[str, Any] = None,
    pos_encoding_as_directions: Dict[str, Any] = None,
    dtype: np.dtype = np.float16,
    on_error: str = "ignore",
    mask_nan: Union[str, float, type(None)] = "raise",
) -> Dict:
    r"""
    Transforms a molecule into an adjacency matrix representing the molecular graph
    and a set of atom and bond features, and re-organizes them into a dictionary
    that allows to build a `DGLGraph` object.

    Compared to `mol_to_dglgraph`, this function does not build the graph directly,
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

        pos_encoding_as_directions: keyword arguments for function `graph_positional_encoder`
            to generate positional encoding for directional features.

        dtype:
            The numpy data type used to build the graph

        on_error:
            What to do when the featurization fails. This can change the
            behavior of `mask_nan`.

            - "raise": Raise an error
            - "warn": Raise a warning and return None
            - "ignore": Ignore the error and return None

        mask_nan:
            Deal with molecules that fail a part of the featurization.
            NaNs can happen when taking the of a noble gas,
            or other properties that are not measured for specific atoms.

            - "raise": Raise an error when there is a nan or inf in the featurization
            - "warn": Raise a warning when there is a nan or inf in the featurization
            - "None": DEFAULT. Don't do anything
            - "Floating value": Replace nans or inf by the specified value

    Returns:

        graph:
            A dictionary containing the keys required to build a DGLGraph fromt the function
            `mol_to_dglgraph_dict`. The keys are:

            - "adj": A sparse int-array containing the adjacency matrix

            - "ndata": A dictionnary containing different keys and numpy
              arrays associated to the node features.

            - "edata": A dictionnary containing different keys and numpy
              arrays associated to the edge features.

            - "dtype": The numpy dtype for the floating data.
    """

    input_mol = mol

    try:

        if isinstance(mol, str):
            mol = to_mol(mol)
        if explicit_H:
            mol = Chem.AddHs(mol)
        else:
            mol = Chem.RemoveHs(mol)

        # Get the adjacency, node features and edge features
        (
            adj,
            ndata,
            edata,
            pos_enc_feats_sign_flip,
            pos_enc_feats_no_flip,
            pos_enc_dir,
        ) = mol_to_adj_and_features(
            mol=mol,
            atom_property_list_onehot=atom_property_list_onehot,
            atom_property_list_float=atom_property_list_float,
            edge_property_list=edge_property_list,
            add_self_loop=add_self_loop,
            explicit_H=explicit_H,
            use_bonds_weights=use_bonds_weights,
            pos_encoding_as_features=pos_encoding_as_features,
            pos_encoding_as_directions=pos_encoding_as_directions,
            mask_nan=mask_nan,
        )
    except Exception as e:
        if on_error.lower() == "raise":
            raise e
        elif on_error.lower() == "warn":
            smiles = input_mol
            if isinstance(smiles, Chem.rdchem.Mol):
                smiles = Chem.MolToSmiles(input_mol)

            msg = str(e) + "\nIgnoring following molecule:" + smiles
            logger.warning(msg)
            return None
        elif on_error.lower() == "ignore":
            return None

    dgl_dict = {"adj": adj, "edata": {}, "ndata": {}, "dtype": dtype}

    # Assign the node data
    if ndata is not None:
        dgl_dict["ndata"]["feat"] = ndata

    # Assign the edge data. Due to DGL only supporting Hetero-graphs, we
    # need to duplicate each edge information for its 2 entries
    if edata is not None:
        edata = _to_dense_array(edata, dtype=dtype)
        src_ids, dst_ids = np.argwhere(adj).transpose()
        hetero_edata = np.zeros(shape=(edata.shape[0] * 2, edata.shape[1]), dtype=edata.dtype)
        for ii in range(mol.GetNumBonds()):
            bond = mol.GetBondWithIdx(ii)
            src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            id1 = np.where((src == src_ids) & (dst == dst_ids))[0]
            id2 = np.where((dst == src_ids) & (src == dst_ids))[0]
            hetero_edata[id1, :] = edata[ii, :]
            hetero_edata[id2, :] = edata[ii, :]

        dgl_dict["edata"]["feat"] = coo_matrix(hetero_edata)

    # Add sign-flip positional encoding
    if pos_enc_feats_sign_flip is not None:
        dgl_dict["ndata"]["pos_enc_feats_sign_flip"] = pos_enc_feats_sign_flip

    # Add non-sign-flip positional encoding
    if pos_enc_feats_no_flip is not None:
        dgl_dict["ndata"]["pos_enc_feats_no_flip"] = pos_enc_feats_no_flip

    # Add positional encoding for directional use
    if pos_enc_dir is not None:
        dgl_dict["ndata"]["pos_dir"] = pos_enc_dir

    return dgl_dict


def mol_to_dglgraph(
    mol: Chem.rdchem.Mol,
    atom_property_list_onehot: List[str] = [],
    atom_property_list_float: List[Union[str, Callable]] = [],
    edge_property_list: List[str] = [],
    add_self_loop: bool = False,
    explicit_H: bool = False,
    use_bonds_weights: bool = False,
    pos_encoding_as_features: Dict[str, Any] = None,
    pos_encoding_as_directions: Dict[str, Any] = None,
    dtype: np.dtype = np.float16,
    on_error: str = "ignore",
    mask_nan: Union[str, float, type(None)] = "raise",
) -> dgl.DGLGraph:
    r"""
    Transforms a molecule into an adjacency matrix representing the molecular graph
    and a set of atom and bond features.

    Then, the adjacency matrix and node/edge features are used to build a
    `DGLGraph` with pytorch Tensors.

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

        pos_encoding_as_directions: keyword arguments for function `graph_positional_encoder`
            to generate positional encoding for directional features.

        dtype:
            The numpy data type used to build the graph

        on_error:
            What to do when the featurization fails. This can change the
            behavior of `mask_nan`.

            - "raise": Raise an error
            - "warn": Raise a warning and return None
            - "ignore": Ignore the error and return None

        mask_nan:
            Deal with molecules that fail a part of the featurization.
            NaNs can happen when taking the of a noble gas,
            or other properties that are not measured for specific atoms.

            - "raise": Raise an error when there is a nan in the featurization
            - "warn": Raise a warning when there is a nan in the featurization
            - "None": DEFAULT. Don't do anything
            - "Floating value": Replace nans by the specified value

    Returns:

        graph:
            DGL graph, with `graph.ndata['feat']` corresponding to the concatenated
            node data from `atom_property_list_onehot` and `atom_property_list_float`,
            `graph.edata['feat']` corresponding to the concatenated edge data from `edge_property_list`.
            There are also additional entries for the positional encodings.

    """

    dgl_dict = mol_to_dglgraph_dict(
        mol=mol,
        atom_property_list_onehot=atom_property_list_onehot,
        atom_property_list_float=atom_property_list_float,
        edge_property_list=edge_property_list,
        add_self_loop=add_self_loop,
        explicit_H=explicit_H,
        use_bonds_weights=use_bonds_weights,
        pos_encoding_as_features=pos_encoding_as_features,
        pos_encoding_as_directions=pos_encoding_as_directions,
        dtype=dtype,
        on_error=on_error,
        mask_nan=mask_nan,
    )

    if dgl_dict is not None:
        return dgl_dict_to_graph(**dgl_dict)

    return None


def dgl_dict_to_graph(
    adj,
    ndata: Dict,
    edata: Dict,
    dtype: np.dtype = None,
    mask_nan: Union[str, float, type(None)] = "raise",
) -> dgl.DGLGraph:
    """
    Convert an adjacency matrix with node/edge data into a `DGLGraph`
    of torch Tensors.

    Parameters:

        adj: A numpy array containing the adjacency matrix

        ndata: A dictionnary containing different keys and numpy
            arrays associated to the node features `DGLGraph.ndata`.

        edata: A dictionnary containing different keys and numpy
            arrays associated to the edge features `DGLGraph.edata`.

        dtype: The numpy dtype for the floating data. The arrays
            will be converted to `torch.Tensor` when building the graph.

        mask_nan:
            Deal with molecules that fail a part of the featurization.
            NaNs can happen when taking the of a noble gas,
            or other properties that are not measured for specific atoms.

            - "raise": Raise an error when there is a nan or inf in the featurization
            - "warn": Raise a warning when there is a nan or inf in the featurization
            - "None": DEFAULT. Don't do anything
            - "Floating value": Replace nans or inf by the specified value

    """

    # Transform the matrix and data into a DGLGraph object
    graph = dgl.from_scipy(adj, idtype=torch.int32)

    if ndata is not None:
        for key, val in ndata.items():
            graph.ndata[key] = torch.as_tensor(
                _mask_nans_inf(mask_nan=mask_nan, array=_to_dense_array(val, dtype=dtype), array_name="ndata")
            )

    if edata is not None:
        for key, val in edata.items():
            graph.edata[key] = torch.as_tensor(
                _mask_nans_inf(mask_nan=mask_nan, array=_to_dense_array(val, dtype=dtype), array_name="edata")
            )

    return graph


def mol_to_dglgraph_signature(featurizer_args: Dict[str, Any] = None):
    """Get the default arguments of `mol_to_dglgraph_dict` and update it
    with a provided dict of arguments in order to get a fulle signature
    of the featurizer args actually used for the features computation.
    """

    # Get the signature of `mol_to_dglgraph_dict`
    signature = inspect.signature(mol_to_dglgraph_dict)

    # Filter out empty arguments (without default value)
    parameters = list(filter(lambda param: param.default is not param.empty, signature.parameters.values()))

    # Convert to dict
    parameters = {param.name: param.default for param in parameters}

    # Update the parameters with the supplied ones
    if featurizer_args is not None:
        parameters.update(featurizer_args)

    return parameters
