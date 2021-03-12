from typing import Union, List, Callable

import os
import numpy as np
from scipy.sparse import csr_matrix
import datamol as dm
import dgl
import torch

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from mordred import Calculator, descriptors

from goli.mol_utils import nmp
from goli.commons.utils import one_of_k_encoding


def get_prop_or_none(prop, n, *args, **kwargs):
    r"""
    return properties. If error, return list of `None` with lenght `n`.
    """
    try:
        return prop(*args, **kwargs)
    except RuntimeError:
        return [None] * n


def get_props_from_mol(mol: Union[Chem.rdBase.Mol, str], properties: Union[List[str], str] = "autocorr3d"):
    r"""
    Function to get a given set of desired properties from a molecule,
    and output a property list.

    Parameters:
        mol: The molecule from which to compute the properties.
        properties:
            The list of properties to compute for each molecule. It can be the following:
            - 'descriptors'
            - 'autocorr3d'
            - 'rdf'
            - 'morse'
            - 'whim'
            - 'all'

    Returns:
        props: np.array(float)
            The array of properties for the desired molecule
        classes_start_idx: list(int)
            The list of index specifying the start of each new class of
            descriptor or property. For example, if props has 20 elements,
            the first 5 are rotatable bonds, the next 8 are morse, and
            the rest are whim, then ``classes_start_idx = [0, 5, 13]``.
            This will mainly be useful to normalize the features of
            each class.
        classes_names: list(str)
            The name of the classes associated to each starting index.
            Will be usefull to understand what property is the network learning.

    """

    if isinstance(mol, str):
        mol = dm.to_mol(mol)

    if isinstance(properties, str):
        properties = [properties]

    properties = [p.lower() for p in properties]

    # Initialize arrays
    props = []  # Property vector for the features
    classes_start_idx = []  # The starting index for each property class
    classes_names = []

    # Generate a 3D structure for the molecule
    mol = Chem.AddHs(mol)  # type: ignore

    if ("descriptors" in properties) or ("all" in properties):
        # Calculate the descriptors of the molecule
        for desc in descriptors.all:
            classes_names.append(desc.__name__.replace("mordred.", ""))
            classes_start_idx.append(len(props))
            calc = Calculator(desc, ignore_3D=True)
            props.extend(calc(mol))

    if ("autocorr3d" in properties) or ("all" in properties):
        # Some kind of 3D description of the molecule
        classes_names.append("autocorr3d")
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcAUTOCORR3D, 80, mol))

    if ("rdf" in properties) or ("all" in properties):
        # The radial distribution function (better than the inertia)
        # https://en.wikipedia.org/wiki/Radial_distribution_function
        classes_names.append("rdf")
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcRDF, 210, mol))

    if ("morse" in properties) or ("all" in properties):
        # Molecule Representation of Structures based on Electron diffraction descriptors
        classes_names.append("morse")
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcMORSE, 224, mol))

    if ("whim" in properties) or ("all" in properties):
        # WHIM descriptors are 3D structural descriptors obtained from the
        # (x,y,z)‐atomic coordinates of a molecular conformation of a chemical,
        # and are used successfully in QSAR modelling.
        classes_names.append("whim")
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcWHIM, 114, mol))

    return np.array(props), classes_start_idx, classes_names


def get_mol_atomic_features_onehot(
        mol: Mol, property_list: List[str]) -> Dict[str, np.ndarray]:
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

    Parameters

        mol:
            molecule from which to extract the properties

        property_list:
            A list of integer atomic properties to get from the molecule, such as 'atomic-number',
            'atomic-number', 'valence', 'degree', 'radical-electron'.
            The integer values are converted to a one-hot vector.
            Callables are not supported by this function.

    Returns:
        prop_dict:
            A dictionnary where the element of ``property_list`` are the keys
            and the values are np.ndarray of shape (N, OH). N is the number of atoms
            in ``mol`` and OH the lenght of the one-hot encoding.

    """

    prop_dict = {}

    for prop in property_list:
        property_array = []
        for ii, atom in enumerate(mol.GetAtoms()):

            if prop in ['atomic-number']:
                one_hot = one_of_k_encoding(atom.GetSymbol(), nmp.ATOM_LIST)
            elif prop in ['degree']:
                one_hot = one_of_k_encoding(atom.GetDegree(), nmp.ATOM_DEGREE_LIST)
            elif prop in ['valence', 'total-valence']:
                prop_name = 'valence'
                one_hot = one_of_k_encoding(atom.GetTotalValence(), nmp.VALENCE)
            elif prop in ['implicit-valence']:
                one_hot = one_of_k_encoding(atom.GetImplicitValence(), nmp.VALENCE)
            elif prop in ['hybridization']:
                one_hot = one_of_k_encoding(atom.GetHybridization(), nmp.HYBRIDIZATION_LIST)
            elif prop in ['chirality']:
                try:
                    one_hot = one_of_k_encoding(atom.GetProp("_CIPCode"), nmp.CHIRALITY_LIST)
                    one_hot.append(int(atom.HasProp("_ChiralityPossible")))
                except:
                    one_hot = [0, 0, int(atom.HasProp("_ChiralityPossible"))]
            else:
                raise ValueError(f'Unsupported property `{prop}`')

            property_array.append(np.asarray(one_hot, dtype=np.float32))

        prop_dict[prop_name] = np.stack(property_array, axis=0)

    return prop_dict



def get_mol_atomic_features_float(
            mol: Mol, property_list: List[Union[str, Callable]], 
            offset_carbon: bool=True) -> Dict[str, np.ndarray]:
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

        offset_carbon:
            Whether to subract the Carbon property from the desired atomic property.
            For example, if we want the mass of the Lithium (6.941), the mass of the
            Carbon (12.0107) will be subracted, resulting in a value of -5.0697

    Returns:

        prop_dict:
            A dictionnary where the element of ``property_list`` are the keys
            and the values are np.ndarray of shape (N,). N is the number of atoms
            in ``mol``.

    """

    periodic_table = Chem.GetPeriodicTable()
    prop_dict = {}
    C = Chem.Atom('C')
    offC = bool(offset_carbon)

    for prop in property_list:
        property_array = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for ii, atom in enumerate(mol.GetAtoms()):
            if isinstance(prop, str):
                prop = prop.lower()
                prop_name = prop
                if prop in ['atomic-number']:
                    val = (atom.GetAtomicNum() - (offC * C.GetAtomicNum())) / 5
                elif prop in ['mass', 'weight']:
                    prop_name = 'mass'
                    val = (atom.GetMass() - (offC * C.GetMass())) / 10
                elif prop in ['valence', 'total-valence']:
                    prop_name = 'valence'
                    val = atom.GetTotalValence() - (offC * 4)
                elif prop in ['implicit-valence']:
                    val = atom.GetImplicitValence()
                elif prop in ['hybridization']:
                    val = atom.GetHybridization()
                elif prop in ['chirality']:
                    val = (atom.GetProp('_CIPCode') == "R") if atom.HasProp('_CIPCode') else 2
                elif prop in ['hybridization']:
                    val = atom.GetHybridization()
                elif prop in ['aromatic']:
                    val = atom.GetIsAromatic()
                elif prop in ['ring', 'in-ring']:
                    prop_name = 'in-ring'
                    val = atom.IsInRing()
                elif prop in ['degree']:
                    val = atom.GetTotalDegree() - (offC * 2)
                elif prop in ['radical-electron']:
                    val = atom.GetNumRadicalElectrons()
                elif prop in ['formal-charge']:
                    val = atom.GetFormalCharge()
                elif prop in ['vdw-radius']:
                    val = (periodic_table.GetRvdw(atom.GetAtomicNum()) - offC*periodic_table.GetRvdw(C.GetAtomicNum()))
                elif prop in ['covalent-radius']:
                    val = (periodic_table.GetRCovalent(atom.GetAtomicNum()) - offC*periodic_table.GetRCovalent(C.GetAtomicNum()))
                elif prop in ['electronegativity']:
                    val = (nmp.PERIODIC_TABLE['Electronegativity'][atom.GetAtomicNum()] - offC*nmp.PERIODIC_TABLE['Electronegativity'][C.GetAtomicNum()])
                elif prop in ['ionization', 'first-ionization']:
                    prop_name = 'ionization'
                    val = (nmp.PERIODIC_TABLE['FirstIonization'][atom.GetAtomicNum()] - offC*nmp.PERIODIC_TABLE['FirstIonization'][C.GetAtomicNum()]) / 5
                elif prop in ['melting-point']:
                    val = (nmp.PERIODIC_TABLE['MeltingPoint'][atom.GetAtomicNum()] - offC*nmp.PERIODIC_TABLE['MeltingPoint'][C.GetAtomicNum()]) / 200
                elif prop in ['metal']:
                    if nmp.PERIODIC_TABLE['Metal'][atom.GetAtomicNum()] == 'yes':
                        val = 2
                    elif nmp.PERIODIC_TABLE['Metalloid'][atom.GetAtomicNum()] == 'yes':
                        val = 1
                    else:
                        val = 0
                elif '-bond' in prop:
                    bonds = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
                    if prop in ['single-bond']:
                        val = len([bond == 1 for bond in bonds])
                    elif prop in ['aromatic-bond']:
                        val = len([bond == 1.5 for bond in bonds])
                    elif prop in ['double-bond']:
                        val = len([bond == 2 for bond in bonds])
                    elif prop in ['triple-bond']:
                        val = len([bond == 3 for bond in bonds])
                    val -= offC * 1
                elif prop in ['is-carbon']:
                    val = atom.GetAtomicNum() == 6
                    val -= offC * 1
                else:
                    raise ValueError(f'Unsupported property `{prop}`')

            elif callable(prop):
                prop_name = str(prop)
                val = prop(atom)
            else:
                ValueError(f'Elements in `property_list` must be str or callable, provided `{type(prop)}`')

            property_array[ii] = val

        prop_dict[prop_name] = property_array

    return prop_dict


def get_mol_edge_features(mol, property_list: List[str]):
    r"""
    Get the following set of features for any given bond
    See `goli.mol_utils.nmp` for allowed values in one hot encoding

    * One-hot representation of the bond type. Note that you should not kekulize your
        molecules, if you expect this to take aromatic bond into account.
    * Bond stereo type, following CIP classification
    * Whether the bond is conjugated
    * Whether the bond is in a ring

    Parameters:
        mol: rdkit.Chem.Molecule
            the molecule of interest

    Returns:
        features: float numpy.ndarray
            list of the above-mentionned features

    """

    prop_dict = {}

    # Compute features for each bond
    num_bonds = mol.GetNumBonds()
    for prop in property_list:
        property_array = []
        for ii in range(num_bonds):
            bond = mol.GetBondWithIdx(ii)

            if prop in ['bond-type-onehot']:
                one_hot = one_of_k_encoding(bond.GetBondType(), nmp.BOND_TYPES)
            if prop in ['bond-type-float']:
                one_hot =  [bond.GetBondTypeAsDouble()]
            elif prop in ['stereo']:
                one_hot = one_of_k_encoding(bond.GetStereo(), nmp.BOND_STEREO)
            elif prop in ['in-ring']:
                one_hot = [bond.IsInRing()]
            elif prop in ['conjugated']:
                one_hot = [bond.GetIsConjugated()]
            else:
                raise ValueError(f'Unsupported property `{prop}`')

            property_array.append(np.asarray(one_hot, dtype=np.float32))

        prop_dict[prop_name] = np.stack(property_array, axis=0)

    return prop_dict


def mol_to_adj_and_features(mol, 
        atom_property_list_onehot: List[str] = [],
        atom_property_list_float: List[Union[str, Callable]] = [],
        edge_property_list: List[str] = [],
        add_self_loop: bool = False,
        explicit_H = False,
        use_bonds = False,
        ):
    r"""
    Transforms a molecule into an adjacency matrix representing the molecular graph
    and a set of atom (and bond) features.
    :raises ValueError: when input molecule is None

    Parameters:
        mol (rdkit.Chem.Mol): The molecule to be converted

    Returns:
        
    """

    if explicit_H:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs

    adj = GetAdjacencyMatrix(mol, useBO=use_bonds, force=True)
    if add_self_loop:
        adj = adj + np.eye(adj.shape[0])
    atom_features_onehot = get_mol_atomic_features_onehot(mol, atom_property_list_onehot)
    atom_features_float = get_mol_atomic_features_float(mol, atom_property_list_float)
    edge_features = get_mol_edge_features(mol, edge_property_list)

    return (adj, atom_features_float, atom_features_onehot, edge_features)



def mol_to_dgl(mol,
        mol_transformer: Callable = dgl.DGLGraph,
        atom_property_list_onehot: List[str] = [],
        atom_property_list_float: List[Union[str, Callable]] = [],
        edge_property_list: List[str] = [],
        add_self_loop: bool = False,
        explicit_H = False,
        use_bonds = False,
        dtype = torch.float32,
        ):
    r"""
    Transforms a molecule into an adjacency matrix representing the molecular graph
    and a set of atom (and bond) features.
    :raises ValueError: when input molecule is None

    Parameters:
        mol (rdkit.Chem.Mol): The molecule to be converted

    Returns:
        
    """

    adj, atom_features_float, atom_features_onehot, edge_features = \
        mol_to_adj_and_features(
            mol=mol,
            atom_property_list_onehot=atom_property_list_onehot,
            atom_property_list_float=atom_property_list_float,
            edge_property_list=edge_property_list,
            add_self_loop=add_self_loop,
            explicit_H = explicit_H,
            use_bonds = use_bonds,
            )

    ndata = list(atom_features_float.values()) + list(atom_features_onehot.values())
    edata = list(edge_features.values())
    
    graph = dgl.from_scipy(csr_matrix(adj), idtype=dtype)
    graph['ndata'] = torch.cat(ndata, dim=1)
    graph['edata'] = torch.cat(edata, dim=1)

    return graph



