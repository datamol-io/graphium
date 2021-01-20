import numpy as np

from rdkit import Chem  # install rdkit using a conda environment
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rdMD
from mordred import Calculator, descriptors # in your conda environment, run `pip install mordred`

import goli.mol_utils.nmp as nmp

calc = Calculator(descriptors, ignore_3D=True)


def get_weight(mol):
    mol = mol_from_smiles_or_mol(mol)
    return Chem.rdMolDescriptors.CalcExactMolWt(mol)


def get_prop_or_none(prop, n, *args, **kwargs):
    """
    return properties. If error, return list of `None` with lenght `n`.
    """
    try:
        return prop(*args, **kwargs)
    except RuntimeError:
        return [None] * n


def mol_from_smiles_or_mol(smiles_or_mol):
    """
    Return an rdkit molecule based on the smiles or input molecule.
    """
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles)
    elif isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
    else:
        raise TypeError('Wrong type for `smiles_or_mol`. Must be `str` or `Chem.Mol`')
    return mol



def get_props_from_mol(mol, properties='autocorr3d'):
    """
    Function to get a given set of desired properties from a molecule,
    and output a property list.

    Parameters
    ---------------
        mol: rdkit Mol, str
            The molecule from which to compute the properties, or the SMILES representation
            of the molecule.
        properties: str, list(str)
            The list of properties to compute for each molecule. It can be the following:
            - 'descriptors'
            - 'autocorr3d'
            - 'rdf'
            - 'morse'
            - 'whim'
            - 'all'

    Returns
    -----------
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

    mol = mol_from_smiles_or_mol(mol)
    if isinstance(properties, str):
        properties = [properties]
    properties = [p.lower() for p in properties]
    
    # Initialize arrays
    props = []  # Property vector for the features
    classes_start_idx = []  # The starting index for each property class
    classes_names = []
    
    # Generate a 3D structure for the molecule
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDG()  # define the new code from RDKit Molecule 3D ETKDG.
    ps.randomSeed = 111
    AllChem.EmbedMolecule(mol, ps)

    if ('descriptors' in properties) or ('all' in properties):
        # Calculate the descriptors of the molecule
        for desc in descriptors.all:
            classes_names.append(desc.__name__.replace('mordred.', ''))
            classes_start_idx.append(len(props))
            calc = Calculator(desc, ignore_3D=True)
            props.extend(calc(mol))

    if ('autocorr3d' in properties) or ('all' in properties):
        # Some kind of 3D description of the molecule
        classes_names.append('autocorr3d')
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcAUTOCORR3D, 80, mol))

    if ('rdf' in properties) or ('all' in properties):
        # The radial distribution function (better than the inertia)
        # https://en.wikipedia.org/wiki/Radial_distribution_function
        classes_names.append('rdf')
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcRDF, 210, mol))

    if ('morse' in properties) or ('all' in properties):
        # Molecule Representation of Structures based on Electron diffraction descriptors
        classes_names.append('morse')
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcMORSE, 224, mol))

    if ('whim' in properties) or ('all' in properties):
        # WHIM descriptors are 3D structural descriptors obtained from the 
        # (x,y,z)‐atomic coordinates of a molecular conformation of a chemical,
        # and are used successfully in QSAR modelling.
        classes_names.append('whim')
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcWHIM, 114, mol))

    return np.array(props), classes_start_idx, classes_names


def one_of_k_encoding(val, num_classes, dtype=int):
    r"""Converts a single value to a one-hot vector.

    Arguments
    ----------
        val: int
            class to be converted into a one hot vector
            (integers from 0 to num_classes).
        num_classes: iterator
            a list or 1D array of allowed
            choices for val to take
        dtype: type, optional
            data type of the the return.
            (Default value = int). Other possible types are float, bool, ...
    Returns
    -------
        A numpy 1D array of length len(num_classes) + 1
    """

    encoding = np.zeros(len(num_classes) + 1, dtype=dtype)
    # not using index of, in case, someone fuck up
    # and there are duplicates in the allowed choices
    for i, v in enumerate(num_classes):
        if v == val:
            encoding[i] = 1
    if np.sum(encoding) == 0:  # aka not found
        encoding[-1] = 1
    return encoding


def get_atom_features(atom, explicit_H=False, use_chirality=True):
    r"""
    Get the following set of features for any given atom

    * One-hot representation of the atom
    * One-hot representation of the atom degree
    * One-hot representation of the atom implicit valence
    * One-hot representation of the the atom hybridization
    * Whether the atom is aromatic
    * The atom's formal charge
    * The atom's number of radical electrons
    * Whether the atom is in a ring

    Additionally, the following features can be set, depending on the value of input arguments

    * One-hot representation of the number of hydrogen atom in the the current atom neighborhood if `explicit_H` is false
    * One-hot encoding of the atom chirality, and whether such configuration is even possible

    Arguments
    ----------
        mol: `rdkit.Chem.Molecule`
            the molecule of interest

    Returns
    -------
        features: `numpy.ndarray(float)`
            a numpy array of the above-mentionned features

    """
    feats = []
    # Set type symbol
    feats.extend(one_of_k_encoding(atom.GetSymbol(), nmp.ATOM_LIST))
    # add the degree of the atom now
    feats.extend(one_of_k_encoding(atom.GetDegree(), nmp.ATOM_DEGREE_LIST))
    # mplicit valence
    feats.extend(one_of_k_encoding(
        atom.GetImplicitValence(), nmp.IMPLICIT_VALENCE))
    # add hybridization type of atom
    feats.extend(one_of_k_encoding(
        atom.GetHybridization(), nmp.HYBRIDIZATION_LIST))
    # whether the atom is aromatic or not
    feats.append(int(atom.GetIsAromatic()))
    # atom formal charge
    feats.append(atom.GetFormalCharge())
    # add number of radical electrons
    feats.append(atom.GetNumRadicalElectrons())
    # atom is in ring
    feats.append(int(atom.IsInRing()))

    if not explicit_H:
        # number of hydrogene, is usually 0 after Chem.AddHs(mol) is called
        feats.extend(one_of_k_encoding(atom.GetTotalNumHs(), nmp.ATOM_NUM_H))

    if use_chirality:
        try:
            feats.extend(one_of_k_encoding(
                atom.GetProp('_CIPCode'), nmp.CHIRALITY_LIST))
            feats.append(int(atom.HasProp('_ChiralityPossible')))

        except:
            feats.extend([0, 0, int(atom.HasProp('_ChiralityPossible'))])

    return np.asarray(feats, dtype=np.float32)


def get_edge_features(bond):
    r"""
    Get the following set of features for any given bond
    See `goli.mol_utils.nmp` for allowed values in one hot encoding

    * One-hot representation of the bond type. Note that you should not kekulize your
        molecules, if you expect this to take aromatic bond into account.
    * Bond stereo type, following CIP classification
    * Whether the bond is conjugated
    * Whether the bond is in a ring

    Arguments
    ----------
        mol: rdkit.Chem.Molecule
            the molecule of interest

    Returns
    -------
        features: float numpy.ndarray
            list of the above-mentionned features

    """
    # Initialise bond feature vector as an empty list
    edge_features = []
    # Encode bond type as a feature vector
    bond_type = bond.GetBondType()
    edge_features.extend(one_of_k_encoding(bond_type, nmp.BOND_TYPES))
    edge_features.extend(one_of_k_encoding(int(bond.GetStereo()), nmp.BOND_STEREO))
    # Encode whether the bond is conjugated or not
    edge_features.append(int(bond.GetIsConjugated()))
    # Encode whether the bond is in a ring or not
    edge_features.append(int(bond.IsInRing()))
    return np.array(edge_features, dtype=np.float32)


def mol_to_graph(mol, explicit_H=False, use_chirality=False):
    r"""
    Transforms a molecule into an adjacency matrix representing the molecular graph
    and a set of atom (and bond) features.
    :raises ValueError: when input molecule is None

    Arguments
    ----------
        mol (rdkit.Chem.Mol): The molecule to be converted

    Returns
    -------
        features: a tuple (A, X), where A is the adjacency matrix of size (N, N) for N atoms
            and X the feature matrix of size (N,D) for D features
    """

    

    n_atoms = mol.GetNumAtoms()
    # for each atom, we would have one neighbor at each of its valence state

    adj_matrix = np.zeros((n_atoms, n_atoms), dtype=np.int)
    atom_arrays = []
    for a_idx in range(0, min(n_atoms, mol.GetNumAtoms())):
        atom = mol.GetAtomWithIdx(a_idx)
        atom_arrays.append(get_atom_features(atom, explicit_H=explicit_H, use_chirality=use_chirality))
        # adj_matrix[a_idx, a_idx] = 1  # add self loop
        for n_pos, neighbor in enumerate(atom.GetNeighbors()):
            n_idx = neighbor.GetIdx()
            # do not exceed hard limit on the maximum number of atoms
            # allowed
            if n_idx < n_atoms:
                adj_matrix[n_idx, a_idx] = 1
                adj_matrix[a_idx, n_idx] = 1
                
    n_atom_shape = len(atom_arrays[0])
    atom_matrix = np.zeros(
        (n_atoms, n_atom_shape)).astype(np.int)
    for idx, atom_array in enumerate(atom_arrays):
        atom_matrix[idx, :] = atom_array

    return (adj_matrix, atom_matrix)



if __name__ == "__main__":
    smiles = 'CC(=O)NCCC1=CNc2c1cc(OC)cc2'
    weight = get_weight(smiles)
    print('________________________\nweight = {}\n____________________'.format(weight))

    # Example using 'autocorr3d' properties
    properties = 'autocorr3d'
    props, classes_start_idx, classes_names = get_props_from_mol(smiles, properties=properties)
    print('________________________\nproperties = {}\n____________________'.format(properties))
    print('props: \n{}'.format(props))
    print('classes_start_idx: \n{}'.format(classes_start_idx))
    print('classes_names: \n{}'.format(classes_names))

    # Example using 'all' properties
    properties = 'all'
    props, classes_start_idx, classes_names = get_props_from_mol(smiles, properties=properties)
    print('\n\n________________________\nproperties = {}\n____________________'.format(properties))
    print('props: \n{}'.format(props))
    print('classes_start_idx: \n{}'.format(classes_start_idx))
    print('classes_names: \n{}'.format(classes_names))


    print('done :)')


