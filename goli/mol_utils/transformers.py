import torch
import numpy as np
import warnings

from sklearn.base import TransformerMixin

from goli.commons.utils import (
    to_tensor,
    is_dtype_torch_tensor,
    is_dtype_numpy_array,
)

from goli.mol_utils import nmp
from goli.mol_utils.properties import get_atom_features, get_edge_features

from rdkit import Chem
from rdkit.Chem.rdmolops import RenumberAtoms


def to_mol(mol, addHs=False, explicitOnly=True, ordered=True, kekulize=True, sanitize=True):
    r"""
    Convert an imput molecule (smiles representation) into a Chem.Mol
    :raises ValueError: if the input is neither a CHem.Mol nor a string

    Parameters:
        mol: str or rdkit.Chem.Mol
            SMILES of a molecule or a molecule
        addHs: bool): Whether hydrogens should be added the molecule.

        explicitOnly: bool
            Whether to only add explicit hydrogen or both
            (implicit and explicit) when addHs is set to True.

        ordered: bool
            Whether the atom should be ordered. This option is important if you want to ensure
            that the features returned will always maintain a sinfle atom order for the same molecule,
            regardless of its original smiles representation
        kekulize: bool
            Kekulize input molecule
        sanitize: bool
            Whether to apply rdkit sanitization when input is a smiles

    Returns:
        mol: rdkit.Chem.Molecule
            the molecule if some conversion have been made.
            If the conversion fails None is returned so make sure that you handle this case on your own.
    """
    if not isinstance(mol, (str, Chem.Mol)):
        raise ValueError("Input should be a Chem.Mol or a valid SMILES string, received: ", type(mol))
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol, sanitize=sanitize)
        if not sanitize and mol is not None:
            mol.UpdatePropertyCache(False)
    # make more sense to add hydrogen before ordering
    if mol is not None and addHs:
        mol = Chem.AddHs(mol, explicitOnly=explicitOnly)
    if mol and mol.GetNumAtoms() > 0 and ordered:
        new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=True)
        new_order = sorted([(y, x) for x, y in enumerate(new_order)])
        mol = RenumberAtoms(mol, [y for (x, y) in new_order])
    if kekulize and mol is not None:
        Chem.Kekulize(mol, clearAromaticFlags=False)
    return mol


class MoleculeTransformer(TransformerMixin):
    r"""
    Transform a molecule (rdkit.Chem.Mol object) into a feature representation.
    This class is an abstract class, and all its children are expected to implement the `_transform` method.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, **fit_params):
        return self

    def _transform(self, mol):
        r"""
        Compute features for a single molecule.
        This method need to be implemented by each child that inherits from MoleculeTransformer
        :raises NotImplementedError: if the method is not implemented by the child class
        Parameters:
            mol: Chem.Mol
                molecule to transform into features

        Returns:
            features: the list of features

        """
        raise NotImplementedError("Missing implementation of _transform.")

    def transform(self, mols, ignore_errors=True, **kwargs):
        r"""
        Compute the features for a set of molecules.

        !!! note
            Note that depending on the `ignore_errors` argument, all failed
            featurization (caused whether by invalid smiles or error during
            data transformation) will be substitued by None features for the
            corresponding molecule. This is done, so you can find the positions
            of these molecules and filter them out according to your own logic.

        Parameters:
            mols: list(Chem.Mol) or list(str)
                a list containing smiles or Chem.Mol objects
            ignore_errors: bool
                Whether to silently ignore errors
            kwargs:
                named Parameters that are to be passed to the `to_mol` function.

        Returns:
            features: a list of features for each molecule in the input set
        """

        features = []
        for i, mol in enumerate(mols):
            feat = None
            if ignore_errors:
                try:
                    mol = to_mol(mol, **kwargs)
                    feat = self._transform(mol)
                except:
                    pass
            else:
                mol = to_mol(mol, **kwargs)
                feat = self._transform(mol)
            features.append(feat)
        return features

    @staticmethod
    def _change_type(fp, dtype=list, device=None, ignore_errors=True):
        if len(fp) == 0:
            if ignore_errors:
                return fp
            else:
                raise ValueError("Expected non-empty fingerprint list in type change.")

        if dtype == list:
            fp = list(fp)
        elif is_dtype_numpy_array(dtype):
            if isinstance(fp, (list, tuple)):
                fp = np.stack([np.array(this_fp, dtype=dtype) for this_fp in fp], axis=0)
            fp = np.array(fp, dtype=dtype)
        elif is_dtype_torch_tensor(dtype):
            if isinstance(fp, (list, tuple)):
                fp = torch.stack([to_tensor(this_fp, device=device, dtype=dtype) for this_fp in fp], dim=0)
            else:
                fp = to_tensor(fp, device=device, dtype=dtype)
        else:
            raise TypeError("The type {} is not supported".format(dtype))

        return fp

    def __call__(self, mols, ignore_errors=True, **kwargs):
        r"""
        Calculate features for molecules. Using __call__, instead of transform. This function
        will force ignore_errors to be true, regardless of your original settings, and is offered
        mainly as a shortcut for data preprocessing. Note that most Transfomers allow you to specify
        a return datatype.

        Parameters:
            mols: (str or rdkit.Chem.Mol) iterable
                SMILES of the molecules to be transformed
            ignore_errors: bool
                Whether to ignore errors and silently fallback
            kwargs: Named parameters for the transform method

        Returns:
            feats: array
                list of valid features
            ids: array
                all valid molecule positions that did not failed during featurization

        !!! seealso
            `goli.mol_utils.transformers.MoleculeTransformer.transform`

        """
        feats = self.transform(mols, ignore_errors=ignore_errors, **kwargs)

        ids = []
        for f_id, feat in enumerate(feats):
            if feat is not None:
                ids.append(f_id)
        return self._filter_None(feats)

    @staticmethod
    def _filter_None(features):
        ids_bad = []

        # If the features are a list, filter the None ids
        if isinstance(features, (tuple, list)):
            for f_id, feat in enumerate(features):
                if feat is None:
                    ids_bad.append(f_id)
            all_ids = np.arange(0, len(features))
            ids_to_keep = [this_id for this_id in all_ids if this_id not in ids_bad]
            features = [features[ii] for ii in ids_to_keep]

        else:
            ids_to_keep = np.arange(0, features.shape[0])

        return features, ids_to_keep


class AdjGraphTransformer(MoleculeTransformer):
    r"""
    Transforms a molecule into a molecular graph representation formed by an
    adjacency matrix of atoms and a set of features for each atom (and potentially bond).

    Parameters:
        max_n_atoms: int
            Maximum number of atom, to set the size of the graph.
            Use default value None, to allow graph with different size that will be packed together later
            
        with_bond: bool
            Whether to enable the feature of the bond formed by each atom in the atom feature
            
        explicit_H: bool
            Whether to consider hydrogen atoms explicitely. If this option
            is set to False, the number of hydrogen bond formed by the atom will be considered as a feature.
            
        chirality: bool
            Use chirality as a feature.
            
        max_valence: int
            Maximum number of neighbor for each atom.
            This option is required if you want to return use bond features
            
        padding_val: int
            Padding value to fill missing edges features
            when the atom valence is lower than the maximum allowed valence
            
    Attributes:
        n_atom_feat: Number of features per atom. This is computed dynamically according to the
            input parameters
        n_bond_feat: Number of bond features. This attribute is also computed dynamically
        explicit_H: Indicates if explicit hydrogens are to be added to the molecules
        with_bond: Indicates if bond feature are to be added to atom features
        use_chirality: Indicates if chirality is being used as a feature.
        max_n_atoms: Maximum number of atoms allowed in the graph. Can be None
        padding_val: Padding value used to fill missing edges features


    """

    def __init__(
        self,
        max_n_atoms=None,
        with_bond=False,
        explicit_H=False,
        chirality=True,
        max_valence=4,
        padding_val=0,
    ):
        self.max_valence = max_valence
        # if this is not set, packing of graph would be expected later
        self.max_n_atoms = max_n_atoms
        self.n_atom_feat = 0
        self.n_bond_feat = 0
        self.explicit_H = explicit_H
        self.use_chirality = chirality
        self.with_bond = with_bond
        self.padding_val = padding_val
        self._set_num_features()

    @property
    def atom_dim(self):
        r"""
        Get the number of features per atom

        Returns:
            atom_dim: int
                Number of atom features
        """
        return self.n_atom_feat

    @property
    def bond_dim(self):
        r"""
        Get the number of features for a bond

        Returns:
            bond_dim: int
                Number of bond features
        """
        return self.n_bond_feat

    def _set_num_features(self):
        r"""Compute the number of features for each atom and bond"""
        self.n_atom_feat = 0
        # add atom type required
        self.n_atom_feat += len(nmp.ATOM_LIST) + 1
        # add atom degree
        self.n_atom_feat += len(nmp.ATOM_DEGREE_LIST) + 1
        # add valence implicit
        self.n_atom_feat += len(nmp.IMPLICIT_VALENCE) + 1
        # aromatic, formal charge, radical electrons, in_ring
        self.n_atom_feat += 4
        # hybridation_list
        self.n_atom_feat += len(nmp.HYBRIDIZATION_LIST) + 1
        # number of hydrogen
        if not self.explicit_H:
            self.n_atom_feat += len(nmp.ATOM_NUM_H) + 1
        # chirality
        if self.use_chirality:
            self.n_atom_feat += 3

        # do the same thing but with bond now
        # start with bond types
        self.n_bond_feat += len(nmp.BOND_TYPES) + 1
        # stereo
        self.n_bond_feat += len(nmp.BOND_STEREO) + 1
        # bond is conjugated, in rings
        self.n_bond_feat += 2

        if self.with_bond:
            self.n_atom_feat += self.n_bond_feat * (self.max_valence)

    def transform(self, mols, ignore_errors=True, max_atom_update=False):
        r"""
        Transforms a batch of N molecules or smiles into an Adjacency graph with a set of
        atom feature

        Parameters:
            mols: (str or rdkit.Chem.Mol) iterable
                Molecules to transform into graphs
            ignore_errors: bool
                Whether to silently ignore errors
                
            max_atom_update: bool
                Whether the maximum number of atoms in the graph be dynamically updated for eacher longuer molecule seen ?
                If you are using this, all molecules should be transformed before batching

        Returns:
            features: A list of tuple (A, x), where A is an adjacency matrix
                and x is the set of atom features

        """

        features = []
        mol_list = []
        for i, ml in enumerate(mols):
            mol = None
            if ignore_errors:
                try:
                    mol = to_mol(ml, addHs=self.explicit_H, ordered=True)
                except Exception as e:
                    pass
            else:
                mol = to_mol(ml, addHs=self.explicit_H, ordered=True)

            if mol:
                num_atom = mol.GetNumAtoms()
                # getting the max_n_atoms is why we are doing this stuff
                # in two loop
                if self.max_n_atoms and max_atom_update and self.max_n_atoms < num_atom:
                    warnings.warn("Max number of atoms is not enough, Updating to {}".format(num_atom))
                    self.max_n_atoms = num_atom
            elif mol is None and not ignore_errors:
                raise (ValueError("Molecule {} cannot be transformed adjency graph".format(ml)))
            mol_list.append(mol)

        for mol in mol_list:
            feat = None
            if mol is not None:
                if ignore_errors:
                    try:
                        feat = self._transform(mol)
                    except:
                        pass
                else:
                    feat = self._transform(mol)
            features.append(feat)

        return features

    def _transform(self, mol):
        r"""
        Transforms a molecule into an adjacency matrix representing the molecular graph
        and a set of atom (and bond) features.
        :raises ValueError: when input molecule is None

        Parameters:
            mol (rdkit.Chem.Mol): The molecule to be converted

        Returns:
            features: a tuple (A, X), where A is the adjacency matrix of size (N, N) for N atoms
                and X the feature matrix of size (N,D) for D features
        """

        if mol is None:
            raise ValueError("Expecting a Chem.Mol object, got None")

        n_atoms = self.max_n_atoms or mol.GetNumAtoms()
        # for each atom, we would have one neighbor at each of its valence state

        if self.with_bond:
            # Use padding value to fill
            bond_matrix = (
                np.zeros((n_atoms, self.n_bond_feat * self.max_valence), dtype=np.int) + self.padding_val
            )
            # type of bond for each of its neighbor respecting max valence

        adj_matrix = np.zeros((n_atoms, n_atoms), dtype=np.int)
        atom_arrays = []
        for a_idx in range(0, min(n_atoms, mol.GetNumAtoms())):
            atom = mol.GetAtomWithIdx(a_idx)
            atom_arrays.append(
                get_atom_features(atom, explicit_H=self.explicit_H, use_chirality=self.use_chirality)
            )
            # adj_matrix[a_idx, a_idx] = 1  # add self loop
            for n_pos, neighbor in enumerate(atom.GetNeighbors()):
                n_idx = neighbor.GetIdx()
                # do not exceed hard limit on the maximum number of atoms
                # allowed
                if n_idx < n_atoms:
                    adj_matrix[n_idx, a_idx] = 1
                    adj_matrix[a_idx, n_idx] = 1
                    if self.with_bond:
                        bond = mol.GetBondBetweenAtoms(a_idx, n_idx)
                        bond_feat = get_edge_features(bond)
                        cur_neigb = n_pos % self.max_valence
                        bond_matrix[a_idx][
                            (self.n_bond_feat * cur_neigb) : (self.n_bond_feat) * (cur_neigb + 1)
                        ] = bond_feat

        n_atom_shape = len(atom_arrays[0])
        atom_matrix = np.zeros((n_atoms, n_atom_shape)).astype(np.int)
        for idx, atom_array in enumerate(atom_arrays):
            atom_matrix[idx, :] = atom_array

        if self.with_bond:
            atom_matrix = np.concatenate([atom_matrix, bond_matrix], axis=1)

        return (adj_matrix, atom_matrix)

    def __call__(self, mols, dtype=torch.float, device=None, **kwargs):
        r"""
        Transforms a batch of N molecules or smiles into an Adjacency graph with a set of
        atom feature and return the transformation in the desired data type format and
        the set of valid indexes.

        Parameters:
            mols: (str or rdkit.Chem.Mol) iterable
                The list of input smiles or molecules
            dtype: torch.dtype or numpy.dtype
                Datatype of the transformed variable.
                Expect a tensor if you provide a torch dtype, a numpy array if you provide a
                numpy dtype (supports valid strings) or a vanilla int/float. Any other option will
                return the output of the transform function.
            device: torch.device
                The device on which to run the computation
            kwargs: named parameters for transform

        Returns:
            features : (list of tuple)
                see `transform` method for more information
            ids: array
                all valid molecule positions that did not failed during featurization

        !!! seealso
            `goli.mol_utils.transformers.AdjGraphTransformer.transform`

        !!! Example
                ``` python linenums="1"
                data = [
                        "CCOc1c(OC)cc(CCN)cc1OC",
                        "COc1cc(CCN)cc(OC)c1OC",
                        "COc1c2OCCc2c(CCN)c2CCOc12",
                        "COc1cc(CCN)c2CCOc2c1OC",
                        "C[C@H](N)Cn1ncc2ccc(O)cc12",
                        "COc1ccc2cnn(C[C@H](C)N)c2c1",
                        "C[C@H](N)Cn1ncc2ccc(O)c(C)c12",
                        ]

                transf = AdjGraphTransformer()
                # X is a list of 7 tuple each containing two tensors
                # one Adj. Mat and the corresponding atom feature set
                X = transf(data)
                ```

        """

        graphs, ids = super().__call__(mols, **kwargs)

        if is_dtype_torch_tensor(dtype):
            graphs = [
                (
                    to_tensor(x1, device=device, dtype=dtype),
                    to_tensor(x2, device=device, dtype=dtype),
                )
                for (x1, x2) in graphs
            ]
        elif is_dtype_numpy_array(dtype):
            graphs = [[np.array(x, dtype=dtype) for x in y] for y in graphs]
        elif dtype is None:
            pass
        else:
            raise (TypeError("The type {} is not supported".format(dtype)))
        return graphs, ids
