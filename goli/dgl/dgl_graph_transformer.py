import torch
import numpy as np
import warnings

from dgl import DGLGraph

from goli.commons.utils import (
    one_of_k_encoding,
    to_tensor,
    is_dtype_torch_tensor,
    is_dtype_numpy_array,
)

from goli.mol_utils import nmp
from goli.mol_utils.transformers import MoleculeTransformer, AdjGraphTransformer
from goli.mol_utils.properties import get_atom_features, get_edge_features


class DGLGraphTransformer(AdjGraphTransformer):
    r"""
    Transforms a molecule into a DGL graph for neural message passing algorithms

    Arguments:
        explicit_H: bool, optional
            Whether to consider hydrogen atoms explicitely. If this option
            is set to False, the number of hydrogen bond formed by the atom will be considered as a feature.
            (Default value = False)
        chirality: bool, optional
            Use chirality as a feature.
            (Default value = True)
        edge_label_only: bool, optional
            Do not compute bond features, and only use edge type as features
            (Default value = False)

    Attributes:
        n_atom_feat: Number of features per atom. This is computed dynamically according to the
            input parameters
        n_bond_feat: Number of bond features. This attribute is also computed dynamically


    """

    def __init__(self, explicit_H=False, chirality=True, edge_label_only=False):
        super().__init__(
            max_n_atoms=None,
            with_bond=False,
            explicit_H=explicit_H,
            chirality=chirality,
        )
        self.edge_labels = edge_label_only
        if self.edge_labels:
            self.n_bond_feat = 1

    def _transform(self, mol):
        r"""
        Transforms a molecule into a DGL graph and a set of atom and bond features
        :raises ValueError: when input molecule is None

        Arguments
        ----------
            mol: rdkit.Chem.Mol
                The molecule to be converted

        Returns
        -------
            features: A tuple (G, V, E), where G is the DGL graph, V is the node features
                and E is the edge features
        """
        if mol is None:
            raise ValueError("Expecting a Chem.Mol object, got None")
        atom_feats = []
        bond_feats = []
        n_atoms = mol.GetNumAtoms()
        # n_bonds = mol.GetNumBonds()
        graph = DGLGraph()

        for a_idx in range(0, n_atoms):
            atom = mol.GetAtomWithIdx(a_idx)
            atom_feats.append(
                get_atom_features(atom, explicit_H=self.explicit_H, use_chirality=self.use_chirality)
            )
        graph.add_nodes(n_atoms)

        bond_src = []
        bond_dst = []
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtom().GetIdx()
            end_idx = bond.GetEndAtom().GetIdx()
            features = one_of_k_encoding(bond.GetBondType(), nmp.BOND_TYPES)
            features = np.nonzero(features)[0][0]
            if not self.edge_labels:
                features = get_edge_features(bond)
            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_feats.append(features)
            # set up the reverse direction
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
            bond_feats.append(features)
        graph.add_edges(bond_src, bond_dst)

        # we need to call torch.stack on atom_feats and bond_feats
        return (graph, np.asarray(atom_feats), np.asarray(bond_feats))

    @staticmethod
    def to(graphs, dtype, device):
        out = []
        for graph in graphs:
            if graph is None:
                out.append(graph)
                continue
            else:
                g, v, e = graph
            if is_dtype_torch_tensor(dtype):
                v = to_tensor(v, device=device, dtype=dtype)
                e = to_tensor(e, device=device, dtype=dtype)
            elif is_dtype_numpy_array(dtype):
                v = np.asarray(v, dtype=dtype)
                e = np.asarray(e, dtype=dtype)
            else:
                raise (TypeError("The type {} is not supported".format(dtype)))

            g = g.to(device=device)
            g.ndata["h"] = v
            g.edata["e"] = e
            out.append(g)

        print("---------------------------------")
        print(out[0])
        print(out[0].__dict__)
        return out

    def __call__(self, mols, dtype=torch.float, device=None, **kwargs):
        r"""
        Transforms a batch of N molecules into a DGL representation.

        .. note::
            In contrast to the transform method for this class (DGLGraphTransformer),
            the __call__ returns a list of DGL objects (with atom and bond features already added),
            and not a list of tuples. It's important to understand the difference,
            as the `__call__` method is seen as a shortcut to speed up data processing in most experiments.

        Arguments
        ----------
            mols: (str or rdkit.Chem.Mol) iterable
                The list of input smiles or molecules
            dtype: torch.dtype or None
                Datatype of the transformed variable.
                Expect a tensor if you provide a torch dtype, a numpy array if you provide anything
                else that remains a valid (numpy) dtype.
                (Default value = torch.long)
            device: torch.device, optional
                The device on which to run the computation
            kwargs: named parameters for transform (see below)

        Returns
        -------
            features: list of DGL object
                By convention, the atom features are saved in `g.ndata['hv']`
                and the bond features are saved in `g.edata['he']`, where `g` corresponds to a DGL graph.
                See `transform` method for more information
            ids: array
                all valid molecule positions that did not failed during featurization

        See Also
        --------
            :func:`~goli.mol_utils.MolGraphTransformer.transform`

        """

        graphs, ids = super().__call__(mols, dtype=None, **kwargs)
        out = self.to(graphs, dtype=dtype, device=device)

        return out, ids
