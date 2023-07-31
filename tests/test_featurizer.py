"""
Unit tests for the different datasets of graphium/features/featurizer.py
"""

import numpy as np
import unittest as ut
from copy import deepcopy
from rdkit import Chem
import datamol as dm

from graphium.features.featurizer import (
    get_mol_atomic_features_onehot,
    get_mol_atomic_features_float,
    get_mol_edge_features,
    mol_to_adj_and_features,
    mol_to_pyggraph,
)


class test_featurizer(ut.TestCase):
    smiles = [
        "C",
        "CC",
        "C1(C[N]CCC1)=O",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N",
        "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5",
        "CC1(C2C1C(N(C2)C(=O)C(C(C)=B)NC(=O)C(F)(F)F)C(=O)NC(C(C3CCNC3=O)[Cl])C#N)C",
    ]

    smiles_noble = ["[He].[He]", "[He][He]", "[Kr][Kr]"]

    atomic_onehot_props = [
        "atomic-number",
        "valence",
        "degree",
        "implicit-valence",
        "hybridization",
        "chirality",
        "phase",
        "type",
        "group",
        "period",
    ]

    atomic_float_props = [
        "atomic-number",
        "mass",
        "valence",
        "implicit-valence",
        "hybridization",
        "chirality",
        "aromatic",
        "in-ring",
        "min-ring",
        "max-ring",
        "num-ring",
        "degree",
        "radical-electron",
        "formal-charge",
        "vdw-radius",
        "covalent-radius",
        "electronegativity",
        "ionization",
        "melting-point",
        "metal",
        "single-bond",
        "aromatic-bond",
        "double-bond",
        "triple-bond",
        "is-carbon",
        "group",
        "period",
    ]

    edge_props = [
        "bond-type-onehot",
        "bond-type-float",
        "stereo",
        "in-ring",
        "conjugated",
        "estimated-bond-length",
        "conformer-bond-length",
    ]

    def test_get_mol_atomic_features_onehot(self):
        props = deepcopy(self.atomic_onehot_props)
        bad_props = ["bob"]

        all_smiles = self.smiles + self.smiles_noble

        for s in all_smiles:
            err_msg = f"\n\tError for params:\n\t\tSMILES: {s}"
            mol = dm.to_mol(s)

            for ii in range(len(props)):
                this_props = props[:ii]
                err_msg2 = err_msg + f"\n\t\tprops: {this_props}"
                prop_dict = get_mol_atomic_features_onehot(mol, property_list=this_props)
                self.assertListEqual(list(prop_dict.keys()), this_props, msg=err_msg)
                for key, val in prop_dict.items():
                    err_msg3 = err_msg2 + f"\n\t\tkey: {key}"
                    self.assertEqual(val.shape[0], mol.GetNumAtoms(), msg=err_msg3)
                    self.assertGreater(val.shape[1], 1, msg=err_msg3)
                    self.assertTrue(np.all((val == 0) | (val == 1)), msg=err_msg3)

            with self.assertRaises(ValueError, msg=err_msg):
                get_mol_atomic_features_onehot(mol, property_list=bad_props)

    def test_get_mol_atomic_features_float(self):
        props = deepcopy(self.atomic_float_props)

        bad_props = ["bob"]

        all_smiles = self.smiles + self.smiles_noble
        for s in all_smiles:
            err_msg = f"\n\tError for params:\n\t\tSMILES: {s}"
            mol = dm.to_mol(s)

            for ii in range(len(props)):
                this_props = props[:ii]
                err_msg2 = err_msg + f"\n\t\tprops: {this_props}"
                prop_dict = get_mol_atomic_features_float(mol, property_list=this_props, mask_nan=None)
                self.assertListEqual(list(prop_dict.keys()), this_props, msg=err_msg)
                for key, val in prop_dict.items():
                    err_msg3 = err_msg2 + f"\n\t\tkey: {key}"
                    self.assertListEqual(list(val.shape), [mol.GetNumAtoms()], msg=err_msg3)

            with self.assertRaises(ValueError, msg=err_msg):
                get_mol_atomic_features_float(mol, property_list=bad_props)

    def test_get_mol_atomic_features_float_nan_mask(self):
        for s in self.smiles_noble:
            mol = dm.to_mol(s)

            # Nothing happens when `mask_nan = None`, nans are still in the property array
            prop_dict = get_mol_atomic_features_float(
                mol, property_list=self.atomic_float_props, mask_nan=None
            )
            prop_array = np.concatenate(list(prop_dict.values()), axis=0)
            nans = np.isnan(prop_array)

            # Capture a raised error when `mask_nan = "raise"`
            with self.assertRaises(ValueError):
                prop_dict = get_mol_atomic_features_float(
                    mol, property_list=self.atomic_float_props, mask_nan="raise"
                )

            # Not sure how to Capture a logged warning when `mask_nan = "warn"`
            # Here, I'm testing a behaviour similar to `mask_nan = None`
            prop_dict = get_mol_atomic_features_float(
                mol, property_list=self.atomic_float_props, mask_nan="warn"
            )
            prop_array = np.concatenate(list(prop_dict.values()), axis=0)
            self.assertEqual(len(self.atomic_float_props), len(prop_dict))
            self.assertTrue(any(np.isnan(prop_array)))

            # NaNs are replaced by `42` when `mask_nan=42`
            prop_dict = get_mol_atomic_features_float(mol, property_list=self.atomic_float_props, mask_nan=42)
            prop_array = np.concatenate(list(prop_dict.values()), axis=0)
            self.assertEqual(len(self.atomic_float_props), len(prop_dict))
            self.assertFalse(any(np.isnan(prop_array)))
            self.assertTrue(all(prop_array[nans] == 42))

    def test_get_mol_edge_features(self):
        props = deepcopy(self.edge_props)
        bad_props = ["bob"]

        all_smiles = self.smiles + self.smiles_noble
        for s in all_smiles:
            err_msg = f"\n\tError for params:\n\t\tSMILES: {s}"
            mol = dm.to_mol(s)
            for ii in range(len(props)):
                this_props = props[: ii + 1]
                err_msg2 = err_msg + f"\n\t\tprops: {this_props}"
                prop_dict = get_mol_edge_features(mol, property_list=this_props)
                self.assertListEqual(list(prop_dict.keys()), this_props, msg=err_msg)
                for key, val in prop_dict.items():
                    err_msg3 = err_msg2 + f"\n\t\tkey: {key}"
                    self.assertEqual(val.shape[0], mol.GetNumBonds(), msg=err_msg3)

            if mol.GetNumBonds() > 0:
                with self.assertRaises(ValueError, msg=err_msg):
                    get_mol_edge_features(mol, property_list=bad_props)

    def test_mol_to_adj_and_features(self):
        np.random.seed(42)

        for s in self.smiles:
            err_msg = f"\n\tError for params:\n\t\tSMILES: {s}"
            mol = dm.to_mol(s)
            mol_Hs = Chem.AddHs(mol)  # type: ignore
            mol_No_Hs = Chem.RemoveHs(mol)  # type: ignore

            for explicit_H in [True, False]:
                this_mol = mol_Hs if explicit_H else mol_No_Hs
                for ii in np.arange(0, 5, 0.2):
                    num_props = int(round(ii))
                    err_msg2 = err_msg + f"\n\t\texplicit_H: {explicit_H}\n\t\tii: {ii}"

                    adj, ndata, edata, _, _ = mol_to_adj_and_features(
                        mol=mol,
                        atom_property_list_onehot=np.random.choice(
                            self.atomic_onehot_props, size=num_props, replace=False
                        ),
                        atom_property_list_float=np.random.choice(
                            self.atomic_float_props, size=num_props, replace=False
                        ),
                        edge_property_list=np.random.choice(self.edge_props, size=num_props, replace=False),
                        add_self_loop=False,
                        explicit_H=explicit_H,
                        use_bonds_weights=False,
                    )

                    self.assertEqual(adj.shape[0], this_mol.GetNumAtoms(), msg=err_msg2)
                    if num_props > 0:
                        self.assertEqual(ndata.shape[0], this_mol.GetNumAtoms(), msg=err_msg2)
                        if this_mol.GetNumBonds() > 0:
                            self.assertEqual(edata.shape[0], this_mol.GetNumBonds(), msg=err_msg2)
                            self.assertGreaterEqual(edata.shape[1], num_props, msg=err_msg2)
                        self.assertGreaterEqual(ndata.shape[1], num_props, msg=err_msg2)

    def test_mol_to_pyggraph(self):
        np.random.seed(42)

        for s in self.smiles:
            err_msg = f"\n\tError for params:\n\t\tSMILES: {s}"
            mol = dm.to_mol(s)
            mol_Hs = Chem.AddHs(mol)  # type: ignore
            mol_No_Hs = Chem.RemoveHs(mol)  # type: ignore

            graph = mol_to_pyggraph(
                mol=mol,
                atom_property_list_onehot=[],
                atom_property_list_float=["atomic-number"],
                edge_property_list=["bond-type-float"],
                add_self_loop=False,
                explicit_H=False,
                use_bonds_weights=False,
                on_error="raise",
            )

            # Check the number of nodes and edges
            self.assertListEqual(list(graph["feat"].shape), [mol.GetNumAtoms(), 1], msg=err_msg)
            self.assertListEqual(list(graph["edge_feat"].shape), [2 * mol.GetNumBonds(), 1], msg=err_msg)

            # Check the node features
            feat = graph["feat"].to_dense().numpy() * 5 + 6  # Undo the scaling
            atom_nums = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            np.testing.assert_array_almost_equal(feat[:, 0], atom_nums, decimal=5, err_msg=err_msg)

            # Check the edge features
            edge_feat = graph["edge_feat"].to_dense().numpy()
            bond_types = np.asarray([bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]).repeat(2)
            np.testing.assert_array_almost_equal(edge_feat[:, 0], bond_types, decimal=5, err_msg=err_msg)

            # Check the edge indices
            if mol.GetNumBonds() > 0:
                edge_index = graph["edge_index"].to_dense().numpy()
                true_edge_index = []
                for bond in mol.GetBonds():
                    true_edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                    true_edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
                true_edge_index = np.asarray(true_edge_index).T
                np.testing.assert_array_equal(edge_index, true_edge_index, err_msg=err_msg)

            # Loop over many possible combinations of properties
            for explicit_H in [True, False]:
                this_mol = mol_Hs if explicit_H else mol_No_Hs
                for ii in np.arange(0, 5, 0.2):
                    num_props = int(round(ii))
                    err_msg2 = err_msg + f"\n\t\texplicit_H: {explicit_H}\n\t\tii: {ii}"

                    graph = mol_to_pyggraph(
                        mol=mol,
                        atom_property_list_onehot=np.random.choice(
                            self.atomic_onehot_props, size=num_props, replace=False
                        ),
                        atom_property_list_float=np.random.choice(
                            self.atomic_float_props, size=num_props, replace=False
                        ),
                        edge_property_list=np.random.choice(self.edge_props, size=num_props, replace=False),
                        add_self_loop=False,
                        explicit_H=explicit_H,
                        use_bonds_weights=False,
                        on_error="raise",
                    )

                    self.assertEqual(graph.num_nodes, this_mol.GetNumAtoms(), msg=err_msg2)
                    self.assertEqual(graph.num_edges, 2 * this_mol.GetNumBonds(), msg=err_msg2)
                    if num_props > 0:
                        ndata = graph["feat"]
                        edata = graph["edge_feat"]
                        self.assertEqual(ndata.shape[0], this_mol.GetNumAtoms(), msg=err_msg2)
                        self.assertEqual(edata.shape[0], 2 * this_mol.GetNumBonds(), msg=err_msg2)
                        self.assertGreaterEqual(ndata.shape[1], num_props, msg=err_msg2)
                        self.assertGreaterEqual(edata.shape[1], num_props, msg=err_msg2)


if __name__ == "__main__":
    ut.main()
