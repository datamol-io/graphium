"""
Unit tests for the different datasets of goli/features/featurizer.py
"""

import numpy as np
import unittest as ut
from copy import deepcopy
from rdkit import Chem
import datamol as dm

from goli.features.featurizer import (
    get_mol_atomic_features_onehot,
    get_mol_atomic_features_float,
    get_mol_edge_features,
    mol_to_adj_and_features,
    mol_to_dglgraph,
)


class test_featurizer(ut.TestCase):

    smiles = [
        "CC",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N",
        "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5",
    ]

    atomic_onehot_props = [
        "atomic-number",
        "valence",
        "degree",
        "implicit-valence",
        "hybridization",
        "chirality",
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
    ]

    edge_props = [
        "bond-type-onehot",
        "bond-type-float",
        "stereo",
        "in-ring",
        "conjugated",
        "bond-length",
    ]

    def test_get_mol_atomic_features_onehot(self):
        props = deepcopy(self.atomic_onehot_props)
        bad_props = ["bob"]

        for s in self.smiles:
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

        for s in self.smiles:
            err_msg = f"\n\tError for params:\n\t\tSMILES: {s}"
            mol = dm.to_mol(s)

            for ii in range(len(props)):
                this_props = props[:ii]
                err_msg2 = err_msg + f"\n\t\tprops: {this_props}"
                prop_dict = get_mol_atomic_features_float(mol, property_list=this_props)
                self.assertListEqual(list(prop_dict.keys()), this_props, msg=err_msg)
                for key, val in prop_dict.items():
                    err_msg3 = err_msg2 + f"\n\t\tkey: {key}"
                    self.assertListEqual(list(val.shape), [mol.GetNumAtoms()], msg=err_msg3)

            with self.assertRaises(ValueError, msg=err_msg):
                get_mol_atomic_features_float(mol, property_list=bad_props)

    def test_get_mol_edge_features(self):
        props = deepcopy(self.edge_props)
        bad_props = ["bob"]

        for s in self.smiles:
            err_msg = f"\n\tError for params:\n\t\tSMILES: {s}"
            mol = dm.to_mol(s)
            for ii in range(len(props)):
                this_props = props[:ii]
                err_msg2 = err_msg + f"\n\t\tprops: {this_props}"
                prop_dict = get_mol_edge_features(mol, property_list=this_props)
                self.assertListEqual(list(prop_dict.keys()), this_props, msg=err_msg)
                for key, val in prop_dict.items():
                    err_msg3 = err_msg2 + f"\n\t\tkey: {key}"
                    self.assertEqual(val.shape[0], mol.GetNumBonds(), msg=err_msg3)

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

                    adj, ndata, edata = mol_to_adj_and_features(
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
                        self.assertEqual(edata.shape[0], this_mol.GetNumBonds(), msg=err_msg2)
                        self.assertGreaterEqual(ndata.shape[1], num_props, msg=err_msg2)
                        self.assertGreaterEqual(edata.shape[1], num_props, msg=err_msg2)

    def test_mol_to_dglgraph(self):

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

                    graph = mol_to_dglgraph(
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

                    self.assertEqual(graph.num_nodes(), this_mol.GetNumAtoms(), msg=err_msg2)
                    self.assertEqual(graph.num_edges(), 2 * this_mol.GetNumBonds(), msg=err_msg2)
                    if num_props > 0:
                        ndata = graph.ndata["feat"]
                        edata = graph.edata["feat"]
                        self.assertEqual(ndata.shape[0], this_mol.GetNumAtoms(), msg=err_msg2)
                        self.assertEqual(edata.shape[0], 2 * this_mol.GetNumBonds(), msg=err_msg2)
                        self.assertGreaterEqual(ndata.shape[1], num_props, msg=err_msg2)
                        self.assertGreaterEqual(edata.shape[1], num_props, msg=err_msg2)


if __name__ == "__main__":
    ut.main()
