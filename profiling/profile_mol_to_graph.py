from tqdm import tqdm
import datamol as dm
import pickle

from goli.data.utils import load_micro_zinc
from goli.features.featurizer import mol_to_dglgraph, mol_to_adj_and_features

def main():
    df = load_micro_zinc()
    smiles = df["SMILES"].values.tolist()
    smiles = smiles * 1
    print("Num smiles: ", len(smiles))

    pos_enc = {
        "pos_type": "laplacian_eigvec",
        "num_pos": 3,
        "normalization": "none",
        "disconnected_comp": True,
    }
    atom_property_list_float1 = ["mass", "in-ring", "hybridization", "chirality", "aromatic",
                                 "degree", "formal-charge", "single-bond", "double-bond", "radical-electron",
                                 ]
    atom_property_list_float2 = ["electronegativity",
                                 "vdw-radius", "covalent-radius", "metal"]

    featurizer = {
        "atom_property_list_onehot": ["atomic-number", "valence"],
        "atom_property_list_float": atom_property_list_float1 + atom_property_list_float2,
        "edge_property_list": ["bond-type-onehot", "bond-type-float", "stereo", "in-ring", "conjugated", "estimated-bond-length"],
        "add_self_loop": False,
        "explicit_H": False,
        "use_bonds_weights": False,
        "pos_encoding_as_features": pos_enc,
        "pos_encoding_as_directions": pos_enc,
        }

    graphs = []
    for s in tqdm(smiles):
        mol = dm.to_mol(s)
        graphs.append(mol_to_dglgraph(mol, **featurizer))

    print(graphs[0])

if __name__ == "__main__":
    main()