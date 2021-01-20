from rdkit import Chem

ATOM_LIST = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
]
ATOM_NUM_H = [0, 1, 2, 3, 4]
IMPLICIT_VALENCE = [0, 1, 2, 3, 4, 5, 6]
CHARGE_LIST = [-3, -2, -1, 0, 1, 2, 3]
RADICAL_E_LIST = [0, 1, 2]
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.names[k]
    for k in sorted(Chem.rdchem.HybridizationType.names.keys(), reverse=True)
    if k != "OTHER"
]
ATOM_DEGREE_LIST = range(5)
CHIRALITY_LIST = ["R"]  # alternative is just S
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [0, 1, 2, 3, 4, 5]
