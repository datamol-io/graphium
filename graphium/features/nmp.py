from typing import Tuple, Optional, Dict, Union
import importlib.resources
from copy import deepcopy
import pandas as pd
import numpy as np
import math
from rdkit import Chem

# NOTE(hadim): usually it's best to embed this in a function.
with importlib.resources.open_text("graphium.features", "periodic_table.csv") as f:
    PERIODIC_TABLE = pd.read_csv(f)
PERIODIC_TABLE = PERIODIC_TABLE.set_index("AtomicNumber")


# Small function to convert strings to floats
def float_or_none(string: str) -> Union[float, None]:
    """
    check if a string can be converted to float, return none if it can't
    Parameters:
        string: str
    Returns:
        val: float or None
    """
    try:
        val = float(string)
    except:
        val = None
    return val


# It's much faster to index from a list than a DataFrame, which can have big impact
# when featurizing millions of molecules
BOND_RADIUS_SINGLE = [float_or_none(elem) for elem in PERIODIC_TABLE["SingleBondRadius"]]
BOND_RADIUS_DOUBLE = [float_or_none(elem) for elem in PERIODIC_TABLE["DoubleBondRadius"]]
BOND_RADIUS_TRIPLE = [float_or_none(elem) for elem in PERIODIC_TABLE["TripleBondRadius"]]
ELECTRONEGATIVITY = [float_or_none(elem) for elem in PERIODIC_TABLE["Electronegativity"]]
FIRST_IONIZATION = [float_or_none(elem) for elem in PERIODIC_TABLE["FirstIonization"]]
MELTING_POINT = [float_or_none(elem) for elem in PERIODIC_TABLE["MeltingPoint"]]
METAL = (2 * (PERIODIC_TABLE["Metal"] == "yes") + (PERIODIC_TABLE["Metalloid"] == "yes")).tolist()

PHASE = list(PERIODIC_TABLE["Phase"].values)
PHASE_SET = list(set(PHASE))
TYPE = list(deepcopy(PERIODIC_TABLE["Type"]))
TYPE = ["none" if (isinstance(t, float) and math.isnan(t)) else t for t in TYPE]
TYPE_SET = list(set(TYPE))
GROUP = deepcopy(PERIODIC_TABLE["Group"].values)
GROUP[np.isnan(GROUP)] = 19
GROUP_SET = list(set(GROUP))
PERIOD = list(PERIODIC_TABLE["Period"].values)
PERIOD_SET = list(set(PERIOD))

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
VALENCE = [0, 1, 2, 3, 4, 5, 6]
CHARGE_LIST = [-3, -2, -1, 0, 1, 2, 3]
RADICAL_E_LIST = [0, 1, 2]
ATOM_DEGREE_LIST = [0, 1, 2, 3, 4]

HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.names[k]
    for k in sorted(Chem.rdchem.HybridizationType.names.keys(), reverse=True)
    if k != "OTHER"
]


CHIRALITY_LIST = ["R"]  # alternative is just S


BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
