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
from goli.mol_utils.properties import get_atom_features, get_edge_features
