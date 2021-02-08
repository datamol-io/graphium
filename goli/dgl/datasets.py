import os
import torch
import joblib
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from itertools import compress

from goli.commons.utils import to_tensor, is_device_cuda
from goli.mol_utils.smiles_parser import filter_bad_smiles
from goli.data_parser.data_parser import read_file
from goli.commons.arg_checker import check_arg_iterator


def load_csv_to_dgl_dataset(data_dir, name, smiles_cols, y_col, max_mols, trans, device):

    # Get the number of molecules to parse from the CSV file
    csv_name = os.path.join(data_dir, f"{name}.csv")
    if max_mols is None:
        num_mols = None
    elif isinstance(max_mols, str):
        if max_mols.lower() in ["null", "none", ""]:
            num_mols = None
        else:
            raise ValueError(f"Unsupported max_mols=`{max_mols}`")
    else:
        num_mols = min(max_mols, pd.read_csv(csv_name, index_col=0).shape[0])

    # If there is already a pickle file, read it. Otherwise, read the CSV file
    pkl_name = os.path.join(data_dir, f"dataset_{name}_{num_mols}.pkl")
    if os.path.isfile(pkl_name):
        print(f'Reading file :"{pkl_name}"')
        dt = joblib.load(pkl_name)
        print(f"File read successfully")
    else:
        # Read the CSV file, remove bad smiles, and remove nan values
        smiles_cols = check_arg_iterator(smiles_cols, enforce_type=list)
        y_col = check_arg_iterator(y_col, enforce_type=list)
        df = read_file(csv_name, nrows=num_mols)
        for col in smiles_cols:
            df = filter_bad_smiles(dataframe=df, smiles_cols=col, sanitize=True)
        nans = np.isnan(df[y_col].values)
        nans = nans.max(axis=1)
        df = df[~nans]
        y = df[y_col].values

        # Transform the SMILES into DGL graphs
        invalid_ids = np.zeros(df.shape[0], dtype=bool)
        X_list = []
        for col in smiles_cols:
            print(f"Transforming SMILES column `{col}` into DGL graphs")
            X = trans.transform(df[col].values)
            X = trans.to(X, dtype=torch.float32, device=device)

            # print(f'Adding edge weights of 1 for SMILES column `{col}`')
            # for x in X:
            #     x.edata['w'] = torch.ones((x.number_of_edges(), 1), dtype=float)

            X_list.append(X)
            invalid_ids |= np.array([x is None for x in X])

        y_filtered = y[~invalid_ids, :].astype(np.float32)
        X_list_filtered = [list(compress(this_X, ~invalid_ids)) for this_X in X_list]

        print(f"Adding DGL graphs to DGLMultiDataset")
        dt = DGLMultiDataset(X_list_filtered, y_filtered, device=device)

        # Save the DGL graphs into a pickle file
        print(f'Saving file :"{pkl_name}"')
        joblib.dump(dt, pkl_name)
        print("file saved successfully")

    return dt


class DGLMultiDataset(Dataset):
    """
    Dataset class for models that use DGL.

    Parameters:
        X_list: List of all featurized data returned by the DGLTransformer, given by a triple:
            [(graph 1, atom features 1 and bond features 1), (graph 2, atom features 2 and bond features 2), ...]
            You can also use a list of dgl object with features already set
        y: array
            output matrix that contains the readout for each graph.
        w: array
            sample or prediction weights to be used in a cost function
        device: torch.device
            The device on which to run the computation

    Attributes:
        G: list(tuple(dgl.DGLGraph, dgl.DGLGraph))
            A dgl object that contains the graph and the atom features
        edata: array
            An array with bond features, can be None.
        ndata: array
            An array with atom features, can be None.

    """

    def __init__(self, X_list, y, w=None, device="cpu", e_size=14):

        G_list = []
        for X in X_list:
            G_list.append(self._parse_X(X, device=device, e_size=e_size))
        self.G = list(zip(*G_list))

        self.y = to_tensor(y, device=device, dtype=torch.float32)
        self.w = w
        if self.w is not None:
            self.w = to_tensor(self.w, gpu=device, dtype=torch.float32)
        self.device = device

    @property
    def is_cuda(self):
        return is_device_cuda(self.device)

    def __len__(self):
        return len(self.G)

    def __getitem__(self, idx):
        if self.w is not None:
            return self.G[idx], self.y[idx], self.w[idx]
        return self.G[idx], self.y[idx]

    def to(self, device):
        self.y = self.y.to(device)
        self.G = [(sub_g.to(device) for sub_g in g) for g in self.G]
        if self.w is not None:
            self.w = self.w.to(device)

    def _parse_X(self, X, device, e_size):
        ndata = None
        edata = None
        if isinstance(X[0], (tuple, list)):
            G, *data = zip(*X)
            if len(data) > 0:
                data = iter(data)
                ndata = next(data, None)
                edata = next(data, None)
        else:
            # received only the graph, not ndata and edata:
            G = X

        G = list(G)
        for k, g in enumerate(G):
            if ndata:
                g.ndata["h"] = to_tensor(ndata[k], device=device)
            if edata:
                g.edata["e"] = to_tensor(edata[k], device=device)
                was_empty = g.number_of_edges() == 0
                if was_empty:
                    g.edata["e"] = torch.zeros(g.edata["e"].shape[0], e_size)

        return G
