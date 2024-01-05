import torch
from tqdm import tqdm
import datamol as dm

input_features = torch.load("input_features.pt")
batch_size = 100        

all_results = []

for i, index in tqdm(enumerate(range(0, len(input_features), batch_size))):

    results = torch.load(f'results/res-{i:04}.pt')
    all_results.extend(results)

del input_features

torch.save(all_results, 'results/all_results.pt')

smiles_to_process = torch.load("saved_admet_smiles.pt")

# Generate dictionary SMILES -> fingerprint vector
smiles_to_fingerprint = dict(zip(smiles_to_process, results))
torch.save(smiles_to_fingerprint, "results/smiles_to_fingerprint.pt")

# Generate dictionary unique IDs -> fingerprint vector
ids = [dm.unique_id(smiles) for smiles in smiles_to_process]
ids_to_fingerprint = dict(zip(ids, results))
torch.save(ids_to_fingerprint, "results/ids_to_fingerprint.pt")
