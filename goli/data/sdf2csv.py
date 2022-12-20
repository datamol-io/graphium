import datamol as dm
import zipfile
import pandas as pd
from rdkit import Chem
from ogb.lsc import PCQM4Mv2Dataset
from goli.data.datamodule import BaseDataModule
import csv

#*extract sdf from zip
def extract_zip(fname):
    zf = zipfile.ZipFile(fname)
    zf.extractall(".")



#* load sdf into mols  
def extract_mols_from_sdf(fname):
    mol_df = BaseDataModule._read_sdf(fname)
    mols = mol_df['_rdkit_molecule_obj']
    #smiles = mol_df['smiles']
    #dm.unique_id(mol) #checking the unique id of mols
    return mols



def mols2cxs(mols):
    #* convert into a smiles that contains the 3D structure
    cxs = []
    for mol in mols:
        cxs.append(dm.to_smiles(mol, cxsmiles=True))
    return cxs

def write_csv(cxs, homos, fname):
    outname = fname + ".csv"
    fieldnames = ['cxsmiles', 'homo_lumo_gap']

    with open(outname, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(cxs)):
            writer.writerow({'cxsmiles': cxs[i], 'homo_lumo_gap': homos[i]})




'''
function converting sdf file molecules into csv format using CXSmiles
fname: input file name * for the *.sdf.zip file
'''
def sdf2csv(fname):
    
    #* extract zip file 
    # extract_zip(fname+".sdf.zip")

    mols= extract_mols_from_sdf(fname + ".sdf")
    
    #*download ogb
    dataset = PCQM4Mv2Dataset(root = ".", only_smiles = True)  #(smiles, homo_lomo)
    homos = []
    for i in range(len(mols)):
        homo = dataset[i][1]
        homos.append(homo)

    cxs = mols2cxs(mols)
    write_csv(cxs, homos, fname)

'''
#! this script need to be located at the specific data folder as it uses relative dependencies
for example   #* goli/data/PCQM4Mv2
'''
if __name__ == "__main__":
    fname = "pcqm4m-v2-train" #"pcqm4m-v2-train-mini"
    sdf2csv(fname)










