import datamol as dm
import zipfile
import pandas as pd
from rdkit import Chem
from ogb.lsc import PCQM4Mv2Dataset
import graphium
from graphium.data.datamodule import BaseDataModule
import csv


def extract_zip(fname):
    """
    #extract sdf from zip
    """
    zf = zipfile.ZipFile(fname)
    zf.extractall(".")


def extract_mols_from_sdf(fname):
    """
    load sdf into mols
    """
    mol_df = BaseDataModule._read_sdf(fname)
    mols = mol_df["_rdkit_molecule_obj"]
    return mols


def mols2cxs(mols):
    """
    convert into a smiles that contains the 3D structure
    """
    cxs = []
    for mol in mols:
        cxs.append(dm.to_smiles(mol, cxsmiles=True))
    return cxs


def write_csv(cxs: any, homos: any, fname: str):
    """
    write cxsmiles and homo lomo to file
    write the training molecules with cxsmiles first
    """
    outname = fname + ".csv"
    fieldnames = ["cxsmiles", "homo_lumo_gap"]

    with open(outname, "w") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(cxs)):
            writer.writerow({"cxsmiles": cxs[i], "homo_lumo_gap": homos[i]})


def sdf2csv(sdf_name: str = "pcqm4m-v2-train", outname: str = "pcqm4m-v2-train"):
    """
    function converting sdf file molecules into csv format using CXSmiles and combine with mols without 3d positions from ogb
    Parameters:
        sdf_name: name to the extracted sdf file
        outname: output name of the cxsmile file
    """
    mols = extract_mols_from_sdf(sdf_name + ".sdf")

    # download ogb smiles
    dataset = PCQM4Mv2Dataset(root=".", only_smiles=True)  # (smiles, homo_lomo)
    homos = []
    for i in range(len(mols)):
        homo = dataset[i][1]
        homos.append(homo)

    # write the trainning set molecules first with cxsmiles
    cxs = mols2cxs(mols)
    for j in range(len(mols), len(dataset)):
        cxs.append(dataset[j][0])
        homos.append(dataset[j][1])

    write_csv(cxs, homos, outname)


if __name__ == "__main__":
    """
    #* main function
    #! this script need to be located at the specific data folder as it uses relative dependencies
    for example   #* graphium/data/PCQM4Mv2

    instruction on how to generate the csv file:
    1. download the extract the sdf file from ogb: https://ogb.stanford.edu/docs/lsc/pcqm4mv2/
    $ wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz
    $ md5sum pcqm4m-v2-train.sdf.tar.gz
    $ tar -xf pcqm4m-v2-train.sdf.tar.gz
    2. run this function (the smiles csv file will be directed downloaded in code)
    """
    sdf_name = "pcqm4m-v2-train"
    outname = "pcqm4m-v2-train"
    sdf2csv(sdf_name=sdf_name, outname=outname)

    #! check how many warning you get from loading cxsmiles
    # path = "pcqm4m-v2-train.csv"
    # df = pd.read_csv(path)
    # smiles = df["cxsmiles"]
    # print (smiles[0])
    # graphium.data.datamodule.smiles_to_unique_mol_ids(smiles)
