import pandas as pd

import goli


def test_list_datasets():
    datasets = goli.data.utils.list_goli_datasets()
    assert isinstance(datasets, set)
    assert len(datasets) > 0


def test_download_datasets(tmpdir):
    dataset_dir = tmpdir.mkdir("goli-datasets")

    data_path = goli.data.utils.download_goli_dataset("goli-zinc-micro", output_path=dataset_dir)

    fpath = goli.utils.fs.join(data_path, "ZINC-micro.csv")
    df = pd.read_csv(fpath)
    assert df.shape == (1000, 4)  # type: ignore
    assert df.columns.tolist() == ["SMILES", "SA", "logp", "score"]  # type: ignore
