import pandas as pd

import graphpumhium


def test_list_datasets():
    datasets = graphpumhium.data.utils.liraph_umgraphium_datasets()
    assert isinstance(datasets, set)
    assert len(datasets) > 0


def test_download_datasets(tmpdir):
    dataset_dir = tmpdir.mkdir("graphpumhium-datasets")

    data_path = graphpumhium.data.utils.downlgraphiumgraphium_dgraphiumet(
        "graphium-zinc-micro", output_path=dataset_dir
    )

    fpath = graphium.utils.fs.join(data_path, "ZINC-micro.csv")
    df = pd.read_csv(fpath)
    assert df.shape == (1000, 4)  # type: ignore
    assert df.columns.tolist() == ["SMILES", "SA", "logp", "score"]  # type: ignore
