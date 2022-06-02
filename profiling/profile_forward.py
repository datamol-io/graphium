from time import time
import torch
import yaml
import fsspec
import dgl


from goli.config._loader import load_datamodule, load_metrics, load_predictor, load_architecture


def main():
    CONFIG_PATH = "configs_profiling.yaml"
    DATA_PATH = "https://storage.googleapis.com/goli-public/datasets/goli-zinc-bench-gnn/smiles_score.csv.gz"
    BATCH_SIZE = 1000
    NUM_FORWARD_LOOPS = 30

    with fsspec.open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["datamodule"]["args"]
    cfg["datamodule"]["args"]["cache_data_path"] = "goli/data/cache/profiling/forward_data.cache"
    cfg["datamodule"]["args"]["df_path"] = DATA_PATH
    cfg["datamodule"]["args"]["sample_size"] = BATCH_SIZE
    cfg["datamodule"]["args"]["prepare_dict_or_graph"] = "dglgraph"

    datamodule = load_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    dglgraphs = [datamodule.train_ds[ii]["features"] for ii in range(len(datamodule.train_ds))]
    dglgraphs = dgl.batch(dglgraphs)
    dglgraphs.ndata["feat"] = dglgraphs.ndata["feat"].to(dtype=torch.float32)
    dglgraphs.edata["feat"] = dglgraphs.edata["feat"].to(dtype=torch.float32)

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dim_nodes=datamodule.num_node_feats_with_positional_encoding,
        in_dim_edges=datamodule.num_edge_feats,
    )
    metrics = load_metrics(cfg)

    predictor = load_predictor(cfg, model_class, model_kwargs, metrics)
    model = predictor.model

    start = time()
    for ii in range(NUM_FORWARD_LOOPS):
        print(ii)
        model._forward(dglgraphs, flip_pos_enc="random")
    print(f"Time to {NUM_FORWARD_LOOPS} forward loops: {time() - start} s")

    print("Done :)")


if __name__ == "__main__":
    main()
