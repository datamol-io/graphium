from time import time
import torch
import yaml
import fsspec
import dgl
from torch.profiler import profile, record_function, ProfilerActivity

from goli.config._loader import (
    load_datamodule,
    load_metrics,
    load_predictor,
    load_architecture,
)


def timed_fulldgl_forward(self, g: dgl.DGLGraph, flip_pos_enc: str) -> torch.Tensor:
    # Get the node features and positional embedding

    t0 = time()
    h = g.ndata["feat"]
    if "pos_enc_feats_sign_flip" in g.ndata.keys():
        pos_enc = g.ndata["pos_enc_feats_sign_flip"]
        if flip_pos_enc == "random":
            rand_sign_shape = ([1] * (pos_enc.ndim - 1)) + [pos_enc.shape[-1]]
            rand_sign = torch.sign(torch.randn(rand_sign_shape, dtype=h.dtype, device=h.device))
            pos_enc = pos_enc * rand_sign
        elif flip_pos_enc == "no-flip":
            pass
        elif flip_pos_enc == "sign-flip":
            pos_enc = -pos_enc
        h = torch.cat((h, pos_enc), dim=-1)
    if "pos_enc_feats_no_flip" in g.ndata.keys():
        pos_enc = g.ndata["pos_enc_feats_no_flip"]
        h = torch.cat((h, pos_enc), dim=-1)

    g.ndata["h"] = h
    print("pos encoding: ", time() - t0)

    if "feat" in g.edata.keys():
        g.edata["e"] = g.edata["feat"]

    t0 = time()
    # Run the pre-processing network on node features
    if self.pre_nn is not None:
        h = g.ndata["h"]
        h = self.pre_nn.forward(h)
        g.ndata["h"] = h
    print("node features: ", time() - t0)

    t0 = time()
    # Run the pre-processing network on edge features
    # If there are no edges, skip the forward and change the dimension of e
    if self.pre_nn_edges is not None:
        e = g.edata["e"]
        if torch.prod(torch.as_tensor(e.shape[:-1])) == 0:
            e = torch.zeros(
                list(e.shape[:-1]) + [self.pre_nn_edges.out_dim],
                device=e.device,
                dtype=e.dtype,
            )
        else:
            e = self.pre_nn_edges.forward(e)
        g.edata["e"] = e
    print("edge features: ", time() - t0)

    t0 = time()
    # Run the graph neural network
    h = self.gnn.forward(g)
    print("GNN: ", time() - t0)

    t0 = time()
    # Run the output network
    if self.post_nn is not None:
        if self.concat_last_layers is None:
            h = self.post_nn.forward(h)
        else:
            # Concatenate the output of the last layers according to `self._concat_last_layers``.
            # Useful for generating fingerprints
            h = [h]
            for ii in range(len(self.post_nn.layers)):
                h.insert(0, self.post_nn.layers[ii].forward(h[0]))  # Append in reverse order
            h = torch.cat([h[ii] for ii in self._concat_last_layers], dim=-1)
    print("output nn: ", time() - t0)

    return h


def timed_dgllayer_forward(self, layer, g, h, e, step_idx) -> torch.Tensor:
    # Apply the GNN layer with the right inputs/outputs
    t0 = time()
    if layer.layer_inputs_edges and layer.layer_outputs_edges:
        h, e = layer(g=g, h=h, e=e)
    elif layer.layer_inputs_edges:
        h = layer(g=g, h=h, e=e)
    elif layer.layer_outputs_edges:
        h, e = layer(g=g, h=h)
    else:
        h = layer(g=g, h=h)
    print("gnn layer: ", time() - t0)

    t0 = time()
    # Apply the residual layers on the features and edges (if applicable)
    if step_idx < len(self.layers) - 1:
        h, h_prev = self.residual_layer.forward(h, h_prev, step_idx=step_idx)
        if (self.residual_edges_layer is not None) and (layer.layer_outputs_edges):
            e, e_prev = self.residual_edges_layer.forward(e, e_prev, step_idx=step_idx)
    print("residual: ", time() - t0)

    return h, e, h_prev, e_prev


def main():
    CONFIG_PATH = "profiling/configs_profiling.yaml"
    DATA_PATH = "https://storage.googleapis.com/goli-public/datasets/goli-zinc-bench-gnn/smiles_score.csv.gz"
    BATCH_SIZE = 1000
    NUM_FORWARD_LOOPS = 6
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    with fsspec.open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["datamodule"]["args"]
    cfg["datamodule"]["args"]["cache_data_path"] = "goli/data/cache/profiling/forward_data.cache"
    cfg["datamodule"]["args"]["df_path"] = DATA_PATH
    cfg["datamodule"]["args"]["sample_size"] = NUM_FORWARD_LOOPS * BATCH_SIZE
    cfg["datamodule"]["args"]["prepare_dict_or_graph"] = "dglgraph"

    datamodule = load_datamodule(cfg)
    datamodule.prepare_data()

    dglgraphs = [datamodule.dataset[ii]["features"] for ii in range(len(datamodule.dataset))]
    dglgraphs = [
        dgl.batch(dglgraphs[ii * (BATCH_SIZE + 1) : (ii + 1) * (BATCH_SIZE + 1)])
        for ii in range(NUM_FORWARD_LOOPS)
    ]
    for ii in range(NUM_FORWARD_LOOPS):
        dglgraphs[ii].ndata["feat"] = dglgraphs[ii].ndata["feat"].to(dtype=torch.float32)
        dglgraphs[ii].edata["feat"] = dglgraphs[ii].edata["feat"].to(dtype=torch.float32)
        dglgraphs[ii] = dglgraphs[ii].to(device=DEVICE)

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dim_nodes=datamodule.num_node_feats_with_positional_encoding,
        in_dim_edges=datamodule.num_edge_feats,
    )
    metrics = load_metrics(cfg)

    predictor = load_predictor(cfg, model_class, model_kwargs, metrics)
    model = predictor.model
    model = model.to(device=DEVICE)

    with torch.no_grad():
        for ii in range(NUM_FORWARD_LOOPS):
            print("---------------------------------")
            timed_fulldgl_forward(model, dglgraphs[ii], flip_pos_enc="random")
            print("---------------------------------")

    print("Done :)")


if __name__ == "__main__":
    main()
