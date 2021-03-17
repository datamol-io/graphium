from typing import List
from typing import Dict
from typing import Union
from typing import Any

import omegaconf

from goli.nn import FullDGLNetwork
from goli.nn import FullDGLSiameseNetwork


def load_datamodule():

    pass


def load_architecture(
    config: Union[omegaconf.DictConfig, Dict[str, Any]],
    in_dim_nodes: int,
    in_dim_edges: int,
):

    if isinstance(config, dict):
        config = omegaconf.OmegaConf.create(config)

    kwargs = {}

    # Select the architecture
    if config.architecture.model_type.lower() == "fulldglnetwork":
        model_class = FullDGLNetwork
    elif config.architecture.model_type.lower() == "fulldglsiamesenetwork":
        model_class = FullDGLSiameseNetwork
        kwargs["dist_method"] = config.architecture.model_type.dist_method
    else:
        raise ValueError(f"Unsupported model_type=`{config.architecture.model_type}`")

    # Prepare the various kwargs
    pre_nn_kwargs = dict(config.architecture.pre_nn)
    gnn_kwargs = dict(config.architecture.gnn)
    post_nn_kwargs = (
        dict(config.architecture.post_nn) if config.architecture.gnn.post_nn is not None else None
    )

    # Set the input dimensions
    if pre_nn_kwargs is not None:
        pre_nn_kwargs = dict(pre_nn_kwargs)
        pre_nn_kwargs.setdefault("in_dim", in_dim_nodes)
    else:
        gnn_kwargs.setdefault("in_dim", in_dim_nodes)

    gnn_kwargs.setdefault("in_dim_edges", in_dim_edges)

    # Build the graph network
    model = model_class(
        gnn_kwargs=gnn_kwargs,
        pre_nn_kwargs=pre_nn_kwargs,
        post_nn_kwargs=post_nn_kwargs,
        **kwargs,
    )

    # Dimensions sanity check
    assert model.in_dim == in_dim_nodes
    assert model.in_dim_edges == in_dim_edges

    return model
