from typing import Iterable, Dict, Any, Optional
from torch_geometric.data import Batch

# Misc imports
import inspect
from copy import deepcopy

# Torch imports
from torch import Tensor, nn
import torch

from goli.nn.encoders import (
    laplace_pos_encoder,
    mlp_encoder,
    signnet_pos_encoder,
    gaussian_kernel_pos_encoder,
)

PE_ENCODERS_DICT = {
    "laplacian_pe": laplace_pos_encoder.LapPENodeEncoder,
    "mlp": mlp_encoder.MLPEncoder,
    "signnet": signnet_pos_encoder.SignNetNodeEncoder,
    "gaussian_kernel": gaussian_kernel_pos_encoder.GaussianKernelPosEncoder,
}


class EncoderManager(nn.Module):
    def __init__(
        self,
        out_dim: int,
        pool: str = "sum",
        last_norm: str = "none",
        in_dims: Dict[str, int] = None,
        max_num_nodes_per_graph: Optional[int] = None,
        encoders: Optional[Dict[str, Any]] = None,
        name: str = "Encoder_Manager",
    ):
        r"""
        Class that allows to runs multiple encoders in parallel and concatenate / pool their outputs.
        Parameters:

            out_dim: Output dimension of the network

            pool: Pooling method to use. Can be one of: "sum", "mean", "max"

            last_norm: Normalization method to use on the output of the network. Can be one of: "batch_norm", "layer_norm", "none"

            in_dims: Dictionary of input dimensions for each input key.

            max_num_nodes_per_graph: Maximum number of nodes per graph. Used for positional encoders that require it.

            encoders:
                key-word arguments to use for the initialization of all positional encoding encoders
            name:
                Name attributed to the current network, for display and printing
                purposes.
        """

        super().__init__()
        self.out_dim = out_dim
        self.pool = pool
        self.last_norm = last_norm
        self.in_dims = in_dims
        self.name = name
        self.max_num_nodes_per_graph = max_num_nodes_per_graph
        if encoders is not None:
            max_nodes = encoders.pop("max_num_nodes_per_graph", None)
            if max_nodes is not None:
                if self.max_num_nodes_per_graph is not None:
                    assert (
                        self.max_num_nodes_per_graph == max_nodes
                    ), f"max_num_nodes_per_graph mismatch {self.max_num_nodes_per_graph}!={max_nodes}"
                self.max_num_nodes_per_graph = max_nodes

        self.pe_encoders_kwargs = deepcopy(encoders)
        self.pe_encoders = self._initialize_positional_encoders(encoders)

    def _initialize_positional_encoders(self, pe_encoders_kwargs: Dict[str, Any]) -> Optional[nn.ModuleDict]:
        r"""Initialize the positional encoders for each positional/structural encodings.
        Parameters:

            pe_encoders_kwargs: key-word arguments to use for the initialization of all positional encoding encoders

        Returns:
            pe_encoders: a nn.ModuleDict containing all positional encoders specified by encoder_name in pe_encoders_kwargs["encoders"]
        """
        # TODO: Currently only supports PE/SE on the nodes. Need to add edges.

        if (pe_encoders_kwargs is None) or (len(pe_encoders_kwargs) == 0):
            return

        pe_encoders = nn.ModuleDict()

        # Pooling options here for pe encoders
        self.pe_pool = pe_encoders_kwargs["pool"]
        pe_out_dim = pe_encoders_kwargs["out_dim"]
        in_dim_dict = pe_encoders_kwargs["in_dims"]

        # Loop every positional encoding to assign it
        for encoder_name, encoder_kwargs in pe_encoders_kwargs["encoders"].items():
            encoder_kwargs = deepcopy(encoder_kwargs)
            encoder_type = encoder_kwargs.pop("encoder_type")
            encoder = PE_ENCODERS_DICT[encoder_type]

            # Get the keys associated to in_dim. First check if there's a key that starts with `encoder_name/`
            # Then check for the exact key
            this_in_dims = {}
            for key, dim in in_dim_dict.items():
                if isinstance(key, str) and key.startswith(f"{encoder_name}/"):
                    key_name = "in_dim_" + key[len(encoder_name) + 1 :]
                    this_in_dims[key_name] = dim
            if len(this_in_dims) == 0:
                for key in encoder_kwargs.get("input_keys", []):
                    if key in in_dim_dict:
                        this_in_dims[key] = in_dim_dict[key]
                    else:
                        raise ValueError(
                            f"Key '{key}' not found in `in_dim_dict`. Encoder '{encoder_name}/' is also not found.\n Available keys: {in_dim_dict.keys()}"
                        )

            # Parse the in_dims based on Encoder's signature
            accepted_keys = inspect.signature(encoder).parameters.keys()
            if all([key in accepted_keys for key in this_in_dims.keys()]):
                pass
            elif "in_dim" in accepted_keys:
                if len(set(this_in_dims.values())) == 1:
                    this_in_dims = {"in_dim": list(this_in_dims.values())[0]}
                else:
                    raise ValueError(
                        f"All `in_dims` must be equal for encoder {encoder_name}. Provided: {this_in_dims}"
                    )
            else:
                raise ValueError(
                    f"`in_dim` not understood for encoder {encoder_name}. Provided: {this_in_dims}. Accepted keys are: {accepted_keys}"
                )

            # Add the max_num_nodes_per_graph if it's in the accepted input keys
            if "max_num_nodes_per_graph" in accepted_keys:
                encoder_kwargs["max_num_nodes_per_graph"] = self.max_num_nodes_per_graph

                # Initialize the pe_encoder layer
                pe_out_dim = encoder_kwargs.pop("out_dim", None)
                if pe_out_dim is None:
                    pe_out_dim = self.out_dim
                if self.out_dim is not None:
                    assert pe_out_dim == self.out_dim, f"values mismatch {pe_out_dim}!={self.out_dim}"
                pe_encoders[encoder_name] = encoder(out_dim=pe_out_dim, **this_in_dims, **encoder_kwargs)

        return pe_encoders

    def forward(self, g: Batch) -> Batch:
        r"""
        forward pass of the pe encoders and pooling

        Parameters:
            g:
                ptg Batch on which the convolution is done.
                Must contain the following elements:

                - Node key `"feat"`: `torch.Tensor[..., N, Din]`.
                  Input node feature tensor, before the network.
                  `N` is the number of nodes, `Din` is the input features dimension ``self.pre_nn.in_dim``

                - Edge key `"edge_feat"`: `torch.Tensor[..., N, Ein]` **Optional**.
                  The edge features to use. It will be ignored if the
                  model doesn't supporte edge features or if
                  `self.in_dim_edges==0`.

                - Other keys related to positional encodings `"pos_enc_feats_sign_flip"`,
                  `"pos_enc_feats_no_flip"`.

        Returns:
            g:
                pyg Batch with the positional encodings added to the graph
        """
        # Apply the positional encoders
        pe_pooled = self.forward_positional_encoding(g)

        # Add the processed positional encodings to the graphs.
        # If the key is already present, concatenate the pe_pooled to the pre-existing feature.
        for pe_key, this_pe in pe_pooled.items():
            feat = this_pe
            if pe_key in g.keys:
                feat = torch.cat((feat, g[pe_key]), dim=-1)
            g[pe_key] = feat
        return g

    def forward_positional_encoding(self, g: Batch) -> Dict[str, Tensor]:
        """
        Forward pass for the positional encodings (PE),
        with each PE having it's own encoder defined in `self.pe_encoders`.
        All the positional encodings with the same keys are pooled together
        using `self.pe_pooling`.

        Parameters:
            g: pyg Batch containing the node positional encodings

        Returns:
            pe_node_pooled: The positional / structural encodings go through
            encoders, then are pooled together according to their keys.

        """

        # Return None if no positional encoders
        if (self.pe_encoders is None) or len(self.pe_encoders) == 0:
            return {}

        encoder_outs = []
        # Run every node positional-encoder
        for encoder_name, encoder in self.pe_encoders.items():
            encoder_outs.append(encoder(g, key_prefix=encoder_name))

        # list of dict to dict of list, with concatenation of the tensors
        pe_cats = {
            key: torch.stack([d[key] for d in encoder_outs if key in d], dim=-1)
            for key in set().union(*encoder_outs)
        }

        # Pool the node positional encodings
        pe_pooled = {}
        for key, pe_cat in pe_cats.items():
            pe_pooled[key] = self.forward_simple_pooling(pe_cat, pooling=self.pool, dim=-1)

        return pe_pooled

    def forward_simple_pooling(self, h: Tensor, pooling: str, dim: int) -> Tensor:
        """
        Apply sum, mean, or max pooling on a Tensor.
        Parameters:
            h: the Tensor to pool
            pooling: string specifiying the pooling method
            dim: the dimension to pool over

        Returns:
            pooled: the pooled Tensor
        """

        if pooling == "sum":
            pooled = torch.sum(h, dim=dim)
        elif pooling == "mean":
            pooled = torch.mean(h, dim=dim)
        elif pooling == "max":
            pooled = torch.max(h, dim=dim).values
        else:
            raise Exception(f"Pooling method `{pooling}` is not defined")
        return pooled

    def make_mup_base_kwargs(self, divide_factor: float = 2.0) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.

        Returns:
            pe_kw: the model kwargs where the dimensions are divided by the factor
        """

        # Create the base model kwargs
        base_kwargs = {
            "out_dim": round(self.out_dim / divide_factor),
            "pool": self.pool,
            "last_norm": self.last_norm,
            "in_dims": self.in_dims,
            "max_num_nodes_per_graph": self.max_num_nodes_per_graph,
            "name": self.name,
        }

        # For the pe-encoders, don't factor the in_dim and in_dim_edges
        pe_kw = deepcopy(self.pe_encoders_kwargs)
        if self.pe_encoders_kwargs is not None:
            new_pe_kw = {
                key: encoder.make_mup_base_kwargs(divide_factor=divide_factor, factor_in_dim=False)
                for key, encoder in self.pe_encoders.items()
            }
            for key, enc in pe_kw["encoders"].items():
                new_pe_kw[key].pop("in_dim", None)
                new_pe_kw[key].pop("in_dim_edges", None)
                enc.update(new_pe_kw[key])

        base_kwargs["encoders"] = pe_kw

        return base_kwargs

    @property
    def input_keys(self) -> Iterable[str]:
        r"""
        Returns the input keys for all pe-encoders

        Returns:
            input_keys: the input keys for all pe-encoders
        """
        if self.pe_encoders is not None:
            return self.pe_encoders_kwargs["input_keys"]
        else:
            raise ValueError("pe_encoders is not initialized, so there are no input keys.")

    @property
    def in_dims(self) -> Iterable[int]:
        r"""
        Returns the input dimensions for all pe-encoders

        Returns:
            in_dims: the input dimensions for all pe-encoders
        """
        if self.pe_encoders is not None:
            return self.pe_encoders_kwargs["in_dims"]
        else:
            raise ValueError("pe_encoders is not initialized, so there are no input dimensions.")

    @property
    def out_dim(self) -> int:
        r"""
        Returns the output dimension of the pooled embedding from all the pe encoders

        Returns:
            out_dim: the output dimension of the pooled embedding from all the pe encoders
        """
        if self.pe_encoders is not None:
            return self.pe_encoders_kwargs["out_dim"]
        else:
            raise ValueError("pe_encoders is not initialized, so there is no output dimension.")
