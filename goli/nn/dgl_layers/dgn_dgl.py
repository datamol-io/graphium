import torch
from typing import Dict, List, Tuple, Union, Callable
from functools import partial

from goli.nn.dgl_layers.pna_dgl import PNAConvolutionalDgl, PNAMessagePassingDgl
from goli.nn.pna_operations import PNA_DGL_AGGREGATORS
from goli.nn.dgn_operations import DGN_AGGREGATORS


class BaseDGNDgl:
    def parse_aggregators(self, aggregators_name: List[str]) -> List[Callable]:
        r"""
        Parse the aggregators from a list of strings into a list of callables.

        The possibilities are:

        - `"mean"`
        - `"sum"`
        - `"min"`
        - `"max"`
        - `"std"`
        - `"dir{dir_idx:int}/smooth/{Optional[temperature:float]}"`
        - `"dir{dir_idx:int}/dx_abs/{Optional[temperature:float]}"`
        - `"dir{dir_idx:int}/dx_no_abs/{Optional[temperature:float]}"`
        - `"dir{dir_idx:int}/dx_abs_balanced/{Optional[temperature:float]}"`
        - `"dir{dir_idx:int}/forward/{Optional[temperature:float]}"`
        - `"dir{dir_idx:int}/backward/{Optional[temperature:float]}"`

        `dir_idx` is an integer specifying the index of the positional encoding
        to use for direction. In the case of eigenvector-based directions, `dir_idx=1`
        is chosen for the first non-trivial eigenvector and `dir_idx=2` for the second.

        `temperature` is used to harden the direction using a softmax on the directional
        matrices. If it is not provided, then no softmax is applied. The larger the temperature,
        the more weight is attributed to the dominant direction.

        The graph. Must have the key `graph.ndata["pos_dir"]`

        Example:
            ```
            In:     self.parse_aggregators(["dir1/dx_abs", "dir2/smooth/0.2"])
            Out:    [partial(aggregate_dir_dx_abs, dir_idx=1, temperature=None),
                     partial(aggregate_dir_smooth, dir_idx=2, temperature=0.2)]
            ```

        Parameters:
            aggregators_name: The list of all aggregators names to use, selected
                from the list of possible strings.

        Returns:
            aggregators: The list of all callable aggregators.

        """
        aggregators = []

        for agg_name in aggregators_name:
            agg_name = agg_name.lower()
            this_agg = None

            # Get the aggregator from PNA if not a directional aggregation
            if agg_name in PNA_DGL_AGGREGATORS.keys():
                this_agg = PNA_DGL_AGGREGATORS[agg_name]

            # If the directional, get the right aggregator
            elif "dir" == agg_name[:3]:
                agg_split = agg_name.split("/")
                agg_dir, agg_fn_name = agg_split[0], agg_split[1]
                dir_idx = int(agg_dir[3:])
                temperature = None
                if len(agg_split) > 2:
                    temperature = float(agg_split[2])
                this_agg = partial(DGN_AGGREGATORS[agg_fn_name], dir_idx=dir_idx, temperature=temperature)

            if this_agg is None:
                raise ValueError(f"aggregator `{agg_name}` not a valid choice.")

            aggregators.append(this_agg)

        return aggregators

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        The message function to generate messages along the edges.
        """
        return {
            "edge_attr": edges.data["edge_attr"],
            "source_pos": edges.data["source_pos"],
            "dest_pos": edges.data["dest_pos"],
        }

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        r"""
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data["h"]
        h = nodes.mailbox["edge_attr"]
        source_pos = nodes.mailbox["source_pos"]
        dest_pos = nodes.mailbox["dest_pos"]
        D = h.shape[-2]

        # aggregators and scalers
        h_to_cat = [
            aggr(h=h, h_in=h_in, source_pos=source_pos, dest_pos=dest_pos) for aggr in self.aggregators
        ]
        h = torch.cat(h_to_cat, dim=-1)
        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)

        return {"h": h}


class DGNConvolutionalDgl(BaseDGNDgl, PNAConvolutionalDgl):
    r"""
    Implementation of the convolutional architecture of the DGN layer,
    previously known as `DGNSimpleLayer`. This layer aggregates the
    neighbouring messages using multiple aggregators and scalers,
    concatenates their results, then applies an MLP on the concatenated
    features.

    The graph. Must have the key `graph.ndata["pos_dir"]`

    DGN: Directional Graph Networks
    Dominique Beaini, Saro Passaro, Vincent Létourneau, William L. Hamilton, Gabriele Corso, Pietro Liò
    https://arxiv.org/pdf/2010.02863.pdf
    """

    def _parse_aggregators(self, aggregators: List[str]) -> List[Callable]:
        return BaseDGNDgl.parse_aggregators(self, aggregators)

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        return BaseDGNDgl.message_func(self, edges)

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        return BaseDGNDgl.reduce_func(self, nodes)

    def pretrans_edges(self, edges):
        pretrans = PNAConvolutionalDgl.pretrans_edges(self, edges)
        if not ("pos_dir" in edges.src.keys()):
            raise KeyError("`pos_dir` key missing. Positional encodings are required for the DGN layer, make sure you add them under the `dglGraph.ndata['pos_dir']` key.")
        pretrans.update({"source_pos": edges.src["pos_dir"], "dest_pos": edges.dst["pos_dir"]})
        return pretrans


class DGNMessagePassingDgl(BaseDGNDgl, PNAMessagePassingDgl):
    r"""
    Implementation of the message passing architecture of the DGN message passing layer,
    previously known as `DGNLayerComplex`. This layer applies an MLP as
    pretransformation to the concatenation of $[h_u, h_v, e_{uv}]$ to generate
    the messages, with $h_u$ the node feature, $h_v$ the neighbour node features,
    and $e_{uv}$ the edge feature between the nodes $u$ and $v$.

    After the pre-transformation, it aggregates the messages using
    multiple aggregators and scalers,
    concatenates their results, then applies an MLP on the concatenated
    features.

    The graph. Must have the key `graph.ndata["pos_dir"]`

    DGN: Directional Graph Networks
    Dominique Beaini, Saro Passaro, Vincent Létourneau, William L. Hamilton, Gabriele Corso, Pietro Liò
    https://arxiv.org/pdf/2010.02863.pdf
    """

    def _parse_aggregators(self, aggregators: List[str]) -> List[Callable]:
        return BaseDGNDgl.parse_aggregators(self, aggregators)

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        return BaseDGNDgl.message_func(self, edges)

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        return BaseDGNDgl.reduce_func(self, nodes)

    def pretrans_edges(self, edges):
        pretrans = PNAMessagePassingDgl.pretrans_edges(self, edges)
        if not ("pos_dir" in edges.src.keys()):
            raise KeyError("`pos_dir` key missing. Positional encodings are required for the DGN layer, make sure you add them under the `dglGraph.ndata['pos_dir']` key.")
        pretrans.update({"source_pos": edges.src["pos_dir"], "dest_pos": edges.dst["pos_dir"]})
        return pretrans
