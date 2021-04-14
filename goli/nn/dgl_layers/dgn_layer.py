import torch
from typing import Dict, List, Tuple, Union, Callable
from functools import partial

from goli.nn.dgl_layers.pna_layer import PNAConvolutionalLayer, PNAMessagePassingLayer
from goli.nn.pna_operations import PNA_AGGREGATORS
import goli.nn.dgn_operations as dgn_ops


class BaseDGNLayer:
    def parse_aggregators(self, aggregators_name: List[str]) -> List[Callable]:
        r"""
        Parse the aggregators from a list of strings into a list of callables.

        The possibilities are:

        - `"mean"`
        - `"sum"`
        - `"min"`
        - `"max"`
        - `"std"`
        - `"dir{dir_idx:int}/smooth"`
        - `"dir{dir_idx:int}/softmax/{alpha:float}"`
        - `"dir{dir_idx:int}/dx_abs"`
        - `"dir{dir_idx:int}/dx_no_abs"`
        - `"dir{dir_idx:int}/dx_abs_balanced"`
        - `"dir{dir_idx:int}/forward"`
        - `"dir{dir_idx:int}/backward"`

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
            if agg_name in PNA_AGGREGATORS.keys():
                this_agg = PNA_AGGREGATORS[agg_name]

            # If the directional, get the right aggregator
            elif "dir" == agg_name[:3]:
                agg_split = agg_name.split("/", maxsplit=1)[0]
                agg_dir, agg_fn_name = agg_split[0], agg_split[1]
                dir_idx = int(agg_dir[3:])

                # Initialize the functions
                if agg_fn_name == "smooth":
                    this_agg = partial(dgn_ops.aggregate_dir_smooth, dir_idx=dir_idx)
                elif agg_fn_name == "softmax":
                    alpha = float(agg_split[2])
                    this_agg = partial(dgn_ops.aggregate_dir_softmax, dir_idx=dir_idx, alpha=alpha)
                elif agg_fn_name == "dx_abs":
                    this_agg = partial(dgn_ops.aggregate_dir_dx_abs, dir_idx=dir_idx)
                elif agg_fn_name == "dx_no_abs":
                    this_agg = partial(dgn_ops.aggregate_dir_dx_no_abs, dir_idx=dir_idx)
                elif agg_fn_name == "dx_abs_balanced":
                    this_agg = partial(dgn_ops.aggregate_dir_dx_abs_balanced, dir_idx=dir_idx)
                elif agg_fn_name == "forward":
                    this_agg = partial(dgn_ops.aggregate_dir_forward, dir_idx=dir_idx)
                elif agg_fn_name == "backward":
                    this_agg = partial(dgn_ops.aggregate_dir_backward, dir_idx=dir_idx)

            if this_agg is None:
                raise ValueError(f"aggregator `{agg_name}` not a valid choice.")

            aggregators.append(this_agg)

        return aggregators

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        The message function to generate messages along the edges.
        """
        return {
            "e": edges.data["e"],
            "source_pos": edges.data["source_pos"],
            "dest_pos": edges.data["dest_pos"],
        }

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        r"""
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data["h"]
        h = nodes.mailbox["e"]
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


class DGNConvolutionalLayer(BaseDGNLayer, PNAConvolutionalLayer):
    r"""
    Implementation of the convolutional architecture of the DGN layer,
    previously known as `DGNSimpleLayer`. This layer aggregates the
    neighbouring messages using multiple aggregators and scalers,
    concatenates their results, then applies an MLP on the concatenated
    features.

    DGN: Directional Graph Networks
    Dominique Beaini, Saro Passaro, Vincent Létourneau, William L. Hamilton, Gabriele Corso, Pietro Liò
    https://arxiv.org/pdf/2010.02863.pdf
    """

    def parse_aggregators(self, aggregators: List[str]) -> List[Callable]:
        return super(BaseDGNLayer, self).parse_aggregators(aggregators)

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        return super(BaseDGNLayer, self).message_func(edges)

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        return super(BaseDGNLayer, self).reduce_func(nodes)

    def pretrans_edges(self, edges):
        return {"e": edges.src["h"], "source_pos": edges.src["pos_dir"], "dest_pos": edges.dst["pos_dir"]}


class DGNMessagePassingLayer(BaseDGNLayer, PNAMessagePassingLayer):
    r"""
    Implementation of the message passing architecture of the DGN message passing layer,
    previously known as `DGNLayerComplex`. This layer applies an MLP as
    pretransformation to the concatenation of $[h_u, h_v, e_{uv}]$ to generate
    the messages, with $h_u$ the node feature, $h_v$ the neighbour node features,
    and $e_{uv}$ the edge feature between the nodes $u$ and $v$.

    After the pre-transformation, it aggregates the messages
    multiple aggregators and scalers,
    concatenates their results, then applies an MLP on the concatenated
    features.

    DGN: Directional Graph Networks
    Dominique Beaini, Saro Passaro, Vincent Létourneau, William L. Hamilton, Gabriele Corso, Pietro Liò
    https://arxiv.org/pdf/2010.02863.pdf
    """

    def parse_aggregators(self, aggregators: List[str]) -> List[Callable]:
        return super(BaseDGNLayer, self).parse_aggregators(aggregators)

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        return super(BaseDGNLayer, self).message_func(edges)

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        return super(BaseDGNLayer, self).reduce_func(nodes)

    def pretrans_edges(self, edges):
        pretrans = super().pretrans_edges(edges)
        pretrans.update({"source_pos": edges.src["pos_dir"], "dest_pos": edges.dst["pos_dir"]})
        return pretrans
