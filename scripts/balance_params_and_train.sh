#!/bin/bash

set -e

old_dim=0
num_params=10000000
rel_error=0.005
rel_step=0.5

out=$(graphium params balance "${@}" "$num_params" "$rel_error" "$rel_step" "$old_dim")
read -r new_dim new_edge_dim rel_step stop <<< "$out"

while true; do
    tmp_dim=$new_dim
    out=$(graphium params balance "${@}" constants.gnn_dim="$new_dim" constants.gnn_edge_dim="$new_edge_dim" "$num_params" "$rel_error" "$rel_step" "$old_dim")
    read -r new_dim new_edge_dim rel_step stop <<< "$out"
    old_dim=$tmp_dim
    [[ $stop == "true" ]] && break
done

graphium-train "${@}" constants.gnn_dim="$new_dim" constants.gnn_edge_dim="$new_edge_dim"