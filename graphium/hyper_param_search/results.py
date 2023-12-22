"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


_OBJECTIVE_KEY = "objective"


def extract_main_metric_for_hparam_search(results: dict, cfg: dict):
    """Processes the results in the context of a hyper-parameter search."""

    # Extract the objectives
    objectives = cfg[_OBJECTIVE_KEY]
    if isinstance(objectives, str):
        objectives = [objectives]

    # Extract the objective values
    objective_values = [results[k] for k in objectives]
    if len(objective_values) == 1:
        objective_values = objective_values[0]
    return objective_values
