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
