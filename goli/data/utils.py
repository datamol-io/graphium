import importlib.resources
import pandas as pd


def load_micro_zinc() -> pd.DataFrame:
    """Return a dataframe of micro ZINC."""

    with importlib.resources.open_text("goli.data", "micro_ZINC.csv") as f:
        df = pd.read_csv(importlib.resources.open_text("goli.data", "micro_ZINC.csv"))

    return df
