import importlib.resources
import pandas as pd


def load_micro_zinc() -> pd.DataFrame:
    """Return a dataframe of micro ZINC (1000 data points)."""

    with importlib.resources.open_text("goli.data", "micro_ZINC.csv") as f:
        df = pd.read_csv(f)

    return df


def load_tiny_zinc() -> pd.DataFrame:
    """Return a dataframe of tiny ZINC (100 data points)."""

    with importlib.resources.open_text("goli.data", "micro_ZINC.csv") as f:
        df = pd.read_csv(f, nrows=100)

    return df
