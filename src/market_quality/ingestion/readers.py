"""Read source tables without applying canonical schema or market-data policies."""

from pathlib import Path

import pandas as pd

from market_quality.exceptions import UnsupportedFileTypeError


def read_market_data(path: str | Path) -> pd.DataFrame:
    """Return raw data using Pandas parsing defaults; file/parser errors propagate."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension == ".csv":
        return pd.read_csv(path)
    if extension == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    raise UnsupportedFileTypeError(
        f"Unsupported file type {extension!r}; expected .csv or .parquet"
    )
