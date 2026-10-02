"""Structural normalization preserves raw values pending typed normalization."""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from market_quality.constants import OHLC_COLUMNS
from market_quality.exceptions import SchemaValidationError


def normalize_market_fields(
    raw: pd.DataFrame,
    field_map: Mapping[str, str],
    source_file: str | Path,
) -> pd.DataFrame:
    """Use a resolved mapping without parsing values or rediscovering aliases.

    source_row is a zero-based loaded-row ordinal, not an input index label or
    physical file line number. The source path is stringified without resolution.
    This intermediate frame retains timestamp rather than timestamp_utc.
    Duplicate optional metadata labels are rejected rather than selected implicitly.
    """
    fields = ("contract", "timestamp", *OHLC_COLUMNS, "volume")
    metadata = [column for column in ("exchange", "root") if column in raw.columns]
    duplicates = [column for column in metadata if (raw.columns == column).sum() > 1]
    if duplicates:
        raise SchemaValidationError(
            f"Duplicate optional metadata column labels: {', '.join(duplicates)}"
        )
    source_columns = [field_map[field] for field in fields]
    normalized = raw.loc[:, [*source_columns, *metadata]].copy()
    normalized.columns = [*fields, *metadata]
    normalized = normalized.reset_index(drop=True)
    normalized.insert(0, "source_row", range(len(normalized)))
    normalized.insert(0, "source_file", str(source_file))
    return normalized
