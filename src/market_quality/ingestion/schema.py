"""Exact source aliases; resolving a timestamp does not imply UTC semantics."""

import pandas as pd

from market_quality.constants import OHLC_COLUMNS
from market_quality.exceptions import SchemaValidationError

_FIELD_ALIASES = {
    "contract": ("contract_symbol", "contract"),
    "timestamp": ("timestamp_chicago_wall", "timestamp"),
    **{field: (field,) for field in OHLC_COLUMNS},
    "volume": ("volume",),
}


def resolve_market_fields(raw: pd.DataFrame) -> dict[str, str]:
    """Map logical fields to exact source labels without inspecting values.

    Multiple matches, including duplicate column labels, are rejected rather
    than silently choosing a source. Unrelated columns are allowed.
    """
    matches = {
        field: [column for column in raw.columns if column in aliases]
        for field, aliases in _FIELD_ALIASES.items()
    }
    missing = [field for field, columns in matches.items() if not columns]
    if missing:
        raise SchemaValidationError(f"Missing required logical fields: {', '.join(missing)}")

    ambiguous = [field for field, columns in matches.items() if len(columns) > 1]
    if ambiguous:
        details = "; ".join(f"{field}: {', '.join(matches[field])}" for field in ambiguous)
        raise SchemaValidationError(f"Ambiguous source columns for logical fields: {details}")

    return {field: columns[0] for field, columns in matches.items()}
