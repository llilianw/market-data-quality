"""Shared column vocabulary; canonical observations need not be clean."""

from typing import Final

OHLC_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")
CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "row_id",
    "source_file",
    "source_row",
    "contract",
    "timestamp_utc",
    *OHLC_COLUMNS,
    "volume",
    "exchange",
    "root",
)
