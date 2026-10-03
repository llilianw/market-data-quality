"""Issue columns shared by deterministic quality detectors."""

from typing import Final

ISSUE_COLUMNS: Final[tuple[str, ...]] = (
    "rule_id",
    "severity",
    "blocking",
    "source_file",
    "source_row",
    "contract",
    "timestamp_utc",
    "field",
    "actual_value",
    "message",
)
