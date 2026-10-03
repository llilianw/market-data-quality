"""Deterministic findings for canonical values, without repairing observations."""

import pandas as pd

from market_quality.constants import OHLC_COLUMNS
from market_quality.exceptions import SchemaValidationError
from market_quality.models import RuleId, Severity
from market_quality.quality.constants import ISSUE_COLUMNS


def validate_row_quality(data: pd.DataFrame) -> pd.DataFrame:
    """Report independent defects in already-parsed canonical market data.

    Missing columns are structural errors; missing cells are blocking findings.
    INVALID_OHLC uses field='ohlc' and an actual_value dictionary of all four
    prices. Negative prices are warnings because futures can trade below zero.
    Optional source lineage is null in findings when unavailable.
    """
    required = ("contract", "timestamp_utc", *OHLC_COLUMNS, "volume")
    missing = [field for field in required if field not in data.columns]
    if missing:
        raise SchemaValidationError(f"Missing required canonical columns: {', '.join(missing)}")

    source = data.reset_index(drop=True)
    context = source.reindex(columns=["source_file", "source_row", "contract", "timestamp_utc"])
    issue_frames = []

    def add_issues(mask: pd.Series, rule_id: RuleId, field: str, message: str | pd.Series) -> None:
        mask = mask.fillna(False)
        if not mask.any():
            return
        frame = context.loc[mask].copy()
        blocking = rule_id != RuleId.NEGATIVE_PRICE
        frame["rule_id"] = rule_id.value
        frame["severity"] = Severity.ERROR.value if blocking else Severity.WARNING.value
        frame["blocking"] = blocking
        frame["field"] = field
        frame["actual_value"] = (
            source.loc[mask, list(OHLC_COLUMNS)].to_dict("records")
            if field == "ohlc"
            else source.loc[mask, field].astype(object)
        )
        frame["message"] = message
        issue_frames.append(frame.loc[:, ISSUE_COLUMNS])

    for field in required:
        add_issues(
            source[field].isna(),
            RuleId.MISSING_REQUIRED_VALUE,
            field,
            f"Required field {field} is missing",
        )

    complete_ohlc = source.loc[:, OHLC_COLUMNS].notna().all(axis=1)
    violations = pd.DataFrame(
        {
            "high < open": source["high"] < source["open"],
            "high < close": source["high"] < source["close"],
            "high < low": source["high"] < source["low"],
            "low > open": source["low"] > source["open"],
            "low > close": source["low"] > source["close"],
        }
    ).fillna(False)
    invalid = complete_ohlc & violations.any(axis=1)
    if invalid.any():
        messages = violations.loc[invalid].apply(
            lambda row: "Invalid OHLC relationships: " + ", ".join(row.index[row]), axis=1
        )
        add_issues(invalid, RuleId.INVALID_OHLC, "ohlc", messages)

    add_issues(source["volume"] < 0, RuleId.NEGATIVE_VOLUME, "volume", "Volume is negative")
    for field in OHLC_COLUMNS:
        add_issues(source[field] < 0, RuleId.NEGATIVE_PRICE, field, f"Price in {field} is negative")

    return (
        pd.concat(issue_frames, ignore_index=True)
        if issue_frames
        else pd.DataFrame(columns=ISSUE_COLUMNS).astype(
            {"blocking": bool, "timestamp_utc": data["timestamp_utc"].dtype}
        )
    )
