"""Separate structural field normalization from typed canonicalization."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from market_quality.config import SessionConfig
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


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    """Stage outputs; rejected rows retain their original structural values."""

    data: pd.DataFrame
    issues: pd.DataFrame
    rejected_rows: pd.DataFrame


def _timestamps_to_utc(timestamps: pd.Series, timezone: str) -> pd.Series:
    # Text parsing avoids silently assigning nanosecond epoch units to numbers.
    values = (
        timestamps
        if pd.api.types.is_datetime64_any_dtype(timestamps.dtype)
        else timestamps.astype("string")
    )
    if not pd.api.types.is_datetime64_any_dtype(values.dtype):
        # Relative tokens would make historical ingestion depend on the machine clock.
        relative = values.str.strip().str.lower().isin(("now", "today"))
        values = values.mask(relative)
    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    except ValueError as exc:
        raise SchemaValidationError(
            "Timestamp column must use one homogeneous timezone representation; "
            "mixed naive/aware timestamps or mixed UTC offsets are not supported"
        ) from exc
    if not pd.api.types.is_datetime64_any_dtype(parsed.dtype):
        raise SchemaValidationError(
            "Timestamp column must use one homogeneous timezone representation"
        )
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    return parsed.dt.tz_convert("UTC")


def canonicalize_market_data(
    normalized: pd.DataFrame,
    session_config: SessionConfig | None = None,
) -> CanonicalizationResult:
    """Canonicalize a structural frame without applying market-value DQ rules.

    Existing nulls remain canonical. Non-null parsing/localization failures yield
    field-level issues and reject their row once, retaining the original evidence.
    Timestamp representations must be homogeneous within the source column.
    """
    session_config = SessionConfig() if session_config is None else session_config
    source = normalized.reset_index(drop=True)
    data = source.rename(columns={"timestamp": "timestamp_utc"}).copy()
    converted = {"timestamp": _timestamps_to_utc(source["timestamp"], session_config.timezone)}
    for field in (*OHLC_COLUMNS, "volume"):
        values = source[field]
        if pd.api.types.is_datetime64_any_dtype(values.dtype) or pd.api.types.is_timedelta64_dtype(
            values.dtype
        ):
            # Temporal storage integers, including the NaT sentinel, are not market numbers.
            converted[field] = pd.Series(float("nan"), index=source.index)
        else:
            converted[field] = pd.to_numeric(values, errors="coerce")
    rejected = pd.Series(False, index=source.index)
    issue_frames = []
    issue_columns = ("source_file", "source_row", "field", "issue_type", "raw_value", "message")
    for field, values in converted.items():
        data["timestamp_utc" if field == "timestamp" else field] = values
        malformed = source[field].notna() & values.isna()
        rejected |= malformed
        if not malformed.any():
            continue
        issues = source.loc[malformed, ["source_file", "source_row"]].copy()
        issues["field"] = field
        issues["issue_type"] = (
            "TIMESTAMP_NORMALIZATION_FAILED" if field == "timestamp" else "MALFORMED_NUMERIC"
        )
        issues["raw_value"] = source.loc[malformed, field].astype(object)
        issues["message"] = (
            "Timestamp cannot be parsed or normalized to UTC without guessing DST"
            if field == "timestamp"
            else f"Non-null {field} value cannot be parsed as numeric"
        )
        issue_frames.append(issues)
    return CanonicalizationResult(
        data=data.loc[~rejected].reset_index(drop=True),
        issues=(
            pd.concat(issue_frames, ignore_index=True)
            if issue_frames
            else pd.DataFrame(columns=issue_columns)
        ),
        rejected_rows=source.loc[rejected].copy().reset_index(drop=True),
    )
