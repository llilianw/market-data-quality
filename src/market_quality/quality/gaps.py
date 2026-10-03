"""Bounded missing-timestamp intervals classified by existing session rules."""

from itertools import pairwise

import pandas as pd

from market_quality.config import QualityConfig, SessionConfig
from market_quality.exceptions import SchemaValidationError
from market_quality.models import GapClassification
from market_quality.sessions.assignment import assign_sessions

_GAP_COLUMNS = (
    "contract",
    "session_date",
    "previous_timestamp",
    "next_timestamp",
    "gap_start",
    "gap_end",
    "missing_count",
    "classification",
)


def detect_gaps(
    data: pd.DataFrame,
    session_config: SessionConfig | None = None,
    quality_config: QualityConfig | None = None,
) -> pd.DataFrame:
    """Report bounded, contract-specific gaps without assuming file coverage.

    Cadence is anchored to each preceding observation. Segment endpoints are
    inclusive missing timestamps; observed bounds are shared by split segments.
    Active segments also split at session-date changes to retain one closing date.
    """
    missing = [field for field in ("contract", "timestamp_utc") if field not in data.columns]
    if missing:
        raise SchemaValidationError(f"Missing required canonical columns: {', '.join(missing)}")

    quality_config = QualityConfig() if quality_config is None else quality_config
    frequency_error = "expected_frequency must represent a positive fixed duration"
    try:
        frequency = pd.Timedelta(
            pd.tseries.frequencies.to_offset(quality_config.expected_frequency).nanos, unit="ns"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(frequency_error) from exc
    if frequency <= pd.Timedelta(0):
        raise ValueError(frequency_error)

    keys = ["contract", "timestamp_utc"]
    observations = data.loc[data.loc[:, keys].notna().all(axis=1), keys].drop_duplicates()
    records = []
    for contract, group in observations.groupby("contract", sort=True, observed=True):
        timestamps = group["timestamp_utc"].sort_values().tolist()
        for previous, following in pairwise(timestamps):
            if following - previous <= frequency:
                continue
            candidates = pd.DataFrame(
                {
                    "timestamp_utc": pd.date_range(
                        previous + frequency, following, freq=frequency, inclusive="left"
                    )
                }
            )
            classified = assign_sessions(candidates, session_config)
            classified["classification"] = GapClassification.EXPECTED_CLOSURE.value
            classified.loc[classified["is_trading_time"], "classification"] = (
                GapClassification.UNEXPECTED_GAP.value
            )
            dates = classified["session_date"]
            prior_dates = dates.shift()
            # Adjacent closure timestamps both have NaT; they remain one segment.
            date_changed = dates.ne(prior_dates) & (dates.notna() | prior_dates.notna())
            changed = classified["classification"].ne(classified["classification"].shift())
            segment_ids = (changed | date_changed).fillna(True).cumsum()
            for _, segment in classified.groupby(segment_ids, sort=False):
                records.append(
                    {
                        "contract": contract,
                        "session_date": segment["session_date"].iloc[0],
                        "previous_timestamp": previous,
                        "next_timestamp": following,
                        "gap_start": segment["timestamp_utc"].iloc[0],
                        "gap_end": segment["timestamp_utc"].iloc[-1],
                        "missing_count": len(segment),
                        "classification": segment["classification"].iloc[0],
                    }
                )

    result = pd.DataFrame(records, columns=_GAP_COLUMNS).astype(
        {"session_date": "datetime64[ns]", "missing_count": "int64"}
    )
    if result.empty:
        for field in ("previous_timestamp", "next_timestamp", "gap_start", "gap_end"):
            result[field] = result[field].astype(data["timestamp_utc"].dtype)
    return result
