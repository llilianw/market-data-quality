"""Daily bars derived from eligible observations and existing session enrichment."""

import pandas as pd

from market_quality.config import AnalyticsConfig
from market_quality.constants import OHLC_COLUMNS
from market_quality.exceptions import SchemaValidationError


def aggregate_daily_ohlcv(
    data: pd.DataFrame,
    config: AnalyticsConfig | None = None,
) -> pd.DataFrame:
    """Aggregate eligible bars without repairing values or filling missing observations.

    Session mode uses assigned closing dates and explicitly active trading rows.
    Calendar mode uses local dates, including eligible scheduled-closure rows;
    its dates also occupy the output session_date column. Results are ordered by
    contract then date. Complete numeric values and unique market keys are upstream
    eligibility preconditions, not validated again here.
    """
    config = AnalyticsConfig() if config is None else config
    base = ["contract", "timestamp_utc", *OHLC_COLUMNS, "volume"]
    if config.daily_boundary == "session":
        required = [*base, "session_date", "is_trading_time"]
    elif config.daily_boundary == "calendar":
        required = [*base, "local_timestamp"]
    else:
        raise ValueError(f"Unsupported daily boundary: {config.daily_boundary!r}")
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SchemaValidationError(f"Missing required analytics columns: {', '.join(missing)}")

    if config.daily_boundary == "session":
        active = data["is_trading_time"].eq(True).fillna(False) & data["session_date"].notna()
        observations = data.loc[active, [*base, "session_date"]].copy()
    else:
        # Remove timezone before midnight normalization: these are local calendar dates.
        local_date = data["local_timestamp"].dt.tz_localize(None).dt.normalize()
        valid = local_date.notna()
        observations = data.loc[valid, base].copy()
        observations["session_date"] = local_date.loc[valid]

    observations = observations.sort_values("timestamp_utc")
    return (
        observations.groupby(["contract", "session_date"], sort=True, observed=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            bar_count=("timestamp_utc", "size"),
            first_timestamp=("timestamp_utc", "first"),
            last_timestamp=("timestamp_utc", "last"),
        )
        .reset_index()
    )
