"""Rolling bar-based VWAP using a fixed Typical Price proxy."""

import pandas as pd

from market_quality.config import AnalyticsConfig
from market_quality.exceptions import SchemaValidationError


def compute_rolling_vwap(
    data: pd.DataFrame,
    config: AnalyticsConfig | None = None,
) -> pd.DataFrame:
    """Enrich active eligible bars with elapsed-time VWAP within each contract/session.

    Windows are (t - window, t]; partial windows use only existing observations.
    Complete canonical values and unique market keys are upstream eligibility
    preconditions. Results retain input columns, ordered by contract, session_date
    and timestamp_utc with a fresh positional index. Daily boundary settings do
    not change VWAP's session isolation.
    """
    required = (
        "contract",
        "timestamp_utc",
        "session_date",
        "is_trading_time",
        "high",
        "low",
        "close",
        "volume",
    )
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SchemaValidationError(f"Missing required VWAP columns: {', '.join(missing)}")

    config = AnalyticsConfig() if config is None else config
    window_error = "vwap_window must represent a positive fixed duration"
    try:
        window = pd.Timedelta(pd.tseries.frequencies.to_offset(config.vwap_window).nanos, unit="ns")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(window_error) from exc
    if window <= pd.Timedelta(0):
        raise ValueError(window_error)

    active = data["is_trading_time"].eq(True).fillna(False) & data["session_date"].notna()
    result = (
        data.loc[active]
        .copy()
        .sort_values(["contract", "session_date", "timestamp_utc"])
        .reset_index(drop=True)
    )
    result["representative_price"] = (result["high"] + result["low"] + result["close"]) / 3
    result["rolling_volume"] = pd.Series(index=result.index, dtype="float64")
    result["vwap"] = pd.Series(index=result.index, dtype="float64")

    for _, group in result.groupby(["contract", "session_date"], sort=False, observed=True):
        values = group.loc[:, ["timestamp_utc", "volume"]].copy()
        values["weighted_value"] = group["representative_price"] * group["volume"]
        totals = values.set_index("timestamp_utc").rolling(window, closed="right").sum()
        volume = totals["volume"]
        vwap = totals["weighted_value"] / volume.where(volume.ne(0))
        # Timestamp-indexed rolling outputs map back to this group's positional rows.
        result.loc[group.index, "rolling_volume"] = volume.to_numpy()
        result.loc[group.index, "vwap"] = vwap.to_numpy()

    return result
