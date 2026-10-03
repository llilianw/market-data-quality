"""Simplified weekly trading sessions without an exchange-holiday calendar."""

import pandas as pd

from market_quality.config import SessionConfig


def assign_sessions(
    data: pd.DataFrame,
    config: SessionConfig | None = None,
) -> pd.DataFrame:
    """Enrich canonical UTC data using local overnight session boundaries.

    session_date is a timezone-naive midnight identifying the closing date.
    Closed timestamps have no session date; missing timestamps have unknown
    trading status. Only overnight schedules (session_end < session_start)
    are supported by this simplified Sunday-to-Friday weekly schedule.
    """
    config = SessionConfig() if config is None else config
    if config.session_end >= config.session_start:
        raise ValueError(
            "Session assignment requires overnight boundaries: session_end < session_start"
        )

    local = data["timestamp_utc"].dt.tz_convert(config.timezone)
    valid = local.notna()
    clock = local.loc[valid].dt.time
    morning = pd.Series(False, index=data.index)
    evening = pd.Series(False, index=data.index)
    morning.loc[valid] = clock < config.session_end
    evening.loc[valid] = clock >= config.session_start
    weekday = local.dt.dayofweek
    trading = (
        ((morning & weekday.between(0, 4)) | (evening & (weekday.between(0, 3) | weekday.eq(6))))
        .astype("boolean")
        .mask(local.isna())
    )

    # Calendar arithmetic after removing the timezone avoids DST-dependent day lengths.
    closing_date = local.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(
        evening.astype(int), unit="D"
    )
    result = data.copy()
    result["local_timestamp"] = local
    result["session_date"] = closing_date.where(trading.fillna(False))
    result["is_trading_time"] = trading
    return result
