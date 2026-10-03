"""Select analytical observations before downstream OHLCV and VWAP calculations."""

from collections.abc import Sequence
from datetime import date

import pandas as pd

from market_quality.exceptions import SchemaValidationError


def filter_analysis_scope(
    data: pd.DataFrame,
    contracts: Sequence[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Select exact contracts and inclusive trading dates without changing row order.

    Dates use existing closing-date session_date values, preserving overnight
    sessions. None leaves contracts unrestricted; an empty sequence selects none.
    Missing session dates remain only when no date restriction is supplied.
    """
    missing = [column for column in ("contract", "session_date") if column not in data.columns]
    if missing:
        raise SchemaValidationError(f"Missing required filtering columns: {', '.join(missing)}")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    selected = pd.Series(True, index=data.index)
    if contracts is not None:
        selected &= data["contract"].isin(contracts)
    if start_date is not None:
        selected &= data["session_date"].ge(pd.Timestamp(start_date))
    if end_date is not None:
        selected &= data["session_date"].le(pd.Timestamp(end_date))
    return data.loc[selected].copy()
