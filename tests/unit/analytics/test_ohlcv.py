from typing import Literal

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.analytics.ohlcv import aggregate_daily_ohlcv
from market_quality.config import AnalyticsConfig
from market_quality.exceptions import SchemaValidationError
from market_quality.sessions.assignment import assign_sessions


@pytest.fixture
def analytical() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "contract": ["ESZ6"] * 3,
            "timestamp_utc": pd.to_datetime(
                ["2026-01-05T16:33:00Z", "2026-01-05T16:31:00Z", "2026-01-05T16:32:00Z"],
                utc=True,
            ),
            "open": pd.Series([103, 100, 102], dtype="Float64"),
            "high": pd.Series([110, 104, 108], dtype="Float64"),
            "low": pd.Series([99, 98, 101], dtype="Float64"),
            "close": pd.Series([105, 102, 104], dtype="Float64"),
            "volume": pd.Series([30, 10, 20], dtype="Int64"),
            "source_file": ["sample.csv"] * 3,
            "source_row": [0, 1, 2],
            "exchange": ["CME"] * 3,
        },
    )
    data.index = [9, 2, 9]
    return assign_sessions(data)


def test_daily_bar_uses_chronological_prices_and_preserves_input(analytical: pd.DataFrame) -> None:
    before = analytical.copy(deep=True)
    result = aggregate_daily_ohlcv(analytical)
    assert result.columns.tolist() == [
        "contract",
        "session_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bar_count",
        "first_timestamp",
        "last_timestamp",
    ]
    assert len(result) == 1
    bar = result.iloc[0]
    assert bar.contract == "ESZ6"
    assert bar.session_date == pd.Timestamp("2026-01-05")
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (100, 110, 98, 105, 60)
    assert bar.bar_count == 3
    assert bar.first_timestamp == pd.Timestamp("2026-01-05T16:31:00Z")
    assert bar.last_timestamp == pd.Timestamp("2026-01-05T16:33:00Z")
    assert_frame_equal(result, aggregate_daily_ohlcv(analytical.iloc[[2, 0, 1]]))
    assert_frame_equal(analytical, before)


def test_contracts_and_session_dates_remain_isolated(analytical: pd.DataFrame) -> None:
    other_contract = analytical.iloc[[1]].copy()
    other_contract["contract"] = "NQZ6"
    other_contract["volume"] = 200
    next_session = analytical.iloc[[0]].copy()
    next_session["timestamp_utc"] += pd.Timedelta(days=1)
    next_session = assign_sessions(next_session)
    data = pd.concat([next_session, other_contract, analytical], ignore_index=True)
    result = aggregate_daily_ohlcv(data)
    assert result["contract"].tolist() == ["ESZ6", "ESZ6", "NQZ6"]
    assert result["session_date"].tolist() == list(
        pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-05"])
    )
    assert result["volume"].tolist() == [60, 30, 200]
    assert result["bar_count"].tolist() == [3, 1, 1]


def test_evening_and_next_day_observations_share_the_closing_date(analytical: pd.DataFrame) -> None:
    data = analytical.iloc[:2].copy()
    data["timestamp_utc"] = pd.DatetimeIndex(
        ["2026-01-05 18:00", "2026-01-06 10:00"], tz="America/Chicago"
    ).tz_convert("UTC")
    result = aggregate_daily_ohlcv(assign_sessions(data))
    assert len(result) == 1
    assert result.loc[0, "session_date"] == pd.Timestamp("2026-01-06")
    assert result.loc[0, "open"] == 103
    assert result.loc[0, "close"] == 102
    assert result.loc[0, "bar_count"] == 2


def test_zero_volume_no_range_and_negative_prices_remain_usable(analytical: pd.DataFrame) -> None:
    data = analytical.iloc[:2].copy()
    data.loc[:, ["open", "high", "low", "close"]] = -2
    data.loc[:, "volume"] = 0
    result = aggregate_daily_ohlcv(data)
    assert result.loc[0, ["open", "high", "low", "close"]].tolist() == [-2] * 4
    assert result.loc[0, "volume"] == 0
    assert result.loc[0, "bar_count"] == 2


def test_session_mode_includes_only_explicitly_active_assigned_rows(
    analytical: pd.DataFrame,
) -> None:
    data = pd.concat([analytical.iloc[[1]]] * 5, ignore_index=True)
    data["timestamp_utc"] = pd.DatetimeIndex(
        [
            "2026-01-05 10:31",
            "2026-01-05 16:30",
            "2026-01-05 10:32",
            "2026-01-05 10:33",
            "2026-01-05 10:34",
        ],
        tz="America/Chicago",
    ).tz_convert("UTC")
    data = assign_sessions(data)
    data.loc[2:3, "is_trading_time"] = pd.NA
    data.loc[4, "session_date"] = pd.NaT
    result = aggregate_daily_ohlcv(data)
    assert len(result) == 1
    assert result.loc[0, "bar_count"] == 1
    assert result.loc[0, "volume"] == 10
    assert result.loc[0, "first_timestamp"] == data.loc[0, "timestamp_utc"]
    assert result.loc[0, "last_timestamp"] == data.loc[0, "timestamp_utc"]


def test_gap_uses_only_available_observations(analytical: pd.DataFrame) -> None:
    result = aggregate_daily_ohlcv(analytical.iloc[:2])
    assert result.loc[0, "bar_count"] == 2
    assert result.loc[0, "volume"] == 40
    assert result.loc[0, "open"] == 100
    assert result.loc[0, "close"] == 105
    assert result.loc[0, "first_timestamp"] == pd.Timestamp("2026-01-05T16:31:00Z")
    assert result.loc[0, "last_timestamp"] == pd.Timestamp("2026-01-05T16:33:00Z")


@pytest.mark.parametrize(
    "boundary,column",
    [
        ("session", "open"),
        ("session", "session_date"),
        ("session", "is_trading_time"),
        ("calendar", "local_timestamp"),
    ],
)
def test_missing_required_columns_fail_clearly(
    analytical: pd.DataFrame, boundary: Literal["session", "calendar"], column: str
) -> None:
    with pytest.raises(
        SchemaValidationError, match=f"Missing required analytics columns: {column}"
    ):
        aggregate_daily_ohlcv(
            analytical.drop(columns=[column]), AnalyticsConfig(daily_boundary=boundary)
        )


@pytest.mark.parametrize("boundary", ["session", "calendar"])
def test_empty_input_preserves_output_schema_and_dtypes(
    analytical: pd.DataFrame, boundary: Literal["session", "calendar"]
) -> None:
    config = AnalyticsConfig(daily_boundary=boundary)
    result = aggregate_daily_ohlcv(analytical.iloc[:0], config)
    assert result.empty
    assert result.columns.tolist() == aggregate_daily_ohlcv(analytical, config).columns.tolist()
    assert result["contract"].dtype == analytical["contract"].dtype
    assert result["session_date"].dtype == analytical["session_date"].dtype
    for column in ["open", "high", "low", "close", "volume"]:
        assert result[column].dtype == analytical[column].dtype
    assert result["bar_count"].dtype == "int64"
    for column in ["first_timestamp", "last_timestamp"]:
        assert result[column].dtype == analytical["timestamp_utc"].dtype


def test_calendar_mode_uses_local_dates_instead_of_session_dates(analytical: pd.DataFrame) -> None:
    data = analytical.copy()
    data["timestamp_utc"] = pd.DatetimeIndex(
        ["2026-01-05 18:00", "2026-01-06 10:00", "2026-01-05 19:00"],
        tz="America/Chicago",
    ).tz_convert("UTC")
    data = assign_sessions(data).drop(columns=["session_date", "is_trading_time"])
    before = data.copy(deep=True)
    result = aggregate_daily_ohlcv(data, AnalyticsConfig(daily_boundary="calendar"))
    assert result["session_date"].tolist() == list(pd.to_datetime(["2026-01-05", "2026-01-06"]))
    assert result["bar_count"].tolist() == [2, 1]
    assert result["volume"].tolist() == [50, 10]
    session_result = aggregate_daily_ohlcv(assign_sessions(data))
    assert session_result["session_date"].tolist() == [pd.Timestamp("2026-01-06")]
    assert session_result["bar_count"].tolist() == [3]
    assert_frame_equal(data, before)


def test_calendar_mode_includes_eligible_closure_observations(analytical: pd.DataFrame) -> None:
    data = analytical.iloc[[1, 0]].copy()
    data["timestamp_utc"] = pd.DatetimeIndex(
        ["2026-01-05 10:31", "2026-01-05 16:30"], tz="America/Chicago"
    ).tz_convert("UTC")
    data = assign_sessions(data)
    assert data["is_trading_time"].tolist() == [True, False]

    session = aggregate_daily_ohlcv(data)
    calendar = aggregate_daily_ohlcv(data, AnalyticsConfig(daily_boundary="calendar"))
    for result in (session, calendar):
        assert len(result) == 1
        assert result.loc[0, "session_date"] == pd.Timestamp("2026-01-05")

    columns = ["open", "high", "low", "close", "volume", "bar_count"]
    assert session.loc[0, columns].tolist() == [100, 104, 98, 102, 10, 1]
    assert calendar.loc[0, columns].tolist() == [100, 110, 98, 105, 40, 2]
