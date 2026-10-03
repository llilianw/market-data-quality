from datetime import time

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.config import SessionConfig
from market_quality.sessions.assignment import assign_sessions


def test_daytime_and_evening_sessions_use_the_closing_local_date() -> None:
    data = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-01-05T16:00:00Z",
                    "2026-01-06T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                    "2026-01-09T00:00:00Z",
                ],
                utc=True,
            )
        }
    )
    result = assign_sessions(data)
    assert result["local_timestamp"].tolist() == [
        pd.Timestamp("2026-01-05 10:00", tz="America/Chicago"),
        pd.Timestamp("2026-01-05 18:00", tz="America/Chicago"),
        pd.Timestamp("2026-01-04 18:00", tz="America/Chicago"),
        pd.Timestamp("2026-01-08 18:00", tz="America/Chicago"),
    ]
    assert result["session_date"].tolist() == list(
        pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-05", "2026-01-09"])
    )
    assert result["session_date"].dt.tz is None
    assert pd.api.types.is_datetime64_dtype(result["session_date"].dtype)
    assert result["is_trading_time"].tolist() == [True] * 4


@pytest.mark.parametrize(
    "local_time,session_date",
    [
        ("2026-01-05 15:59", "2026-01-05"),
        ("2026-01-05 16:00", None),
        ("2026-01-05 16:30", None),
        ("2026-01-05 17:00", "2026-01-06"),
        ("2026-01-09 15:59", "2026-01-09"),
        ("2026-01-09 16:00", None),
        ("2026-01-09 18:00", None),
        ("2026-01-10 10:00", None),
        ("2026-01-11 16:59", None),
        ("2026-01-11 17:00", "2026-01-12"),
    ],
)
def test_half_open_boundaries_and_weekly_closures(
    local_time: str, session_date: str | None
) -> None:
    timestamp = pd.Timestamp(local_time, tz="America/Chicago").tz_convert("UTC")
    result = assign_sessions(pd.DataFrame({"timestamp_utc": [timestamp]}))
    assert bool(result.loc[0, "is_trading_time"]) == (session_date is not None)
    if session_date is None:
        assert pd.isna(result.loc[0, "session_date"])
    else:
        assert result.loc[0, "session_date"] == pd.Timestamp(session_date)


def test_missing_timestamps_have_unknown_trading_status() -> None:
    data = pd.DataFrame({"timestamp_utc": pd.to_datetime([None, "2026-01-05T16:00:00Z"], utc=True)})
    result = assign_sessions(data)
    assert pd.isna(result.loc[0, "local_timestamp"])
    assert pd.isna(result.loc[0, "session_date"])
    assert pd.isna(result.loc[0, "is_trading_time"])
    assert result.loc[1, "is_trading_time"]
    assert result["is_trading_time"].dtype == pd.BooleanDtype()


def test_all_missing_timestamps_preserve_rows_and_unknown_session_outputs() -> None:
    data = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([None, None, None], utc=True),
            "source_row": [2, 0, 1],
        },
        index=[9, 2, 9],
    )
    result = assign_sessions(data)
    assert len(result) == len(data)
    assert_frame_equal(result[data.columns], data)
    assert result["local_timestamp"].isna().all()
    assert result["session_date"].isna().all()
    assert result["is_trading_time"].isna().all()
    assert result["is_trading_time"].dtype == pd.BooleanDtype()


def test_empty_input_keeps_the_enriched_schema() -> None:
    data = pd.DataFrame({"timestamp_utc": pd.to_datetime([], utc=True)})
    result = assign_sessions(data)
    assert result.empty
    assert result.columns.tolist() == [
        "timestamp_utc",
        "local_timestamp",
        "session_date",
        "is_trading_time",
    ]
    assert str(result["local_timestamp"].dt.tz) == "America/Chicago"
    assert pd.api.types.is_datetime64_dtype(result["session_date"].dtype)
    assert result["session_date"].dt.tz is None
    assert result["is_trading_time"].dtype == pd.BooleanDtype()


def test_existing_columns_order_and_input_are_preserved() -> None:
    data = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2026-01-06T00:00:00Z", "2026-01-05T16:00:00Z", None], utc=True
            ),
            "source_file": ["sample.parquet"] * 3,
            "source_row": [2, 0, 1],
            "contract": ["ESZ6", "NQZ6", "ESZ6"],
            "open": [100.5, -2.0, None],
            "high": [99.0, 0.0, None],
            "low": [101.0, -3.0, None],
            "close": [100.0, -1.0, None],
            "volume": [0, -500, 10],
            "exchange": ["CME"] * 3,
            "root": ["ES", "NQ", "ES"],
        },
        index=[9, 2, 9],
    )
    before = data.copy(deep=True)
    result = assign_sessions(data)
    assert result.columns.tolist() == [
        *data.columns,
        "local_timestamp",
        "session_date",
        "is_trading_time",
    ]
    assert_frame_equal(result[data.columns], before)
    assert_frame_equal(data, before)
    result.iloc[0, result.columns.get_loc("volume")] = 123
    assert_frame_equal(data, before)


def test_utc_instants_convert_correctly_across_spring_dst() -> None:
    data = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-03-01T23:00:00Z",
                    "2026-03-08T07:59:00Z",
                    "2026-03-08T08:00:00Z",
                    "2026-03-08T22:00:00Z",
                ],
                utc=True,
            )
        }
    )
    result = assign_sessions(data)
    assert result["local_timestamp"].dt.strftime("%Y-%m-%d %H:%M %z").tolist() == [
        "2026-03-01 17:00 -0600",
        "2026-03-08 01:59 -0600",
        "2026-03-08 03:00 -0500",
        "2026-03-08 17:00 -0500",
    ]
    assert result["is_trading_time"].tolist() == [True, False, False, True]
    assert result.loc[0, "session_date"] == pd.Timestamp("2026-03-02")
    assert result.loc[3, "session_date"] == pd.Timestamp("2026-03-09")


def test_configured_timezone_and_overnight_boundaries_are_used() -> None:
    config = SessionConfig(timezone="Europe/London", session_start=time(18), session_end=time(15))
    data = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-01-05T14:59:00Z",
                    "2026-01-05T15:00:00Z",
                    "2026-01-05T17:00:00Z",
                    "2026-01-05T18:00:00Z",
                ],
                utc=True,
            )
        }
    )
    result = assign_sessions(data, config)
    assert str(result["local_timestamp"].dt.tz) == "Europe/London"
    assert result["is_trading_time"].tolist() == [True, False, False, True]
    assert result.loc[0, "session_date"] == pd.Timestamp("2026-01-05")
    assert result.loc[3, "session_date"] == pd.Timestamp("2026-01-06")


def test_non_overnight_boundaries_fail_clearly() -> None:
    data = pd.DataFrame({"timestamp_utc": pd.to_datetime([], utc=True)})
    with pytest.raises(ValueError, match="requires overnight boundaries"):
        assign_sessions(data, SessionConfig(session_start=time(9), session_end=time(16)))
