import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.analytics.vwap import compute_rolling_vwap
from market_quality.config import AnalyticsConfig
from market_quality.exceptions import SchemaValidationError
from market_quality.sessions.assignment import assign_sessions


def _analytical_bars(
    timestamps: list[str], prices: list[float], volumes: list[int]
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "source_file": ["sample.csv"] * len(timestamps),
            "source_row": range(len(timestamps)),
            "contract": ["ESZ6"] * len(timestamps),
            "timestamp_utc": pd.to_datetime(timestamps, utc=True),
            "high": pd.Series(prices, dtype="Float64"),
            "low": pd.Series(prices, dtype="Float64"),
            "close": pd.Series(prices, dtype="Float64"),
            "volume": pd.Series(volumes, dtype="Int64"),
            "exchange": ["CME"] * len(timestamps),
        }
    )
    return assign_sessions(data)


def test_typical_price_and_vwap_match_hand_calculation_and_preserve_inputs() -> None:
    data = _analytical_bars(
        ["2026-01-05T16:00:00Z", "2026-01-05T16:01:00Z", "2026-01-05T16:02:00Z"],
        [10, 18, 28],
        [2, 3, 5],
    )
    data["high"] = pd.Series([12, 24, 33], dtype="Float64")
    data["low"] = pd.Series([6, 12, 21], dtype="Float64")
    data["close"] = pd.Series([12, 18, 30], dtype="Float64")
    data.index = [9, 2, 9]
    before = data.copy(deep=True)
    result = compute_rolling_vwap(data)
    assert result["representative_price"].tolist() == pytest.approx([10, 18, 28])
    assert result["rolling_volume"].tolist() == pytest.approx([2, 5, 10])
    assert result["vwap"].tolist() == pytest.approx([10, 74 / 5, 214 / 10])
    assert_frame_equal(result, compute_rolling_vwap(data.iloc[[2, 0, 1]]))
    assert_frame_equal(result[data.columns], before.reset_index(drop=True))
    assert_frame_equal(data, before)
    result.loc[0, "close"] = 500
    assert_frame_equal(data, before)


def test_irregular_timestamps_use_elapsed_time_without_interpolation() -> None:
    data = _analytical_bars(
        [
            "2026-01-05T16:00:00Z",
            "2026-01-05T16:01:00Z",
            "2026-01-05T16:14:00Z",
            "2026-01-05T16:30:00Z",
        ],
        [10, 20, 30, 40],
        [1, 1, 2, 4],
    )
    result = compute_rolling_vwap(data)
    assert result["timestamp_utc"].tolist() == data["timestamp_utc"].tolist()
    assert result["rolling_volume"].tolist() == pytest.approx([1, 2, 4, 4])
    assert result["vwap"].tolist() == pytest.approx([10, 15, 22.5, 40])


def test_window_excludes_left_endpoint_and_includes_just_inside_it() -> None:
    data = _analytical_bars(
        ["2026-01-05T16:00:00Z", "2026-01-05T16:00:01Z", "2026-01-05T16:15:00Z"],
        [100, 10, 20],
        [10, 2, 3],
    )
    result = compute_rolling_vwap(data)
    assert result.loc[2, "rolling_volume"] == 5
    assert result.loc[2, "vwap"] == pytest.approx(16)


def test_other_contracts_do_not_enter_a_rolling_window() -> None:
    data = _analytical_bars(
        ["2026-01-05T16:00:00Z", "2026-01-05T16:00:30Z", "2026-01-05T16:01:00Z"],
        [10, 1000, 20],
        [2, 100, 3],
    )
    data.loc[1, "contract"] = "NQZ6"
    result = compute_rolling_vwap(data)
    assert result["contract"].tolist() == ["ESZ6", "ESZ6", "NQZ6"]
    assert result["rolling_volume"].tolist() == pytest.approx([2, 5, 100])
    assert result["vwap"].tolist() == pytest.approx([10, 16, 1000])
    assert_frame_equal(result, compute_rolling_vwap(data.iloc[[1, 2, 0]]))


def test_contracts_at_identical_timestamps_have_independent_vwap() -> None:
    data = _analytical_bars(
        [
            "2026-01-05T16:00:00Z",
            "2026-01-05T16:00:00Z",
            "2026-01-05T16:01:00Z",
            "2026-01-05T16:01:00Z",
        ],
        [10, 1000, 20, 2000],
        [2, 100, 3, 300],
    )
    data["contract"] = ["ESZ6", "NQZ6", "ESZ6", "NQZ6"]
    result = compute_rolling_vwap(data)
    assert result["contract"].tolist() == ["ESZ6", "ESZ6", "NQZ6", "NQZ6"]
    assert result["rolling_volume"].tolist() == pytest.approx([2, 5, 100, 400])
    assert result["vwap"].tolist() == pytest.approx([10, 16, 1000, 1750])


def test_session_boundaries_reset_even_within_the_window() -> None:
    data = _analytical_bars(
        ["2026-01-05T16:00:00Z", "2026-01-05T16:01:00Z", "2026-01-05T16:02:00Z"],
        [100, 10, 20],
        [100, 2, 3],
    )
    # Artificial nearby dates protect isolation independently of the overnight break length.
    data["session_date"] = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-06"])
    result = compute_rolling_vwap(data, AnalyticsConfig(daily_boundary="calendar"))
    assert result["rolling_volume"].tolist() == pytest.approx([100, 2, 5])
    assert result["vwap"].tolist() == pytest.approx([100, 10, 16])


def test_zero_volume_contributes_no_weight_and_zero_windows_are_unavailable() -> None:
    data = _analytical_bars(
        [
            "2026-01-05T16:00:00Z",
            "2026-01-05T16:01:00Z",
            "2026-01-05T16:20:00Z",
            "2026-01-05T16:21:00Z",
        ],
        [10, 1000, 20, 30],
        [2, 0, 0, 0],
    )
    result = compute_rolling_vwap(data)
    assert result["rolling_volume"].tolist() == pytest.approx([2, 2, 0, 0])
    assert result["vwap"].iloc[:2].tolist() == pytest.approx([10, 10])
    assert result["vwap"].iloc[2:].isna().all()


def test_negative_representative_prices_remain_usable() -> None:
    data = _analytical_bars(["2026-01-05T16:00:00Z", "2026-01-05T16:01:00Z"], [-10, -20], [2, 3])
    result = compute_rolling_vwap(data)
    assert result["representative_price"].tolist() == [-10, -20]
    assert result["vwap"].tolist() == pytest.approx([-10, -16])


def test_only_explicitly_active_assigned_rows_participate() -> None:
    data = _analytical_bars(
        [
            "2026-01-05T21:50:00Z",
            "2026-01-05T22:00:00Z",
            "2026-01-05T21:51:00Z",
            "2026-01-05T21:52:00Z",
        ],
        [10, 1000, 2000, 3000],
        [2, 100, 100, 100],
    )
    data.loc[2, "is_trading_time"] = pd.NA
    data.loc[3, "session_date"] = pd.NaT
    result = compute_rolling_vwap(data)
    assert result["source_row"].tolist() == [0]
    assert result["rolling_volume"].tolist() == [2]
    assert result["vwap"].tolist() == [10]


@pytest.mark.parametrize("column", ["high", "session_date", "is_trading_time"])
def test_missing_required_columns_fail_clearly(column: str) -> None:
    data = _analytical_bars(["2026-01-05T16:00:00Z"], [10], [2])
    with pytest.raises(SchemaValidationError, match=f"Missing required VWAP columns: {column}"):
        compute_rolling_vwap(data.drop(columns=[column]))


@pytest.mark.parametrize("window", ["invalid", "0min", "-1min", "1ME"])
def test_invalid_nonpositive_or_nonfixed_windows_fail_clearly(window: str) -> None:
    data = _analytical_bars([], [], [])
    with pytest.raises(ValueError, match="vwap_window must represent a positive fixed duration"):
        compute_rolling_vwap(data, AnalyticsConfig(vwap_window=window))


def test_configured_window_is_used() -> None:
    data = _analytical_bars(["2026-01-05T16:00:00Z", "2026-01-05T16:06:00Z"], [10, 20], [2, 3])
    result = compute_rolling_vwap(data, AnalyticsConfig(vwap_window="5min"))
    assert result["rolling_volume"].tolist() == [2, 3]
    assert result["vwap"].tolist() == [10, 20]


@pytest.mark.parametrize("empty_input", [True, False])
def test_empty_or_fully_filtered_input_keeps_typed_output(empty_input: bool) -> None:
    data = _analytical_bars(["2026-01-05T22:30:00Z"], [10], [2])
    if empty_input:
        data = data.iloc[:0]
    result = compute_rolling_vwap(data)
    assert result.empty
    assert result.columns.tolist() == [
        *data.columns,
        "representative_price",
        "rolling_volume",
        "vwap",
    ]
    assert_frame_equal(result[data.columns], data.iloc[:0].reset_index(drop=True))
    for column in ("representative_price", "rolling_volume", "vwap"):
        assert pd.api.types.is_numeric_dtype(result[column].dtype)
