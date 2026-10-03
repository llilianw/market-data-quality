from collections.abc import Sequence
from datetime import date

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.analytics.filtering import filter_analysis_scope
from market_quality.exceptions import SchemaValidationError
from market_quality.sessions.assignment import assign_sessions


@pytest.fixture
def analytical() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "contract": ["NQZ6", "ESZ6", "ESZ6", "NQZ6", "ESZ6", "ESZ6"],
            "session_date": pd.to_datetime(
                ["2026-01-07", "2026-01-05", "2026-01-06", "2026-01-06", None, "2026-01-07"]
            ),
            "source_file": ["sample.csv"] * 6,
            "source_row": range(6),
            "close": pd.Series([10, 20, 30, 40, 50, 60], dtype="Float64"),
            "volume": pd.Series([0, 2, 3, 4, 5, 6], dtype="Int64"),
            "provider_note": pd.Series(["a", "b", "c", "d", "e", "f"], dtype="string"),
        }
    )
    data.index = [9, 2, 9, 4, 2, 1]
    return data


def test_no_filters_returns_an_independent_equivalent_copy(analytical: pd.DataFrame) -> None:
    before = analytical.copy(deep=True)
    result = filter_analysis_scope(analytical)
    assert result is not analytical
    assert_frame_equal(result, before)
    result.iloc[0, result.columns.get_loc("close")] = 777
    assert_frame_equal(analytical, before)


@pytest.mark.parametrize(
    "contracts,positions",
    [
        (("ESZ6",), [1, 2, 4, 5]),
        (["ESZ6", "NQZ6"], [0, 1, 2, 3, 4, 5]),
        (["UNKNOWN"], []),
        ([], []),
        (["ESZ6", "ESZ6"], [1, 2, 4, 5]),
        (["ES", "esz6", " ESZ6 "], []),
    ],
)
def test_contract_selection_is_exact_and_preserves_matching_rows(
    analytical: pd.DataFrame, contracts: Sequence[str], positions: list[int]
) -> None:
    before = analytical.copy(deep=True)
    result = filter_analysis_scope(analytical, contracts=contracts)
    assert_frame_equal(result, analytical.iloc[positions])
    assert_frame_equal(analytical, before)


def test_bare_string_contract_selection_is_rejected(analytical: pd.DataFrame) -> None:
    with pytest.raises(TypeError):
        filter_analysis_scope(analytical, contracts="ESZ6")


@pytest.mark.parametrize(
    "start_date,end_date,positions",
    [
        (date(2026, 1, 6), None, [0, 2, 3, 5]),
        (None, date(2026, 1, 6), [1, 2, 3]),
        (date(2026, 1, 5), date(2026, 1, 7), [0, 1, 2, 3, 5]),
        (date(2026, 1, 6), date(2026, 1, 6), [2, 3]),
    ],
)
def test_date_bounds_are_inclusive_and_exclude_missing_dates(
    analytical: pd.DataFrame,
    start_date: date | None,
    end_date: date | None,
    positions: list[int],
) -> None:
    result = filter_analysis_scope(analytical, start_date=start_date, end_date=end_date)
    assert_frame_equal(result, analytical.iloc[positions])


def test_contract_and_date_predicates_combine_with_and(analytical: pd.DataFrame) -> None:
    result = filter_analysis_scope(
        analytical, contracts=["ESZ6"], start_date=date(2026, 1, 6), end_date=date(2026, 1, 7)
    )
    assert_frame_equal(result, analytical.iloc[[2, 5]])
    result.iloc[0, result.columns.get_loc("volume")] = 999
    assert analytical.iloc[2]["volume"] == 3


def test_trading_date_selection_preserves_the_whole_overnight_session() -> None:
    timestamps = pd.DatetimeIndex(
        ["2026-01-04 17:00", "2026-01-05 10:00", "2026-01-05 17:00"],
        tz="America/Chicago",
    ).tz_convert("UTC")
    data = assign_sessions(pd.DataFrame({"contract": ["ESZ6"] * 3, "timestamp_utc": timestamps}))
    result = filter_analysis_scope(data, start_date=date(2026, 1, 5), end_date=date(2026, 1, 5))
    assert_frame_equal(result, data.iloc[[0, 1]])
    assert result["local_timestamp"].dt.date.tolist() == [date(2026, 1, 4), date(2026, 1, 5)]
    assert result["timestamp_utc"].dt.date.tolist() == [date(2026, 1, 4), date(2026, 1, 5)]


@pytest.mark.parametrize("column", ["contract", "session_date"])
def test_missing_required_columns_fail_clearly(analytical: pd.DataFrame, column: str) -> None:
    with pytest.raises(
        SchemaValidationError, match=f"Missing required filtering columns: {column}"
    ):
        filter_analysis_scope(analytical.drop(columns=[column]))


def test_reversed_date_range_fails_clearly(analytical: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="start_date must be on or before end_date"):
        filter_analysis_scope(analytical, start_date=date(2026, 1, 7), end_date=date(2026, 1, 5))
