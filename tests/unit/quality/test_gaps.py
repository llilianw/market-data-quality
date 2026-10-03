import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.config import QualityConfig, SessionConfig
from market_quality.exceptions import SchemaValidationError
from market_quality.models import GapClassification
from market_quality.quality.gaps import detect_gaps


def _chicago_frame(*local_times: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract": ["ESZ6"] * len(local_times),
            "timestamp_utc": pd.DatetimeIndex(
                [pd.Timestamp(value, tz="America/Chicago") for value in local_times]
            ).tz_convert("UTC"),
        }
    )


def test_consecutive_observations_are_isolated_by_contract_and_duplicates_are_presence() -> None:
    es = _chicago_frame("2026-01-05 10:31", "2026-01-05 10:31", "2026-01-05 10:32")
    nq = _chicago_frame("2026-01-05 10:32", "2026-01-05 10:33")
    nq["contract"] = "NQZ6"
    assert detect_gaps(pd.concat([es, nq], ignore_index=True)).empty


def test_another_contracts_observations_do_not_fill_a_gap() -> None:
    es = _chicago_frame("2026-01-05 10:31", "2026-01-05 10:34")
    nq = _chicago_frame("2026-01-05 10:32", "2026-01-05 10:33")
    nq["contract"] = "NQZ6"
    gaps = detect_gaps(pd.concat([es, nq], ignore_index=True))
    assert len(gaps) == 1
    gap = gaps.iloc[0]
    assert gap.contract == "ESZ6"
    assert gap.classification == GapClassification.UNEXPECTED_GAP.value
    assert gap.gap_start == pd.Timestamp("2026-01-05T16:32:00Z")
    assert gap.gap_end == pd.Timestamp("2026-01-05T16:33:00Z")
    assert gap.missing_count == 2
    assert gaps.loc[gaps["contract"] == "NQZ6"].empty


@pytest.mark.parametrize("following,count", [("10:33", 1), ("10:34", 2)])
def test_missing_active_minutes_form_one_segment(following: str, count: int) -> None:
    data = _chicago_frame("2026-01-05 10:31", f"2026-01-05 {following}")
    gaps = detect_gaps(data)
    assert len(gaps) == 1
    gap = gaps.iloc[0]
    assert gap.contract == "ESZ6"
    assert gap.classification == GapClassification.UNEXPECTED_GAP.value
    assert gap.session_date == pd.Timestamp("2026-01-05")
    assert gap.previous_timestamp == data.loc[0, "timestamp_utc"]
    assert gap.next_timestamp == data.loc[1, "timestamp_utc"]
    assert gap.gap_start == pd.Timestamp("2026-01-05T16:32:00Z")
    assert gap.gap_end == gap.next_timestamp - pd.Timedelta(minutes=1)
    assert gap.missing_count == count


def test_daily_break_is_one_expected_closure_without_a_session_date() -> None:
    data = _chicago_frame("2026-01-05 15:59", "2026-01-05 17:00")
    gaps = detect_gaps(data)
    assert len(gaps) == 1
    gap = gaps.iloc[0]
    assert gap.classification == GapClassification.EXPECTED_CLOSURE.value
    assert pd.isna(gap.session_date)
    assert gap.gap_start == pd.Timestamp("2026-01-05T22:00:00Z")
    assert gap.gap_end == pd.Timestamp("2026-01-05T22:59:00Z")
    assert gap.missing_count == 60


@pytest.mark.parametrize(
    "friday,sunday,count",
    [("2026-01-09", "2026-01-11", 2940), ("2026-03-06", "2026-03-08", 2880)],
)
def test_weekend_is_expected_closure_including_a_dst_weekend(
    friday: str, sunday: str, count: int
) -> None:
    data = _chicago_frame(f"{friday} 15:59", f"{sunday} 17:00")
    gaps = detect_gaps(data)
    assert len(gaps) == 1
    gap = gaps.iloc[0]
    assert gap.classification == GapClassification.EXPECTED_CLOSURE.value
    assert pd.isna(gap.session_date)
    assert gap.missing_count == count
    assert gap.gap_start == data.loc[0, "timestamp_utc"] + pd.Timedelta(minutes=1)
    assert gap.gap_end == data.loc[1, "timestamp_utc"] - pd.Timedelta(minutes=1)


def test_break_and_missing_reopening_minute_are_separate_segments() -> None:
    data = _chicago_frame("2026-01-05 15:59", "2026-01-05 17:01")
    gaps = detect_gaps(data)
    assert gaps["classification"].tolist() == [
        GapClassification.EXPECTED_CLOSURE.value,
        GapClassification.UNEXPECTED_GAP.value,
    ]
    assert gaps["missing_count"].tolist() == [60, 1]
    assert pd.isna(gaps.loc[0, "session_date"])
    assert gaps.loc[1, "session_date"] == pd.Timestamp("2026-01-06")
    assert gaps.loc[0, "gap_end"] == pd.Timestamp("2026-01-05T22:59:00Z")
    assert gaps.loc[1, "gap_start"] == pd.Timestamp("2026-01-05T23:00:00Z")
    assert gaps.loc[1, "gap_start"] == gaps.loc[1, "gap_end"]
    assert gaps["previous_timestamp"].tolist() == [data.loc[0, "timestamp_utc"]] * 2
    assert gaps["next_timestamp"].tolist() == [data.loc[1, "timestamp_utc"]] * 2


def test_no_leading_or_trailing_coverage_is_inferred() -> None:
    data = _chicago_frame("2026-01-05 18:25", "2026-01-05 18:26")
    assert detect_gaps(data).empty


def test_ordering_and_duplicate_indexes_do_not_change_results_or_mutate_input() -> None:
    data = _chicago_frame("2026-01-05 10:34", "2026-01-05 10:31", "2026-01-05 10:31")
    data["source_row"] = [2, 0, 1]
    data["volume"] = [-500, 0, 10]
    data.index = [9, 2, 9]
    before = data.copy(deep=True)
    gaps = detect_gaps(data)
    assert gaps["missing_count"].tolist() == [2]
    assert_frame_equal(gaps, detect_gaps(data.iloc[::-1]))
    assert_frame_equal(data, before)


def test_missing_keys_are_ignored() -> None:
    data = _chicago_frame(
        "2026-01-05 10:31",
        "2026-01-05 10:33",
        "2026-01-05 10:31",
        "2026-01-05 15:00",
        "2026-01-05 20:00",
    )
    data.loc[[2, 3], "contract"] = None
    data.loc[4, "timestamp_utc"] = pd.NaT
    gaps = detect_gaps(data)
    assert len(gaps) == 1
    assert gaps.loc[0, "contract"] == "ESZ6"
    assert gaps.loc[0, "missing_count"] == 1
    assert detect_gaps(data.iloc[2:]).empty


def test_five_minute_frequency_uses_configured_cadence() -> None:
    data = _chicago_frame("2026-01-05 10:30", "2026-01-05 10:45")
    gaps = detect_gaps(data, quality_config=QualityConfig(expected_frequency="5min"))
    assert len(gaps) == 1
    assert gaps.loc[0, "missing_count"] == 2
    assert gaps.loc[0, "gap_start"] == pd.Timestamp("2026-01-05T16:35:00Z")
    assert gaps.loc[0, "gap_end"] == pd.Timestamp("2026-01-05T16:40:00Z")


def test_coarse_cadence_keeps_unexpected_segments_within_one_session_date() -> None:
    data = _chicago_frame("2026-01-05 06:00", "2026-01-06 06:00")
    gaps = detect_gaps(data, quality_config=QualityConfig(expected_frequency="6h"))
    assert gaps["missing_count"].tolist() == [1, 2]
    assert gaps["session_date"].tolist() == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    assert set(gaps["classification"]) == {GapClassification.UNEXPECTED_GAP.value}


def test_configured_session_timezone_is_used_for_classification() -> None:
    data = pd.DataFrame(
        {
            "contract": ["ESZ6", "ESZ6"],
            "timestamp_utc": pd.to_datetime(
                ["2026-01-05T15:59:00Z", "2026-01-05T17:01:00Z"], utc=True
            ),
        }
    )
    gaps = detect_gaps(data, session_config=SessionConfig(timezone="Europe/London"))
    assert gaps["classification"].tolist() == [
        GapClassification.EXPECTED_CLOSURE.value,
        GapClassification.UNEXPECTED_GAP.value,
    ]
    assert gaps["missing_count"].tolist() == [60, 1]
    assert gaps.loc[1, "session_date"] == pd.Timestamp("2026-01-06")


@pytest.mark.parametrize("frequency", ["banana", "0min", "-1min", "ME"])
def test_invalid_or_nonpositive_or_nonfixed_frequency_fails_clearly(frequency: str) -> None:
    data = _chicago_frame("2026-01-05 10:31")
    with pytest.raises(ValueError, match="positive fixed duration"):
        detect_gaps(data, quality_config=QualityConfig(expected_frequency=frequency))


def test_empty_input_returns_typed_gap_columns() -> None:
    data = pd.DataFrame(
        {"contract": pd.Series(dtype="string"), "timestamp_utc": pd.to_datetime([], utc=True)}
    )
    gaps = detect_gaps(data)
    assert gaps.empty
    assert gaps.columns.tolist() == [
        "contract",
        "session_date",
        "previous_timestamp",
        "next_timestamp",
        "gap_start",
        "gap_end",
        "missing_count",
        "classification",
    ]
    for field in ("previous_timestamp", "next_timestamp", "gap_start", "gap_end"):
        assert gaps[field].dtype == data["timestamp_utc"].dtype
        assert str(gaps[field].dt.tz) == "UTC"
    assert gaps["session_date"].dtype == "datetime64[ns]"
    assert gaps["session_date"].dt.tz is None
    assert pd.api.types.is_integer_dtype(gaps["missing_count"].dtype)


def test_missing_required_columns_raise_schema_validation_error() -> None:
    with pytest.raises(
        SchemaValidationError, match="Missing required canonical columns: timestamp_utc"
    ):
        detect_gaps(pd.DataFrame({"contract": ["ESZ6"]}))
