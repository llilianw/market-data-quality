from datetime import date

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.config import AnalyticsConfig, AppConfig, QualityConfig, SessionConfig
from market_quality.models import GapClassification, RuleId
from market_quality.pipeline import run_analysis
from market_quality.quality.constants import ISSUE_COLUMNS


@pytest.fixture
def canonical() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "source_file": ["a.csv", "b.csv", "a.csv", "a.csv", "a.csv", "a.csv", "a.csv"],
            "source_row": [0, 0, 1, 2, 3, 4, 5],
            "contract": ["ESZ6"] * 5 + ["NQZ6"] * 2,
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-01-05T16:31:00Z",
                    "2026-01-05T16:31:00Z",
                    "2026-01-05T16:33:00Z",
                    "2026-01-05T16:34:00Z",
                    "2026-01-05T16:35:00Z",
                    "2026-01-05T16:31:00Z",
                    "2026-01-05T16:34:00Z",
                ],
                utc=True,
            ),
            "open": pd.Series([10, 10, 20, 30, 40, 1000, 1100], dtype="Float64"),
            "high": pd.Series([12, 12, 24, 33, 44, 1002, 1102], dtype="Float64"),
            "low": pd.Series([8, 8, 16, 27, 36, 998, 1098], dtype="Float64"),
            "close": pd.Series([10, 10, 20, 30, 40, 1000, 1100], dtype="Float64"),
            "volume": pd.Series([2, 2, 3, -5, 5, -100, 7], dtype="Int64"),
            "exchange": ["CME"] * 7,
            "root": ["ES"] * 5 + ["NQ"] * 2,
        }
    )
    data.index = [8, 3, 8, 1, 9, 4, 4]
    return data


def test_end_to_end_assessment_drives_both_analytics_without_mutating_input(
    canonical: pd.DataFrame,
) -> None:
    data = canonical.iloc[:5].copy()
    before = data.copy(deep=True)
    result = run_analysis(data)

    assert_frame_equal(result.enriched_data[data.columns], before)
    assert result.enriched_data["is_trading_time"].tolist() == [True] * 5
    assert result.enriched_data["session_date"].tolist() == [pd.Timestamp("2026-01-05")] * 5
    assert result.enriched_data["local_timestamp"].dt.hour.tolist() == [10] * 5
    assert result.quality_issues.columns.tolist() == list(ISSUE_COLUMNS)
    assert result.quality_issues["rule_id"].tolist() == [
        RuleId.EXACT_DUPLICATE.value,
        RuleId.NEGATIVE_VOLUME.value,
        RuleId.EXACT_DUPLICATE.value,
    ]
    assert result.quality_issues["blocking"].tolist() == [False, True, False]
    assert result.eligible_data["source_file"].tolist() == ["a.csv"] * 3
    assert result.eligible_data["source_row"].tolist() == [0, 1, 3]
    assert_frame_equal(result.scoped_data, result.eligible_data)
    assert result.exclusions["source_row"].tolist() == [2, 0]
    assert result.exclusions["exclusion_reason"].tolist() == [
        "blocking_quality_issue",
        "redundant_exact_duplicate",
    ]
    assert len(result.gaps) == 1
    assert result.gaps.loc[0, "classification"] == GapClassification.UNEXPECTED_GAP.value
    assert result.gaps.loc[0, "gap_start"] == pd.Timestamp("2026-01-05T16:32:00Z")
    assert result.gaps.loc[0, "gap_end"] == pd.Timestamp("2026-01-05T16:32:00Z")
    assert result.gaps.loc[0, "missing_count"] == 1
    assert result.insights["category"].tolist() == ["quality", "completeness", "quality"]
    assert result.insights["severity"].tolist() == ["error", "warning", "warning"]
    assert result.insights["contract"].tolist() == ["ESZ6"] * 3
    assert result.insights.loc[0, "rule_id"] == RuleId.NEGATIVE_VOLUME.value
    assert pd.isna(result.insights.loc[1, "rule_id"])
    assert result.insights.loc[2, "rule_id"] == RuleId.EXACT_DUPLICATE.value
    assert result.insights["evidence_count"].tolist() == [1, 1, 2]
    assert result.insights["affected_sessions"].tolist() == [1, 1, 1]

    assert len(result.daily_ohlcv) == 1
    bar = result.daily_ohlcv.iloc[0]
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume, bar.bar_count) == (
        10,
        44,
        8,
        40,
        10,
        3,
    )
    assert_frame_equal(result.rolling_vwap[result.scoped_data.columns], result.scoped_data)
    assert result.rolling_vwap["rolling_volume"].tolist() == [2, 5, 10]
    assert result.rolling_vwap["vwap"].tolist() == pytest.approx([10, 16, 28])
    assert_frame_equal(data, before)


def test_analytics_scope_does_not_erase_full_dataset_evidence(canonical: pd.DataFrame) -> None:
    result = run_analysis(
        canonical, contracts=["ESZ6"], start_date=date(2026, 1, 5), end_date=date(2026, 1, 5)
    )
    assert result.enriched_data["contract"].tolist() == canonical["contract"].tolist()
    assert set(result.eligible_data["contract"]) == {"ESZ6", "NQZ6"}
    assert result.scoped_data["source_row"].tolist() == [0, 1, 3]
    assert result.scoped_data["contract"].tolist() == ["ESZ6"] * 3
    outside = result.quality_issues.loc[result.quality_issues["contract"] == "NQZ6"]
    assert outside["rule_id"].tolist() == [RuleId.NEGATIVE_VOLUME.value]
    assert "NQZ6" in result.exclusions["contract"].tolist()
    assert set(result.gaps["contract"]) == {"ESZ6", "NQZ6"}
    outside_insights = result.insights.loc[
        (result.insights["contract"] == "NQZ6")
        & (result.insights["rule_id"] == RuleId.NEGATIVE_VOLUME.value)
    ]
    assert outside_insights["evidence_count"].tolist() == [1]
    assert result.daily_ohlcv["contract"].tolist() == ["ESZ6"]
    assert result.daily_ohlcv["bar_count"].tolist() == [len(result.scoped_data)]
    assert result.daily_ohlcv["volume"].tolist() == [10]
    assert_frame_equal(result.rolling_vwap[result.scoped_data.columns], result.scoped_data)
    assert result.rolling_vwap["vwap"].tolist() == pytest.approx([10, 16, 28])


def test_empty_contract_selection_preserves_full_dataset_assessment(
    canonical: pd.DataFrame,
) -> None:
    unrestricted = run_analysis(canonical)
    result = run_analysis(canonical, contracts=[])
    assert result.scoped_data.empty
    assert result.daily_ohlcv.empty
    assert result.rolling_vwap.empty
    for field in (
        "enriched_data",
        "quality_issues",
        "gaps",
        "insights",
        "exclusions",
        "eligible_data",
    ):
        expected = getattr(unrestricted, field)
        assert not expected.empty
        assert_frame_equal(getattr(result, field), expected)


def test_subconfigs_reach_session_gap_and_analytics_stages(canonical: pd.DataFrame) -> None:
    data = canonical.iloc[[0, 2]].copy()
    data["timestamp_utc"] = pd.to_datetime(
        ["2026-01-05T19:00:00Z", "2026-01-05T19:12:00Z"], utc=True
    )
    config = AppConfig(
        session=SessionConfig(timezone="Europe/London"),
        quality=QualityConfig(expected_frequency="5min"),
        analytics=AnalyticsConfig(vwap_window="5min", daily_boundary="calendar"),
    )
    result = run_analysis(data, config=config)
    assert str(result.enriched_data["local_timestamp"].dt.tz) == "Europe/London"
    assert result.enriched_data["session_date"].tolist() == [pd.Timestamp("2026-01-06")] * 2
    assert result.gaps["session_date"].tolist() == [pd.Timestamp("2026-01-06")]
    assert result.gaps["classification"].tolist() == [GapClassification.UNEXPECTED_GAP.value]
    assert result.gaps["missing_count"].tolist() == [2]
    assert result.gaps["gap_start"].tolist() == [pd.Timestamp("2026-01-05T19:05:00Z")]
    assert result.gaps["gap_end"].tolist() == [pd.Timestamp("2026-01-05T19:10:00Z")]
    assert result.daily_ohlcv["session_date"].tolist() == [pd.Timestamp("2026-01-05")]
    assert result.daily_ohlcv["volume"].tolist() == [5]
    assert result.rolling_vwap["rolling_volume"].tolist() == [2, 3]
    assert result.rolling_vwap["vwap"].tolist() == [10, 20]


def test_empty_canonical_input_flows_through_normal_stages(canonical: pd.DataFrame) -> None:
    data = canonical.iloc[:0].copy()
    result = run_analysis(data)
    for output in (
        result.enriched_data,
        result.quality_issues,
        result.gaps,
        result.insights,
        result.exclusions,
        result.eligible_data,
        result.scoped_data,
        result.daily_ohlcv,
        result.rolling_vwap,
    ):
        assert output.empty
    assert result.enriched_data.columns.tolist() == [
        *data.columns,
        "local_timestamp",
        "session_date",
        "is_trading_time",
    ]
    assert result.quality_issues.columns.tolist() == list(ISSUE_COLUMNS)
    assert result.quality_issues["blocking"].dtype == bool
    assert result.quality_issues["timestamp_utc"].dtype == data["timestamp_utc"].dtype
    assert result.gaps["gap_start"].dtype == data["timestamp_utc"].dtype
    assert result.insights["evidence_count"].dtype == "int64"
    assert result.insights["affected_sessions"].dtype == "int64"
    assert result.exclusions["timestamp_utc"].dtype == data["timestamp_utc"].dtype
    assert_frame_equal(result.eligible_data, result.enriched_data)
    assert_frame_equal(result.scoped_data, result.eligible_data)
    assert result.daily_ohlcv["bar_count"].dtype == "int64"
    assert result.rolling_vwap.columns.tolist() == [
        *result.scoped_data.columns,
        "representative_price",
        "rolling_volume",
        "vwap",
    ]
