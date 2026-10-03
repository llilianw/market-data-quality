import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.exceptions import SchemaValidationError
from market_quality.models import RuleId, Severity
from market_quality.quality.duplicates import detect_duplicates
from market_quality.quality.rules import validate_row_quality


@pytest.fixture
def copies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": ["first.csv", "second.parquet"],
            "source_row": [7, 2],
            "contract": ["ESZ6"] * 2,
            "timestamp_utc": pd.to_datetime(["2026-01-05T16:00:00Z"] * 2, utc=True),
            "open": pd.Series([100, 100], dtype="Float64"),
            "high": pd.Series([101, 101], dtype="Float64"),
            "low": pd.Series([99, 99], dtype="Float64"),
            "close": pd.Series([100, 100], dtype="Float64"),
            "volume": pd.Series([10, 10], dtype="Int64"),
            "exchange": ["CME", "OTHER"],
            "root": ["ES", "OTHER"],
        },
    )


def test_unique_keys_do_not_generate_findings(copies: pd.DataFrame) -> None:
    third = copies.iloc[[0]].copy()
    third["timestamp_utc"] += pd.Timedelta(minutes=1)
    data = pd.concat([copies, third], ignore_index=True)
    data.loc[1, "contract"] = "NQZ6"
    assert detect_duplicates(data).empty


def test_exact_duplicates_report_every_participant_ignoring_metadata(copies: pd.DataFrame) -> None:
    copies.index = [9, 9]
    before = copies.copy(deep=True)
    issues = detect_duplicates(copies)
    assert len(issues) == 2
    assert set(issues["rule_id"]) == {RuleId.EXACT_DUPLICATE.value}
    assert set(issues["severity"]) == {Severity.WARNING.value}
    assert not issues["blocking"].any()
    context = ["source_file", "source_row", "contract", "timestamp_utc"]
    assert_frame_equal(issues[context], copies[context].reset_index(drop=True))
    assert issues["field"].tolist() == ["ohlcv"] * 2
    assert (
        issues["actual_value"].tolist()
        == [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}] * 2
    )
    for message in issues["message"]:
        assert "Exact duplicate" in message
        assert "2 observations" in message
        assert "ESZ6" in message
        assert "2026-01-05 16:00:00+00:00" in message
    assert_frame_equal(copies, before)


def test_matching_missing_values_are_exact_duplicates(copies: pd.DataFrame) -> None:
    copies["open"] = pd.Series([pd.NA, pd.NA], dtype="Float64")
    copies["volume"] = pd.Series([pd.NA, pd.NA], dtype="Int64")
    issues = detect_duplicates(copies)
    assert len(issues) == 2
    assert set(issues["rule_id"]) == {RuleId.EXACT_DUPLICATE.value}
    assert not issues["blocking"].any()


@pytest.mark.parametrize("value", [102, pd.NA])
def test_market_disagreement_makes_both_copies_conflicting(
    copies: pd.DataFrame, value: object
) -> None:
    copies.loc[1, "high"] = value
    issues = detect_duplicates(copies)
    assert len(issues) == 2
    assert set(issues["rule_id"]) == {RuleId.CONFLICTING_DUPLICATE.value}
    assert set(issues["severity"]) == {Severity.ERROR.value}
    assert issues["blocking"].all()
    assert issues["source_row"].tolist() == [7, 2]
    assert (
        issues["message"].str.contains("Conflicting duplicate: 2 observations", regex=False).all()
    )
    if pd.isna(value):
        assert pd.isna(issues.loc[1, "actual_value"]["high"])
    else:
        assert issues.loc[1, "actual_value"]["high"] == value


def test_one_disagreement_classifies_the_entire_three_row_group(copies: pd.DataFrame) -> None:
    third = copies.iloc[[0]].copy()
    third["source_file"] = "third.csv"
    third["source_row"] = 1
    third["volume"] = 12
    data = pd.concat([copies, third], ignore_index=True)
    issues = detect_duplicates(data)
    assert len(issues) == 3
    assert set(issues["rule_id"]) == {RuleId.CONFLICTING_DUPLICATE.value}
    assert issues["blocking"].all()
    assert issues["message"].str.contains("3 observations", regex=False).all()
    reordered = detect_duplicates(data.iloc[[2, 0, 1]])
    assert_frame_equal(
        issues.sort_values("source_file").reset_index(drop=True),
        reordered.sort_values("source_file").reset_index(drop=True),
    )


@pytest.mark.parametrize("field", ["contract", "timestamp_utc"])
def test_missing_keys_do_not_participate_in_grouping(copies: pd.DataFrame, field: str) -> None:
    missing = copies.copy()
    missing[field] = pd.NaT if field == "timestamp_utc" else None
    data = pd.concat([copies, missing], ignore_index=True)
    issues = detect_duplicates(data)
    assert len(issues) == 2
    assert issues["source_file"].tolist() == copies["source_file"].tolist()
    assert issues["contract"].notna().all()
    assert issues["timestamp_utc"].notna().all()
    assert detect_duplicates(missing).empty


def test_multiple_groups_are_classified_independently(copies: pd.DataFrame) -> None:
    other = copies.copy()
    other["contract"] = "NQZ6"
    other.loc[1, "close"] = 101
    data = pd.concat([copies, other], ignore_index=True)
    issues = detect_duplicates(data)
    assert len(issues) == 4
    assert set(issues.loc[issues["contract"] == "ESZ6", "rule_id"]) == {
        RuleId.EXACT_DUPLICATE.value
    }
    assert set(issues.loc[issues["contract"] == "NQZ6", "rule_id"]) == {
        RuleId.CONFLICTING_DUPLICATE.value
    }


def test_missing_optional_lineage_does_not_prevent_findings(copies: pd.DataFrame) -> None:
    issues = detect_duplicates(copies.drop(columns=["source_file", "source_row"]))
    assert len(issues) == 2
    assert issues[["source_file", "source_row"]].isna().all().all()


def test_empty_input_returns_the_shared_issue_schema_and_dtypes(copies: pd.DataFrame) -> None:
    empty = copies.iloc[:0]
    issues = detect_duplicates(empty)
    assert issues.empty
    assert issues.columns.tolist() == [
        "rule_id",
        "severity",
        "blocking",
        "source_file",
        "source_row",
        "contract",
        "timestamp_utc",
        "field",
        "actual_value",
        "message",
    ]
    assert_frame_equal(issues, validate_row_quality(empty))
    assert issues["timestamp_utc"].dtype == copies["timestamp_utc"].dtype
    assert pd.api.types.is_bool_dtype(issues["blocking"].dtype)


def test_missing_required_columns_raise_a_structural_error(copies: pd.DataFrame) -> None:
    with pytest.raises(
        SchemaValidationError, match="Missing required canonical columns: timestamp_utc, volume"
    ):
        detect_duplicates(copies.drop(columns=["timestamp_utc", "volume"]))
