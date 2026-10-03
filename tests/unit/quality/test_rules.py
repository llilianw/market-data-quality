import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.exceptions import SchemaValidationError
from market_quality.models import RuleId, Severity
from market_quality.quality.rules import validate_row_quality


@pytest.fixture
def canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": ["sample.parquet"],
            "source_row": [7],
            "contract": ["ESZ6"],
            "timestamp_utc": pd.to_datetime(["2026-01-05T16:00:00Z"], utc=True),
            "open": pd.Series([100], dtype="Float64"),
            "high": pd.Series([101], dtype="Float64"),
            "low": pd.Series([99], dtype="Float64"),
            "close": pd.Series([100.5], dtype="Float64"),
            "volume": pd.Series([10], dtype="Int64"),
        }
    )


@pytest.mark.parametrize(
    "overrides",
    [{}, {"volume": 0}, {"open": 0, "high": 0, "low": 0, "close": 0}],
)
def test_valid_bars_zero_volume_and_no_range_are_allowed(
    canonical: pd.DataFrame, overrides: dict[str, float]
) -> None:
    for field, value in overrides.items():
        canonical.loc[0, field] = value
    assert validate_row_quality(canonical).empty


def test_missing_fields_are_separate_blocking_findings_without_invalid_ohlc(
    canonical: pd.DataFrame,
) -> None:
    canonical.loc[0, "open"] = pd.NA
    canonical.loc[0, "contract"] = None
    canonical.loc[0, "timestamp_utc"] = pd.NaT
    issues = validate_row_quality(canonical)
    assert len(issues) == 3
    assert set(issues["field"]) == {"open", "contract", "timestamp_utc"}
    assert set(issues["rule_id"]) == {RuleId.MISSING_REQUIRED_VALUE.value}
    assert set(issues["severity"]) == {Severity.ERROR.value}
    assert issues["blocking"].all()
    assert issues["actual_value"].isna().all()
    assert issues["timestamp_utc"].isna().all()
    for issue in issues.itertuples(index=False):
        assert issue.field in issue.message


@pytest.mark.parametrize(
    "overrides,relationships",
    [
        ({"high": 98}, {"high < low"}),
        ({"open": 102}, {"high < open"}),
        ({"close": 98}, {"low > close"}),
        (
            {"high": 90, "low": 110},
            {"high < open", "high < close", "high < low", "low > open", "low > close"},
        ),
    ],
)
def test_invalid_ohlc_reports_one_finding_with_violated_relationships(
    canonical: pd.DataFrame, overrides: dict[str, float], relationships: set[str]
) -> None:
    for field, value in overrides.items():
        canonical.loc[0, field] = value
    issues = validate_row_quality(canonical)
    assert len(issues) == 1
    issue = issues.iloc[0]
    assert issue.rule_id == RuleId.INVALID_OHLC.value
    assert issue.severity == Severity.ERROR.value
    assert issue.blocking
    assert issue.field == "ohlc"
    assert issue.actual_value == canonical.loc[0, ["open", "high", "low", "close"]].to_dict()
    for relationship in relationships:
        assert relationship in issue.message


def test_negative_volume_is_blocking_and_retains_context(canonical: pd.DataFrame) -> None:
    canonical.loc[0, "volume"] = -500
    issues = validate_row_quality(canonical)
    assert len(issues) == 1
    issue = issues.iloc[0]
    assert issue.rule_id == RuleId.NEGATIVE_VOLUME.value
    assert issue.severity == Severity.ERROR.value
    assert issue.blocking
    assert (issue.source_file, issue.source_row, issue.contract) == ("sample.parquet", 7, "ESZ6")
    assert issue.timestamp_utc == canonical.loc[0, "timestamp_utc"]
    assert str(issues["timestamp_utc"].dt.tz) == "UTC"
    assert (issue.field, issue.actual_value) == ("volume", -500)
    assert "negative" in issue.message


def test_negative_prices_are_field_level_warnings_with_valid_ohlc(canonical: pd.DataFrame) -> None:
    canonical.loc[0, ["open", "high", "low", "close"]] = [-2, 1, -3, 0]
    issues = validate_row_quality(canonical)
    assert len(issues) == 2
    assert set(issues["rule_id"]) == {RuleId.NEGATIVE_PRICE.value}
    assert set(issues["severity"]) == {Severity.WARNING.value}
    assert not issues["blocking"].any()
    assert dict(zip(issues["field"], issues["actual_value"], strict=True)) == {
        "open": -2,
        "low": -3,
    }
    for issue in issues.itertuples(index=False):
        assert issue.field in issue.message


def test_independent_defects_coexist_and_input_remains_unchanged(canonical: pd.DataFrame) -> None:
    defective = canonical.copy()
    defective.loc[0, ["open", "high", "low", "close", "volume"]] = [-2, -1, -3, 0, -500]
    data = pd.concat([canonical, defective], ignore_index=True)
    data["source_row"] = [7, 8]
    data.index = [9, 9]
    before = data.copy(deep=True)
    issues = validate_row_quality(data)
    assert len(issues) == 5
    assert issues["rule_id"].value_counts().to_dict() == {
        RuleId.INVALID_OHLC.value: 1,
        RuleId.NEGATIVE_VOLUME.value: 1,
        RuleId.NEGATIVE_PRICE.value: 3,
    }
    assert issues["source_row"].tolist() == [8] * 5
    assert set(issues.loc[issues["blocking"], "rule_id"]) == {
        RuleId.INVALID_OHLC.value,
        RuleId.NEGATIVE_VOLUME.value,
    }
    assert_frame_equal(data, before)


def test_missing_optional_lineage_does_not_prevent_findings(canonical: pd.DataFrame) -> None:
    canonical = canonical.drop(columns=["source_file", "source_row"])
    canonical.loc[0, "volume"] = -1
    issues = validate_row_quality(canonical)
    assert len(issues) == 1
    assert issues[["source_file", "source_row"]].isna().all().all()
    assert issues.loc[0, "contract"] == "ESZ6"


def test_empty_canonical_input_returns_the_issue_schema(canonical: pd.DataFrame) -> None:
    issues = validate_row_quality(canonical.iloc[:0])
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
    assert pd.api.types.is_bool_dtype(issues["blocking"].dtype)


@pytest.mark.parametrize("row_count", [0, 1])
def test_zero_findings_preserve_canonical_timestamp_dtype(
    canonical: pd.DataFrame, row_count: int
) -> None:
    issues = validate_row_quality(canonical.iloc[:row_count])
    assert issues.empty
    assert issues["timestamp_utc"].dtype == canonical["timestamp_utc"].dtype
    assert str(issues["timestamp_utc"].dt.tz) == "UTC"


def test_missing_required_columns_raise_a_structural_error(canonical: pd.DataFrame) -> None:
    with pytest.raises(
        SchemaValidationError, match="Missing required canonical columns: timestamp_utc, volume"
    ):
        validate_row_quality(canonical.drop(columns=["timestamp_utc", "volume"]))
