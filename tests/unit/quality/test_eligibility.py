import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.exceptions import SchemaValidationError
from market_quality.models import RuleId
from market_quality.quality.duplicates import detect_duplicates
from market_quality.quality.eligibility import build_analytical_view
from market_quality.quality.rules import validate_row_quality


@pytest.fixture
def canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": ["b.csv", "a.csv", "a.csv"],
            "source_row": [0, 2, 1],
            "contract": ["ESZ6", "NQZ6", "ESZ6"],
            "timestamp_utc": pd.to_datetime(
                ["2026-01-05T16:31:00Z", "2026-01-05T16:31:00Z", "2026-01-05T16:33:00Z"],
                utc=True,
            ),
            "open": pd.Series([100] * 3, dtype="Float64"),
            "high": pd.Series([101] * 3, dtype="Float64"),
            "low": pd.Series([99] * 3, dtype="Float64"),
            "close": pd.Series([100] * 3, dtype="Float64"),
            "volume": pd.Series([10] * 3, dtype="Int64"),
            "exchange": ["CME"] * 3,
            "root": ["ES", "NQ", "ES"],
        }
    )


@pytest.fixture
def exact_copies(canonical: pd.DataFrame) -> pd.DataFrame:
    copies = pd.concat([canonical.iloc[[0]]] * 3, ignore_index=True)
    copies["source_file"] = ["b.csv", "a.csv", "a.csv"]
    copies["source_row"] = [0, 9, 2]
    return copies


def _blocking_finding(data: pd.DataFrame, rows: list[int]) -> pd.DataFrame:
    finding = data.loc[rows, ["source_file", "source_row", "contract", "timestamp_utc"]].copy()
    finding["rule_id"] = "CUSTOM_BLOCKING_RULE"
    finding["blocking"] = True
    return finding


def test_clean_data_and_empty_issues_preserve_observations_and_inputs(
    canonical: pd.DataFrame,
) -> None:
    canonical.index = [9, 2, 9]
    issues = validate_row_quality(canonical)
    before_data = canonical.copy(deep=True)
    before_issues = issues.copy(deep=True)
    result = build_analytical_view(canonical, issues)
    expected = canonical.sort_values(["source_file", "source_row"]).reset_index(drop=True)
    assert_frame_equal(result.data, expected)
    assert result.exclusions.empty
    assert_frame_equal(canonical, before_data)
    assert_frame_equal(issues, before_issues)
    result.data.loc[0, "volume"] = 500
    assert_frame_equal(canonical, before_data)


def test_missing_required_value_excludes_the_referenced_lineage(canonical: pd.DataFrame) -> None:
    canonical.loc[0, "open"] = pd.NA
    result = build_analytical_view(canonical, validate_row_quality(canonical))
    assert result.data["source_file"].tolist() == ["a.csv", "a.csv"]
    assert len(result.exclusions) == 1
    exclusion = result.exclusions.iloc[0]
    assert (exclusion.source_file, exclusion.source_row) == ("b.csv", 0)
    assert exclusion.exclusion_reason == "blocking_quality_issue"
    assert exclusion.rule_id == RuleId.MISSING_REQUIRED_VALUE.value


def test_generic_blocking_policy_requires_no_rule_specific_code(canonical: pd.DataFrame) -> None:
    issues = _blocking_finding(canonical, [1])
    result = build_analytical_view(canonical, issues)
    assert list(zip(result.data["source_file"], result.data["source_row"], strict=True)) == [
        ("a.csv", 1),
        ("b.csv", 0),
    ]
    assert result.exclusions["rule_id"].tolist() == ["CUSTOM_BLOCKING_RULE"]


@pytest.mark.parametrize("blocking", [False, True])
def test_negative_price_warnings_remain_eligible_unless_blocking_wins(
    canonical: pd.DataFrame, blocking: bool
) -> None:
    canonical.loc[0, ["open", "high", "low", "close"]] = [-2, 1, -3, 0]
    if blocking:
        canonical.loc[0, "volume"] = -500
    result = build_analytical_view(canonical, validate_row_quality(canonical))
    assert len(result.data) == (2 if blocking else 3)
    if blocking:
        assert result.exclusions["rule_id"].tolist() == [RuleId.NEGATIVE_VOLUME.value]
        assert "b.csv" not in result.data["source_file"].tolist()
    else:
        assert result.exclusions.empty
        assert result.data.loc[result.data["source_file"] == "b.csv", "open"].tolist() == [-2]


def test_unreported_exact_duplicates_remain_eligible(exact_copies: pd.DataFrame) -> None:
    data = exact_copies.iloc[:2].copy()
    issues = pd.DataFrame(
        columns=["rule_id", "blocking", "source_file", "source_row", "contract", "timestamp_utc"]
    )
    result = build_analytical_view(data, issues)
    expected = data.sort_values(["source_file", "source_row"]).reset_index(drop=True)
    assert_frame_equal(result.data, expected)
    assert result.exclusions.empty


def test_exact_duplicates_retain_the_same_lineage_ordered_representative(
    exact_copies: pd.DataFrame,
) -> None:
    issues = detect_duplicates(exact_copies)
    before_data = exact_copies.copy(deep=True)
    before_issues = issues.copy(deep=True)
    result = build_analytical_view(exact_copies, issues)
    assert_frame_equal(result.data, exact_copies.iloc[[2]].reset_index(drop=True))
    assert list(
        zip(result.exclusions["source_file"], result.exclusions["source_row"], strict=True)
    ) == [("a.csv", 9), ("b.csv", 0)]
    assert set(result.exclusions["exclusion_reason"]) == {"redundant_exact_duplicate"}
    assert set(result.exclusions["rule_id"]) == {RuleId.EXACT_DUPLICATE.value}
    reordered = build_analytical_view(exact_copies.iloc[[2, 0, 1]], issues.iloc[::-1])
    assert_frame_equal(result.data, reordered.data)
    assert_frame_equal(result.exclusions, reordered.exclusions)
    assert_frame_equal(exact_copies, before_data)
    assert_frame_equal(issues, before_issues)


@pytest.mark.parametrize("block_all", [False, True])
def test_blocking_is_applied_before_exact_duplicate_survivor_selection(
    exact_copies: pd.DataFrame, block_all: bool
) -> None:
    blocked_rows = [0, 1, 2] if block_all else [2]
    issues = pd.concat(
        [detect_duplicates(exact_copies), _blocking_finding(exact_copies, blocked_rows)],
        ignore_index=True,
    )
    result = build_analytical_view(exact_copies, issues)
    if block_all:
        assert result.data.empty
        assert result.exclusions["exclusion_reason"].tolist() == ["blocking_quality_issue"] * 3
    else:
        assert_frame_equal(result.data, exact_copies.iloc[[1]].reset_index(drop=True))
        assert result.exclusions["exclusion_reason"].tolist() == [
            "blocking_quality_issue",
            "redundant_exact_duplicate",
        ]


def test_conflicting_duplicates_exclude_every_participant(exact_copies: pd.DataFrame) -> None:
    exact_copies.loc[2, "high"] = 102
    result = build_analytical_view(exact_copies, detect_duplicates(exact_copies))
    assert result.data.empty
    assert len(result.exclusions) == 3
    assert set(result.exclusions["rule_id"]) == {RuleId.CONFLICTING_DUPLICATE.value}
    assert set(result.exclusions["exclusion_reason"]) == {"blocking_quality_issue"}


def test_multiple_blocking_rules_keep_separate_evidence_without_multiplying_data(
    canonical: pd.DataFrame,
) -> None:
    canonical.loc[0, "high"] = 98
    canonical.loc[0, "volume"] = -500
    result = build_analytical_view(canonical, validate_row_quality(canonical))
    assert len(result.data) == 2
    assert not result.data.duplicated(["source_file", "source_row"]).any()
    assert len(result.exclusions) == 2
    assert set(result.exclusions["rule_id"]) == {
        RuleId.INVALID_OHLC.value,
        RuleId.NEGATIVE_VOLUME.value,
    }
    assert result.exclusions["source_file"].tolist() == ["b.csv"] * 2


@pytest.mark.parametrize(
    "target,column",
    [("data", "source_file"), ("data", "source_row"), ("issues", "blocking")],
)
def test_missing_required_columns_fail_clearly(
    canonical: pd.DataFrame, target: str, column: str
) -> None:
    issues = validate_row_quality(canonical)
    if target == "data":
        canonical = canonical.drop(columns=[column])
    else:
        issues = issues.drop(columns=[column])
    with pytest.raises(SchemaValidationError, match=f"Missing required .* columns: {column}"):
        build_analytical_view(canonical, issues)


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_ambiguous_canonical_lineage_is_rejected(canonical: pd.DataFrame, problem: str) -> None:
    if problem == "missing":
        canonical.loc[0, "source_file"] = None
        message = "lineage must be non-missing"
    else:
        canonical.loc[1, ["source_file", "source_row"]] = canonical.loc[
            2, ["source_file", "source_row"]
        ]
        message = "lineage must be unique"
    with pytest.raises(SchemaValidationError, match=message):
        build_analytical_view(canonical, validate_row_quality(canonical))


def test_empty_canonical_input_preserves_typed_outputs(canonical: pd.DataFrame) -> None:
    data = canonical.iloc[:0]
    result = build_analytical_view(data, validate_row_quality(data))
    assert_frame_equal(result.data, data.reset_index(drop=True))
    assert result.exclusions.empty
    assert result.exclusions.columns.tolist() == [
        "source_file",
        "source_row",
        "contract",
        "timestamp_utc",
        "exclusion_reason",
        "rule_id",
    ]
    assert result.exclusions["timestamp_utc"].dtype == data["timestamp_utc"].dtype
