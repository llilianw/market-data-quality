import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.exceptions import SchemaValidationError
from market_quality.insights.engine import generate_quality_insights
from market_quality.models import GapClassification, RuleId, Severity


@pytest.fixture
def enriched() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": ["a.csv", "b.csv", "a.csv", "a.csv"],
            "source_row": [0, 0, 1, 2],
            "contract": ["ESZ6", "ESZ6", "ESZ6", "NQZ6"],
            "session_date": pd.to_datetime(["2026-01-05", "2026-01-06", None, "2026-01-05"]),
        },
        index=[7, 7, 1, 9],
    )


@pytest.fixture
def empty_issues() -> pd.DataFrame:
    return pd.DataFrame(columns=["source_file", "source_row", "contract", "rule_id", "severity"])


@pytest.fixture
def empty_gaps() -> pd.DataFrame:
    return pd.DataFrame(columns=["contract", "classification", "session_date", "missing_count"])


@pytest.mark.parametrize(("count", "wording"), [(1, "detected"), (2, "repeated")])
def test_quality_patterns_report_distinct_observation_counts(
    enriched: pd.DataFrame, empty_gaps: pd.DataFrame, count: int, wording: str
) -> None:
    issues = (
        enriched.iloc[:count]
        .drop(columns="session_date")
        .assign(rule_id=RuleId.NEGATIVE_VOLUME.value, severity=Severity.ERROR.value)
    )
    insights = generate_quality_insights(enriched, issues, empty_gaps)
    assert len(insights) == 1
    insight = insights.iloc[0]
    assert insight.category == "quality"
    assert insight.contract == "ESZ6"
    assert insight.rule_id == RuleId.NEGATIVE_VOLUME.value
    assert insight.severity == Severity.ERROR.value
    assert insight.evidence_count == count
    assert insight.affected_sessions == count
    assert wording in insight.summary
    assert f"{count} distinct source observation" in insight.summary


def test_field_findings_count_observations_and_attribute_sessions_by_lineage(
    enriched: pd.DataFrame, empty_gaps: pd.DataFrame
) -> None:
    issues = (
        enriched.iloc[[0, 0, 1, 1, 2]]
        .drop(columns="session_date")
        .assign(
            rule_id=RuleId.NEGATIVE_PRICE.value,
            severity=Severity.WARNING.value,
            field=["open", "close", "high", "low", "close"],
        )
    )
    before = [data.copy(deep=True) for data in (enriched, issues, empty_gaps)]
    insights = generate_quality_insights(enriched, issues, empty_gaps)
    assert insights["evidence_count"].tolist() == [3]
    assert insights["affected_sessions"].tolist() == [2]
    assert "3 distinct source observations" in insights.loc[0, "summary"]
    assert "2 trading sessions" in insights.loc[0, "summary"]
    assert_frame_equal(
        insights,
        generate_quality_insights(
            enriched.sample(frac=1, random_state=3),
            issues.sample(frac=1, random_state=5),
            empty_gaps,
        ),
    )
    for actual, expected in zip((enriched, issues, empty_gaps), before, strict=True):
        assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("rule_id", "expected_controls"),
    [
        (RuleId.MISSING_REQUIRED_VALUE, ["contract, timestamp, OHLC and volume", "quarantine"]),
        (
            RuleId.INVALID_OHLC,
            ["high >= open, close and low", "low <= open and close", "quarantine"],
        ),
        (RuleId.NEGATIVE_VOLUME, ["volume >= 0", "reject or quarantine", "Zero volume is valid"]),
        (
            RuleId.EXACT_DUPLICATE,
            ["uniqueness", "otherwise eligible", "source_file then source_row"],
        ),
        (RuleId.CONFLICTING_DUPLICATE, ["quarantine", "reconcile", "rather than averaging"]),
        (
            RuleId.NEGATIVE_PRICE,
            ["review rather than automatically rejecting", "legitimate futures"],
        ),
    ],
)
def test_recommendations_respect_existing_rule_policy(
    enriched: pd.DataFrame,
    empty_gaps: pd.DataFrame,
    rule_id: RuleId,
    expected_controls: list[str],
) -> None:
    issues = (
        enriched.iloc[:1]
        .drop(columns="session_date")
        .assign(rule_id=rule_id.value, severity=Severity.WARNING.value)
    )
    recommendation = generate_quality_insights(enriched, issues, empty_gaps).loc[
        0, "recommendation"
    ]
    for control in expected_controls:
        assert control in recommendation


def test_contract_isolation_missing_contract_and_severity_precedence(
    enriched: pd.DataFrame, empty_gaps: pd.DataFrame
) -> None:
    enriched = enriched.copy()
    enriched.iloc[2, enriched.columns.get_loc("contract")] = None
    issues = (
        enriched.iloc[[0, 1, 3, 2]]
        .drop(columns="session_date")
        .assign(
            rule_id=[RuleId.NEGATIVE_VOLUME.value] * 3 + [RuleId.MISSING_REQUIRED_VALUE.value],
            severity=[
                Severity.WARNING.value,
                Severity.ERROR.value,
                Severity.INFO.value,
                Severity.ERROR.value,
            ],
        )
    )
    insights = generate_quality_insights(enriched, issues, empty_gaps)
    assert len(insights) == 3
    assert insights["severity"].tolist() == ["error", "error", "info"]
    assert insights.loc[0, "contract"] == "ESZ6"
    assert insights.loc[0, "evidence_count"] == 2
    assert pd.isna(insights.loc[1, "contract"])
    assert insights.loc[1, "rule_id"] == RuleId.MISSING_REQUIRED_VALUE.value
    assert insights.loc[1, "affected_sessions"] == 0
    assert insights.loc[2, "contract"] == "NQZ6"
    assert_frame_equal(
        insights,
        generate_quality_insights(enriched, issues.sample(frac=1, random_state=2), empty_gaps),
    )


@pytest.mark.parametrize("rule_id", [RuleId.STATISTICAL_OUTLIER.value, "FUTURE_RULE"])
def test_unmapped_rules_receive_a_conservative_review_recommendation(
    enriched: pd.DataFrame, empty_gaps: pd.DataFrame, rule_id: str
) -> None:
    issues = (
        enriched.iloc[:1]
        .drop(columns="session_date")
        .assign(rule_id=rule_id, severity=Severity.WARNING.value)
    )
    insights = generate_quality_insights(enriched, issues, empty_gaps)
    assert insights["rule_id"].tolist() == [rule_id]
    assert "Review the affected observations" in insights.loc[0, "recommendation"]
    assert insights["evidence_count"].tolist() == [1]


def test_unexpected_gap_intervals_aggregate_by_contract_without_expected_closures(
    enriched: pd.DataFrame, empty_issues: pd.DataFrame
) -> None:
    gaps = pd.DataFrame(
        {
            "contract": ["ESZ6"] * 4 + ["NQZ6"],
            "classification": [GapClassification.UNEXPECTED_GAP.value] * 3
            + [GapClassification.EXPECTED_CLOSURE.value, GapClassification.UNEXPECTED_GAP.value],
            "session_date": pd.to_datetime(
                ["2026-01-05", "2026-01-05", "2026-01-06", None, "2026-01-05"]
            ),
            "missing_count": [2, 4, 5, 60, 1],
        }
    )
    before = gaps.copy(deep=True)
    insights = generate_quality_insights(enriched, empty_issues, gaps)
    assert insights["contract"].tolist() == ["ESZ6", "NQZ6"]
    assert insights["category"].tolist() == ["completeness"] * 2
    assert insights["severity"].tolist() == ["warning"] * 2
    assert insights["rule_id"].isna().all()
    assert insights["evidence_count"].tolist() == [3, 1]
    assert insights["affected_sessions"].tolist() == [2, 1]
    assert "3 unexpected gap intervals" in insights.loc[0, "summary"]
    assert "11 missing expected bars" in insights.loc[0, "summary"]
    assert "2 trading sessions" in insights.loc[0, "summary"]
    assert "configured trading-session schedule" in insights.loc[0, "recommendation"]
    assert_frame_equal(
        insights,
        generate_quality_insights(enriched, empty_issues, gaps.sample(frac=1, random_state=4)),
    )
    assert_frame_equal(gaps, before)


@pytest.mark.parametrize("closure_only", [False, True])
def test_empty_evidence_and_expected_closures_return_empty_schema(
    enriched: pd.DataFrame,
    empty_issues: pd.DataFrame,
    empty_gaps: pd.DataFrame,
    closure_only: bool,
) -> None:
    gaps = empty_gaps
    if closure_only:
        gaps = pd.DataFrame(
            {
                "contract": ["ESZ6"],
                "classification": [GapClassification.EXPECTED_CLOSURE.value],
                "session_date": [pd.NaT],
                "missing_count": [60],
            }
        )
    insights = generate_quality_insights(enriched.iloc[:0], empty_issues, gaps)
    assert insights.empty
    assert insights.columns.tolist() == [
        "category",
        "severity",
        "contract",
        "rule_id",
        "title",
        "summary",
        "evidence_count",
        "affected_sessions",
        "recommendation",
    ]
    assert insights["evidence_count"].dtype == "int64"
    assert insights["affected_sessions"].dtype == "int64"


@pytest.mark.parametrize(
    ("input_name", "missing_column"),
    [("enriched_data", "session_date"), ("quality_issues", "rule_id"), ("gaps", "missing_count")],
)
def test_missing_required_structure_fails_clearly(
    enriched: pd.DataFrame,
    empty_issues: pd.DataFrame,
    empty_gaps: pd.DataFrame,
    input_name: str,
    missing_column: str,
) -> None:
    inputs = {"enriched_data": enriched, "quality_issues": empty_issues, "gaps": empty_gaps}
    inputs[input_name] = inputs[input_name].drop(columns=missing_column)
    with pytest.raises(SchemaValidationError, match=rf"{input_name}.*{missing_column}"):
        generate_quality_insights(**inputs)


def test_duplicate_observation_lineage_cannot_attribute_sessions_ambiguously(
    enriched: pd.DataFrame, empty_issues: pd.DataFrame, empty_gaps: pd.DataFrame
) -> None:
    ambiguous = pd.concat([enriched, enriched.iloc[:1]], ignore_index=True)
    with pytest.raises(SchemaValidationError, match=r"unique \(source_file, source_row\)"):
        generate_quality_insights(ambiguous, empty_issues, empty_gaps)
