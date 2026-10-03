"""Derive analytical observations from quality evidence without repairing source data."""

from dataclasses import dataclass

import pandas as pd

from market_quality.exceptions import SchemaValidationError
from market_quality.models import RuleId

_EXCLUSION_COLUMNS = (
    "source_file",
    "source_row",
    "contract",
    "timestamp_utc",
    "exclusion_reason",
    "rule_id",
)


@dataclass(frozen=True, slots=True)
class AnalyticalViewResult:
    """Derived observations and normalized evidence for excluded source rows."""

    data: pd.DataFrame
    exclusions: pd.DataFrame


def build_analytical_view(data: pd.DataFrame, issues: pd.DataFrame) -> AnalyticalViewResult:
    """Apply supplied findings; detection and gap evidence belong to other stages.

    Canonical lineage must be complete and unique. Output observations are ordered
    by source_file then source_row; that same ordering selects exact representatives
    after blocking exclusions. Findings must describe this canonical dataset.
    """
    lineage = ["source_file", "source_row"]
    context = [*lineage, "contract", "timestamp_utc"]
    missing_data = [field for field in context if field not in data.columns]
    if missing_data:
        raise SchemaValidationError(
            f"Missing required canonical columns: {', '.join(missing_data)}"
        )
    required_issues = ["rule_id", "blocking", *context]
    missing_issues = [field for field in required_issues if field not in issues.columns]
    if missing_issues:
        raise SchemaValidationError(f"Missing required issue columns: {', '.join(missing_issues)}")
    if data.loc[:, lineage].isna().any().any():
        raise SchemaValidationError("Canonical source_file/source_row lineage must be non-missing")
    if data.duplicated(lineage).any():
        raise SchemaValidationError("Canonical source_file/source_row lineage must be unique")

    source = data.sort_values(lineage).reset_index(drop=True)
    blocking = issues["blocking"].eq(True).fillna(False)
    blocked_keys = pd.MultiIndex.from_frame(issues.loc[blocking, lineage])
    source_keys = pd.MultiIndex.from_frame(source.loc[:, lineage])
    eligible = source.loc[~source_keys.isin(blocked_keys)].copy()

    exact_keys = pd.MultiIndex.from_frame(
        issues.loc[issues["rule_id"].eq(RuleId.EXACT_DUPLICATE.value), lineage]
    )
    eligible_keys = pd.MultiIndex.from_frame(eligible.loc[:, lineage])
    exact = eligible.loc[eligible_keys.isin(exact_keys)]
    # Exact findings already establish OHLCV equality; policy does not detect it again.
    redundant = exact.loc[exact.duplicated(["contract", "timestamp_utc"], keep="first")]
    redundant_keys = pd.MultiIndex.from_frame(redundant.loc[:, lineage])
    analytical = eligible.loc[~eligible_keys.isin(redundant_keys)].copy().reset_index(drop=True)

    blocked = issues.loc[blocking, [*context, "rule_id"]].copy()
    blocked["exclusion_reason"] = "blocking_quality_issue"
    redundant_evidence = redundant.loc[:, context].copy()
    redundant_evidence["exclusion_reason"] = "redundant_exact_duplicate"
    redundant_evidence["rule_id"] = RuleId.EXACT_DUPLICATE.value
    exclusions = pd.concat(
        [blocked.loc[:, _EXCLUSION_COLUMNS], redundant_evidence.loc[:, _EXCLUSION_COLUMNS]],
        ignore_index=True,
    ).drop_duplicates()
    exclusions = exclusions.sort_values([*lineage, "exclusion_reason", "rule_id"]).reset_index(
        drop=True
    )
    if exclusions.empty:
        exclusions["timestamp_utc"] = exclusions["timestamp_utc"].astype(
            data["timestamp_utc"].dtype
        )
    return AnalyticalViewResult(data=analytical, exclusions=exclusions)
