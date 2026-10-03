import pandas as pd

from market_quality.exceptions import SchemaValidationError
from market_quality.models import GapClassification, RuleId, Severity

_INSIGHT_COLUMNS = (
    "category",
    "severity",
    "contract",
    "rule_id",
    "title",
    "summary",
    "evidence_count",
    "affected_sessions",
    "recommendation",
)
_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
_RULE_TEXT: dict[str, tuple[str, str]] = {
    RuleId.MISSING_REQUIRED_VALUE: (
        "Incomplete required market fields",
        (
            "Require contract, timestamp, OHLC and volume before analytics; "
            "quarantine incomplete observations."
        ),
    ),
    RuleId.INVALID_OHLC: (
        "Inconsistent OHLC relationships",
        (
            "Validate high >= open, close and low, and low <= open and close; "
            "quarantine inconsistent bars before analytics."
        ),
    ),
    RuleId.NEGATIVE_VOLUME: (
        "Negative volume",
        (
            "Require volume >= 0 and reject or quarantine negative-volume observations "
            "before analytics. Zero volume is valid."
        ),
    ),
    RuleId.EXACT_DUPLICATE: (
        "Identical observations for the same contract/timestamp_utc",
        (
            "Enforce (contract, timestamp_utc) uniqueness; among otherwise eligible identical "
            "observations, retain one representative by source_file then source_row ascending."
        ),
    ),
    RuleId.CONFLICTING_DUPLICATE: (
        "Conflicting OHLCV observations for the same contract/timestamp_utc",
        (
            "Enforce (contract, timestamp_utc) uniqueness; quarantine and reconcile conflicting "
            "source records rather than averaging values or selecting an arbitrary survivor."
        ),
    ),
    RuleId.NEGATIVE_PRICE: (
        "Negative prices flagged as non-blocking evidence",
        (
            "Flag negative futures prices for review rather than automatically rejecting them; "
            "legitimate futures prices can historically be negative."
        ),
    ),
}


def generate_quality_insights(
    enriched_data: pd.DataFrame,
    quality_issues: pd.DataFrame,
    gaps: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize supplied evidence without detection, repair or analytics interpretation.

    Findings must originate from the same dataset, whose lineage uniquely identifies
    observations. Missing session dates are allowed and do not count as sessions.
    """
    lineage = ["source_file", "source_row"]
    issue_columns = [*lineage, "contract", "rule_id", "severity"]
    for name, data, required in (
        ("enriched_data", enriched_data, [*lineage, "session_date"]),
        ("quality_issues", quality_issues, issue_columns),
        ("gaps", gaps, ["contract", "classification", "session_date", "missing_count"]),
    ):
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise SchemaValidationError(f"{name} is missing required columns: {', '.join(missing)}")
    if enriched_data.duplicated(lineage).any():
        raise SchemaValidationError(
            "enriched_data must have unique (source_file, source_row) lineage"
        )

    # Lineage, rather than market identity or input index, attributes findings to sessions.
    evidence = quality_issues[issue_columns].merge(
        enriched_data[[*lineage, "session_date"]], on=lineage, how="left"
    )
    records = []
    for (contract, rule_id), group in evidence.groupby(
        ["contract", "rule_id"], dropna=False, sort=False, observed=True
    ):
        count = len(group[lineage].drop_duplicates())
        sessions = group["session_date"].nunique()
        title, recommendation = _RULE_TEXT.get(
            rule_id,
            (
                f"Quality findings for {rule_id}",
                (
                    "Review the affected observations and define an appropriate validation or "
                    "cleansing control before downstream use."
                ),
            ),
        )
        wording = "detected" if count == 1 else "repeated"
        records.append(
            {
                "category": "quality",
                "severity": min(group["severity"], key=_SEVERITY_ORDER.__getitem__),
                "contract": contract,
                "rule_id": rule_id,
                "title": title,
                "summary": f"{title}: {wording} in {count} distinct source "
                f"observation{'s' if count != 1 else ''} across {sessions} "
                f"trading session{'s' if sessions != 1 else ''}.",
                "evidence_count": count,
                "affected_sessions": sessions,
                "recommendation": recommendation,
            }
        )

    unexpected = gaps.loc[gaps["classification"] == GapClassification.UNEXPECTED_GAP]
    for contract, group in unexpected.groupby("contract", dropna=False, sort=False, observed=True):
        count = len(group)
        sessions = group["session_date"].nunique()
        missing_bars = int(group["missing_count"].sum())
        records.append(
            {
                "category": "completeness",
                "severity": Severity.WARNING.value,
                "contract": contract,
                "rule_id": None,
                "title": "Unexpected gaps",
                "summary": f"{count} unexpected gap interval{'s' if count != 1 else ''} "
                f"account{'s' if count == 1 else ''} for {missing_bars} missing expected bars "
                f"across {sessions} "
                f"trading session{'s' if sessions != 1 else ''}.",
                "evidence_count": count,
                "affected_sessions": sessions,
                "recommendation": "Validate expected bar completeness against the configured "
                "trading-session schedule and investigate unexpected missing intervals "
                "before downstream use.",
            }
        )
    insights = pd.DataFrame(records, columns=_INSIGHT_COLUMNS).astype(
        {"evidence_count": "int64", "affected_sessions": "int64"}
    )
    return (
        insights.assign(_severity_order=insights["severity"].map(_SEVERITY_ORDER))
        .sort_values(["_severity_order", "contract", "category", "rule_id"], kind="stable")
        .drop(columns="_severity_order")
        .reset_index(drop=True)
    )
