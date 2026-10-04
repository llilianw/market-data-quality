import pandas as pd
import streamlit as st

from market_quality.config import AppConfig
from market_quality.models import GapClassification
from market_quality.pipeline import AnalysisResult


def render_assessment(assessment: AnalysisResult, config: AppConfig) -> None:
    """Present full-dataset evidence independently of analytics scope."""
    issues = assessment.quality_issues
    unexpected = assessment.gaps.loc[
        assessment.gaps["classification"].eq(GapClassification.UNEXPECTED_GAP)
    ]
    exclusions = assessment.exclusions
    excluded_count = len(exclusions.loc[:, ["source_file", "source_row"]].drop_duplicates())

    st.subheader("Data Quality")
    st.caption("Full assessed uploaded dataset; analytics contract/date filters do not apply.")
    for column, (label, value) in zip(
        st.columns(3),
        [
            ("Quality issues", len(issues)),
            ("Unexpected gap intervals", len(unexpected)),
            ("Excluded observations", excluded_count),
        ],
        strict=True,
    ):
        column.metric(label, f"{value:,}")
    st.caption(
        "Quality issues counts finding rows; unexpected gaps counts intervals; "
        "excluded observations counts distinct source_file/source_row pairs."
    )

    st.markdown("**Row-level and duplicate issues**")
    if issues.empty:
        st.success("No row-level or duplicate quality issues detected.")
    else:
        breakdown = (
            issues.groupby(["rule_id", "severity", "blocking"], dropna=False, sort=True)
            .size()
            .rename("issue_count")
            .reset_index()
        )
        st.dataframe(breakdown, hide_index=True)
        with st.expander("View detailed quality issues"):
            details = issues.loc[
                :,
                [
                    "rule_id",
                    "severity",
                    "blocking",
                    "contract",
                    "timestamp_utc",
                    "field",
                    "actual_value",
                    "message",
                    "source_file",
                    "source_row",
                ],
            ].copy()
            # Arrow tables need a uniform display representation for scalar/dictionary evidence.
            details["actual_value"] = details["actual_value"].map(str)
            st.dataframe(details, hide_index=True)

    st.markdown("**Unexpected gaps**")
    st.caption(
        "Unexpected gaps are timestamps missing under the configured "
        f"{config.quality.expected_frequency} cadence and {config.session.timezone} "
        "trading-session model. They do not prove source records were lost. "
        "Scheduled closures are informational and excluded from this defect view."
    )
    if unexpected.empty:
        st.info("No unexpected gap intervals detected under the configured cadence/session model.")
    else:
        st.caption(
            "Gap timestamps and bounds are shown in UTC; "
            "`session_date` is the trading-session closing date."
        )
        st.dataframe(
            unexpected.loc[
                :,
                [
                    "contract",
                    "session_date",
                    "previous_timestamp",
                    "next_timestamp",
                    "gap_start",
                    "gap_end",
                    "missing_count",
                ],
            ],
            hide_index=True,
            column_config={
                "missing_count": st.column_config.NumberColumn("Missing expected timestamps")
            },
        )

    st.markdown("**Analytics exclusions**")
    if exclusions.empty:
        st.info("No observations were excluded from analytics.")
    else:
        with st.expander("View analytics exclusions"):
            st.dataframe(
                exclusions.loc[
                    :,
                    [
                        "contract",
                        "timestamp_utc",
                        "exclusion_reason",
                        "rule_id",
                        "source_file",
                        "source_row",
                    ],
                ],
                hide_index=True,
            )

    st.subheader("Insights")
    st.caption(
        "Deterministic summaries of full-dataset evidence. Recommendations are advisory; "
        "completeness warnings do not exclude observations."
    )
    if assessment.insights.empty:
        st.info("No deterministic quality or completeness insights generated.")
        return
    for insight in assessment.insights.itertuples(index=False):
        contract = "Unknown contract" if pd.isna(insight.contract) else insight.contract
        with st.expander(f"{insight.title} — {contract}"):
            st.caption(f"{insight.severity.title()} · {insight.category.title()}")
            st.write(insight.summary)
            unit = (
                "gap intervals"
                if insight.category == "completeness"
                else "distinct source observations"
            )
            st.caption(
                f"Evidence: {insight.evidence_count:,} {unit} · "
                f"Affected trading sessions: {insight.affected_sessions:,}"
            )
            st.markdown("**Suggested control**")
            st.write(insight.recommendation)
