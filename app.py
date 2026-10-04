import pandas as pd
import streamlit as st

from market_quality.analytics.filtering import filter_analysis_scope
from market_quality.analytics.ohlcv import aggregate_daily_ohlcv
from market_quality.analytics.vwap import compute_rolling_vwap
from market_quality.config import AppConfig
from market_quality.exceptions import MarketDataError
from market_quality.ingestion.normalize import CanonicalizationResult
from market_quality.models import GapClassification
from market_quality.pipeline import AnalysisResult
from market_quality.ui.analytics import render_analytics
from market_quality.ui.assessment import render_assessment
from market_quality.ui.uploads import process_uploaded_data


@st.cache_data(show_spinner="Ingesting and assessing the full dataset...", max_entries=2)
def assess_upload(
    file_bytes: bytes,
    filename: str,
    config: AppConfig,
) -> tuple[CanonicalizationResult, AnalysisResult]:
    return process_uploaded_data(file_bytes, filename, config)


def main() -> None:
    st.set_page_config(page_title="Market Data Quality", layout="wide")
    st.title("Market Data Quality & Analytics")
    st.write(
        "Upload CSV or Parquet market bars to assess the full dataset, then select "
        "contracts and trading session dates for analytics."
    )
    uploaded = st.file_uploader("Market-data file", type=["csv", "parquet"])
    if uploaded is None:
        st.info("Upload a market-data file to begin.")
        return

    config = AppConfig()
    try:
        canonical, assessment = assess_upload(uploaded.getvalue(), uploaded.name, config)
    except (MarketDataError, ValueError) as exc:
        st.error(f"Could not process the uploaded file: {exc}")
        return

    st.success("Dataset assessment complete.")
    unexpected_count = assessment.gaps["classification"].eq(GapClassification.UNEXPECTED_GAP).sum()
    metrics = [
        ("Source rows", len(canonical.data) + len(canonical.rejected_rows)),
        ("Canonical rows", len(canonical.data)),
        ("Rejected rows", len(canonical.rejected_rows)),
        ("Quality issues", len(assessment.quality_issues)),
        ("Unexpected gap intervals", unexpected_count),
        ("Eligible observations", len(assessment.eligible_data)),
    ]
    for offset in (0, 3):
        for column, (label, value) in zip(st.columns(3), metrics[offset : offset + 3], strict=True):
            column.metric(label, f"{value:,}")
    st.caption(
        f"Unexpected gaps are missing expected timestamps under the configured "
        f"{config.quality.expected_frequency} cadence and {config.session.timezone} "
        f"overnight trading-session schedule "
        f"({config.session.session_start:%H:%M}–{config.session.session_end:%H:%M}). "
        "They do not prove that source records were lost."
    )

    if canonical.data.empty:
        st.info(
            "All source observations were rejected during ingestion."
            if not canonical.rejected_rows.empty
            else "The uploaded file contains no observations."
        )
    elif assessment.eligible_data.empty:
        st.info("No observations are eligible for analytics. Full assessment evidence is retained.")
    else:
        _render_analytics_scope(assessment.eligible_data, config)

    render_assessment(assessment, config)


def _render_analytics_scope(eligible: pd.DataFrame, config: AppConfig) -> None:
    st.subheader("Analytics scope")
    contracts = sorted(eligible["contract"].dropna().unique().tolist())
    selected = st.multiselect(
        "Contracts",
        contracts,
        default=contracts,
        help="All selected means unrestricted. Clear all contracts to show no analytics.",
    )
    # An empty selection means no analytics; selecting all contracts leaves them unrestricted.
    contract_scope = None if len(selected) == len(contracts) else selected
    dates = eligible["session_date"].dropna()
    minimum = dates.min().date() if not dates.empty else None
    maximum = dates.max().date() if not dates.empty else None
    if dates.empty:
        st.info("No assigned trading session dates are available; date filtering is unavailable.")
    left, right = st.columns(2)
    start = left.date_input(
        "Start session date",
        value=minimum,
        min_value=minimum,
        max_value=maximum,
        disabled=dates.empty,
    )
    end = right.date_input(
        "End session date",
        value=maximum,
        min_value=minimum,
        max_value=maximum,
        disabled=dates.empty,
    )
    st.caption(
        "Dates are inclusive trading session closing dates; overnight sessions stay together."
    )
    if start is not None and end is not None and start > end:
        st.error("Start session date must be on or before end session date.")
        return

    scoped = filter_analysis_scope(eligible, contract_scope, start, end)
    daily = aggregate_daily_ohlcv(scoped, config.analytics)
    rolling = compute_rolling_vwap(scoped, config.analytics)
    st.subheader("Scoped analytics")
    for column, (label, value) in zip(
        st.columns(3),
        [
            ("Scoped observations", len(scoped)),
            ("Daily OHLCV bars", len(daily)),
            ("Rolling VWAP observations", len(rolling)),
        ],
        strict=True,
    ):
        column.metric(label, f"{value:,}")
    if scoped.empty:
        st.info(
            "No eligible observations match this selection. Full assessment evidence is retained."
        )
        return

    render_analytics(scoped, daily, rolling, config)


if __name__ == "__main__":
    main()
