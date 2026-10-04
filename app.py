import pandas as pd
import streamlit as st

from market_quality.analytics.filtering import filter_analysis_scope
from market_quality.analytics.ohlcv import aggregate_daily_ohlcv
from market_quality.analytics.vwap import compute_rolling_vwap
from market_quality.config import AppConfig
from market_quality.exceptions import MarketDataError
from market_quality.ingestion.normalize import CanonicalizationResult
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
        "Inspect historical futures OHLCV data, identify quality and completeness "
        "evidence, and explore trading-session and intraday analytics."
    )
    uploaded = st.file_uploader("Market-data file", type=["csv", "parquet"])
    if uploaded is None:
        st.info("Upload a CSV or Parquet market-data file to begin.")
        return

    config = AppConfig()
    try:
        canonical, assessment = assess_upload(uploaded.getvalue(), uploaded.name, config)
    except (MarketDataError, ValueError) as exc:
        st.error(f"Could not process the uploaded file: {exc}")
        st.info("Check the source file and upload a corrected CSV or Parquet file to continue.")
        return

    st.subheader("Dataset overview")
    st.write("Uploaded file:", uploaded.name)
    st.success("Ingestion and assessment complete.")
    metrics = [
        ("Source rows", len(canonical.data) + len(canonical.rejected_rows), None),
        (
            "Canonical rows",
            len(canonical.data),
            "Parsed observations; they may still have quality defects.",
        ),
        ("Rejected rows", len(canonical.rejected_rows), "Rows rejected during ingestion parsing."),
        (
            "Eligible observations",
            len(assessment.eligible_data),
            "Observations retained after blocking findings and exact-duplicate policy.",
        ),
        (
            "Contracts",
            assessment.enriched_data["contract"].nunique(),
            "Distinct canonical contracts.",
        ),
        (
            "Observed sessions",
            assessment.enriched_data["session_date"].nunique(),
            "Distinct non-null trading-session closing dates in the canonical dataset.",
        ),
    ]
    for offset in (0, 3):
        for column, (label, value, help_text) in zip(
            st.columns(3), metrics[offset : offset + 3], strict=True
        ):
            column.metric(label, f"{value:,}", help=help_text)
    st.caption(
        "Quality assessment covers the full uploaded dataset. "
        "Contract and date controls scope analytics only."
    )

    if canonical.data.empty:
        st.info(
            "All source observations were rejected during ingestion. "
            "Review timestamps and numeric values, then upload a corrected file."
            if not canonical.rejected_rows.empty
            else "The uploaded file contains no observations. Upload a file with market-data rows."
        )
    elif assessment.eligible_data.empty:
        st.info(
            "No observations are eligible for analytics. Review the quality and exclusion evidence below."
        )
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
    for column, (label, value) in zip(
        st.columns(3),
        [
            ("Scoped observations", len(scoped)),
            ("Scoped contracts", scoped["contract"].nunique()),
            ("Scoped sessions", scoped["session_date"].nunique()),
        ],
        strict=True,
    ):
        column.metric(label, f"{value:,}")
    if scoped.empty:
        st.info(
            "No eligible observations match this selection. "
            "Select a contract or adjust the session-date range to view analytics."
        )
        return

    render_analytics(scoped, daily, rolling, config)


if __name__ == "__main__":
    main()
