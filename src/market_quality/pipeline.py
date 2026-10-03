"""Compose canonical-data assessment and analytics without duplicating domain rules."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from market_quality.analytics.filtering import filter_analysis_scope
from market_quality.analytics.ohlcv import aggregate_daily_ohlcv
from market_quality.analytics.vwap import compute_rolling_vwap
from market_quality.config import AppConfig
from market_quality.quality.duplicates import detect_duplicates
from market_quality.quality.eligibility import build_analytical_view
from market_quality.quality.gaps import detect_gaps
from market_quality.quality.rules import validate_row_quality
from market_quality.sessions.assignment import assign_sessions


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Assessment evidence covers the input; analytics reflect the selected scope."""

    enriched_data: pd.DataFrame
    quality_issues: pd.DataFrame
    gaps: pd.DataFrame
    exclusions: pd.DataFrame
    eligible_data: pd.DataFrame
    scoped_data: pd.DataFrame
    daily_ohlcv: pd.DataFrame
    rolling_vwap: pd.DataFrame


def run_analysis(
    canonical_data: pd.DataFrame,
    *,
    contracts: Sequence[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    config: AppConfig | None = None,
) -> AnalysisResult:
    """Assess canonical observations before applying scope to downstream analytics.

    Canonical UTC/numeric values and complete unique lineage are input
    preconditions. Raw-file ingestion remains a separate boundary. Component
    errors propagate; analysis filters never erase quality or gap evidence.
    """
    config = AppConfig() if config is None else config
    enriched = assign_sessions(canonical_data, config.session)
    row_issues = validate_row_quality(enriched)
    duplicate_issues = detect_duplicates(enriched)
    issues = (
        pd.concat([row_issues, duplicate_issues], ignore_index=True)
        .sort_values(["source_file", "source_row", "rule_id", "field"], kind="stable")
        .reset_index(drop=True)
    )
    gaps = detect_gaps(enriched, config.session, config.quality)
    analytical = build_analytical_view(enriched, issues)
    scoped = filter_analysis_scope(analytical.data, contracts, start_date, end_date)
    daily_ohlcv = aggregate_daily_ohlcv(scoped, config.analytics)
    rolling_vwap = compute_rolling_vwap(scoped, config.analytics)
    return AnalysisResult(
        enriched_data=enriched,
        quality_issues=issues,
        gaps=gaps,
        exclusions=analytical.exclusions,
        eligible_data=analytical.data,
        scoped_data=scoped,
        daily_ohlcv=daily_ohlcv,
        rolling_vwap=rolling_vwap,
    )
