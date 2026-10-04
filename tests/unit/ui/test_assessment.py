from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from market_quality.config import AppConfig
from market_quality.models import GapClassification
from market_quality.ui.uploads import process_uploaded_data


def _assessment(result) -> None:
    from market_quality.config import AppConfig
    from market_quality.ui.assessment import render_assessment

    render_assessment(result, AppConfig())


def _application() -> None:
    import app

    app.main()


def test_clean_row_quality_and_completeness_evidence_remain_distinct() -> None:
    contents = (
        b"contract,timestamp,open,high,low,close,volume\n"
        b"ESZ6,2026-01-05 15:59:00,100,101,99,100.5,10\n"
        b"ESZ6,2026-01-05 17:01:00,100,101,99,100.5,20\n"
    )
    _, result = process_uploaded_data(contents, "bars.csv", AppConfig())
    originals = {
        name: getattr(result, name).copy(deep=True)
        for name in ("quality_issues", "gaps", "exclusions", "insights")
    }
    view = AppTest.from_function(_assessment, args=(result,)).run()

    assert not view.exception
    assert [(metric.label, metric.value) for metric in view.metric] == [
        ("Quality issues", "0"),
        ("Unexpected gap intervals", "1"),
        ("Excluded observations", "0"),
    ]
    assert any(
        "No row-level or duplicate quality issues detected" in item.value for item in view.success
    )
    assert any("No observations were excluded" in item.value for item in view.info)
    assert any("1min cadence and America/Chicago" in item.value for item in view.caption)
    assert any("do not prove source records were lost" in item.value for item in view.caption)
    assert result.gaps["classification"].eq(GapClassification.EXPECTED_CLOSURE).any()
    unexpected = result.gaps.loc[result.gaps["classification"].eq(GapClassification.UNEXPECTED_GAP)]
    table = view.dataframe[0].value
    pd.testing.assert_frame_equal(table, unexpected.loc[:, table.columns])
    assert result.insights.loc[0, "recommendation"] in [item.value for item in view.markdown]
    assert result.insights.loc[0, "summary"] in [item.value for item in view.markdown]
    assert "Unexpected gaps — ESZ6" in [item.label for item in view.expander]
    for name, original in originals.items():
        pd.testing.assert_frame_equal(getattr(result, name), original)


@pytest.mark.parametrize("defective", [False, True])
def test_assessment_is_visible_without_eligible_observations(
    defective: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    import app

    contents = b"contract,timestamp,open,high,low,close,volume\n"
    if defective:
        contents += b"ESZ6,2026-01-05 10:31:00,100,99,101,100,-1\n"
    canonical, result = process_uploaded_data(contents, "bars.csv", AppConfig())
    original_issues = result.quality_issues.copy(deep=True)
    upload = BytesIO(contents)
    upload.name = "bars.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    monkeypatch.setattr(app, "process_uploaded_data", Mock(return_value=(canonical, result)))
    app.assess_upload.clear()
    try:
        view = AppTest.from_function(_application).run()
        assert not view.exception
        assert "Data Quality" in [item.value for item in view.subheader]
        assert "Insights" in [item.value for item in view.subheader]
        assert not view.get("plotly_chart")
        if defective:
            assert len(result.exclusions) == 2
            assert [
                (item.label, item.value)
                for item in view.metric
                if item.label == "Excluded observations"
            ] == [("Excluded observations", "1")]
            breakdown, details, exclusions = [item.value for item in view.dataframe]
            assert breakdown["issue_count"].sum() == 2
            expected_details = original_issues.loc[:, details.columns].copy()
            expected_details["actual_value"] = expected_details["actual_value"].map(str)
            pd.testing.assert_frame_equal(details, expected_details, check_dtype=False)
            assert any(isinstance(value, dict) for value in original_issues["actual_value"])
            pd.testing.assert_frame_equal(
                exclusions, result.exclusions.loc[:, exclusions.columns], check_dtype=False
            )
            for recommendation in result.insights["recommendation"]:
                assert recommendation in [item.value for item in view.markdown]
        else:
            assert any(
                "No deterministic quality or completeness insights" in item.value
                for item in view.info
            )
            assert any(
                "No row-level or duplicate quality issues detected" in item.value
                for item in view.success
            )
        pd.testing.assert_frame_equal(result.quality_issues, original_issues)
    finally:
        app.assess_upload.clear()


def test_analytics_filters_preserve_full_assessment_and_reuse_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    import app

    upload = BytesIO(
        b"contract,timestamp,open,high,low,close,volume\n"
        b"ESZ6,2026-01-05 10:31:00,100,101,99,100.5,10\n"
        b"ESZ6,2026-01-05 10:34:00,100,101,99,100.5,20\n"
        b"NQZ6,2026-01-06 10:31:00,200,201,199,200.5,10\n"
        b"NQZ6,2026-01-06 10:33:00,200,199,201,200,-1\n"
        b"NQZ6,2026-01-06 10:34:00,200,201,199,200.5,20\n"
    )
    upload.name = "bars.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    process = Mock(wraps=process_uploaded_data)
    monkeypatch.setattr(app, "process_uploaded_data", process)
    app.assess_upload.clear()
    try:
        view = AppTest.from_function(_application).run()
        assert not view.exception
        evidence = {
            field: next(item.value for item in view.dataframe if field in item.value.columns).copy()
            for field in ("actual_value", "gap_start", "exclusion_reason")
        }
        recommendations = [
            item.value for item in view.markdown if item.value.startswith(("Validate ", "Require "))
        ]
        assert recommendations
        assessment_metrics = [
            (item.label, item.value)
            for item in view.metric
            if item.label in ("Quality issues", "Unexpected gap intervals", "Excluded observations")
        ]

        def insight_contents():
            return [
                (
                    item.label,
                    [text.value for text in item.markdown],
                    [caption.value for caption in item.caption],
                )
                for item in view.expander
                if any(text.value == "**Suggested control**" for text in item.markdown)
            ]

        insights = insight_contents()
        assert insights
        view.multiselect[0].set_value(["ESZ6"]).run()
        assert not view.exception
        for field, expected in evidence.items():
            actual = next(item.value for item in view.dataframe if field in item.value.columns)
            pd.testing.assert_frame_equal(actual, expected)
        view.date_input[0].set_value(date(2026, 1, 6)).run()
        assert not view.exception
        assert not view.get("plotly_chart")

        view.date_input[1].set_value(date(2026, 1, 5)).run()
        assert not view.exception
        assert [item.value for item in view.error] == [
            "Start session date must be on or before end session date."
        ]
        assert {"Data Quality", "Insights"}.issubset(item.value for item in view.subheader)
        assert [
            (item.label, item.value)
            for item in view.metric
            if item.label in ("Quality issues", "Unexpected gap intervals", "Excluded observations")
        ] == assessment_metrics
        for field, expected in evidence.items():
            actual = next(item.value for item in view.dataframe if field in item.value.columns)
            pd.testing.assert_frame_equal(actual, expected)
        assert insight_contents() == insights
        assert process.call_count == 1

        view.date_input[1].set_value(date(2026, 1, 6)).run()
        assert not view.exception
        assert not view.error
        view.multiselect[0].set_value([]).run()
        assert not view.exception
        for field, expected in evidence.items():
            actual = next(item.value for item in view.dataframe if field in item.value.columns)
            pd.testing.assert_frame_equal(actual, expected)
        assert all(text in [item.value for item in view.markdown] for text in recommendations)
        assert process.call_count == 1
    finally:
        app.assess_upload.clear()
