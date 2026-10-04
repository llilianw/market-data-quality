from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


def _application() -> None:
    import app

    app.main()


def test_supplied_multi_contract_upload_preserves_assessment_across_ui_selections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = Path(__file__).parents[2]
    monkeypatch.syspath_prepend(str(project))
    import app

    fixture = project / "src/market_quality/data/demo/multi_contract_sample.parquet"
    upload = BytesIO(fixture.read_bytes())
    upload.name = fixture.name
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    process = Mock(wraps=app.process_uploaded_data)
    monkeypatch.setattr(app, "process_uploaded_data", process)
    render_assessment = Mock(wraps=app.render_assessment)
    monkeypatch.setattr(app, "render_assessment", render_assessment)
    analytics = {}
    for function, field in (
        ("filter_analysis_scope", "scoped_data"),
        ("aggregate_daily_ohlcv", "daily_ohlcv"),
        ("compute_rolling_vwap", "rolling_vwap"),
    ):
        original = getattr(app, function)

        def capture(*args, _original=original, _field=field, **kwargs):
            result = _original(*args, **kwargs)
            analytics[_field] = result
            return result

        monkeypatch.setattr(app, function, capture)

    app.assess_upload.clear()
    try:
        view = AppTest.from_function(_application, default_timeout=30).run()
        assert not view.exception
        assessment = render_assessment.call_args.args[0]
        evidence = {
            field: getattr(assessment, field).copy(deep=True)
            for field in (
                "enriched_data",
                "quality_issues",
                "gaps",
                "insights",
                "exclusions",
                "eligible_data",
            )
        }
        assert len(assessment.enriched_data) == 3389
        assert len(assessment.eligible_data) == 3389
        assert not assessment.gaps.empty
        assert not assessment.insights.empty

        def assert_scope(contracts, rows, bars, session=None):
            assert not view.exception
            assert not view.error
            assert process.call_count == 1
            for field, count in (
                ("scoped_data", rows),
                ("daily_ohlcv", bars),
                ("rolling_vwap", rows),
            ):
                data = analytics[field]
                assert set(data["contract"]) == set(contracts)
                assert len(data) == count
                if session is not None:
                    assert set(data["session_date"]) == {pd.Timestamp(session)}
            current = render_assessment.call_args.args[0]
            for field, expected in evidence.items():
                pd.testing.assert_frame_equal(getattr(current, field), expected)
            assert {"Data Quality", "Insights"}.issubset(item.value for item in view.subheader)

        def assert_chart_contract(contract):
            assert len(view.get("plotly_chart")) == 2
            daily = next(item.value for item in view.dataframe if "bar_count" in item.value)
            intraday = next(item.value for item in view.dataframe if "rolling_volume" in item.value)
            expected_daily = analytics["daily_ohlcv"].loc[
                analytics["daily_ohlcv"]["contract"].eq(contract)
            ]
            pd.testing.assert_frame_equal(daily, expected_daily, check_dtype=False)
            session = next(item.value for item in view.selectbox if item.label == "Trading session")
            rolling = analytics["rolling_vwap"]
            expected_intraday = rolling.loc[
                rolling["contract"].eq(contract)
                & rolling["session_date"].eq(pd.Timestamp(session)),
                intraday.columns,
            ]
            pd.testing.assert_frame_equal(intraday, expected_intraday, check_dtype=False)

        assert_scope(["ESZ25", "ESH26"], 3389, 4)
        chart = next(item for item in view.selectbox if item.label == "Chart contract")
        chart.select("ESZ25").run()
        assert_scope(["ESZ25", "ESH26"], 3389, 4)
        assert_chart_contract("ESZ25")
        next(item for item in view.selectbox if item.label == "Chart contract").select(
            "ESH26"
        ).run()
        assert_scope(["ESZ25", "ESH26"], 3389, 4)
        assert_chart_contract("ESH26")

        for contracts, rows, bars in (
            (["ESZ25"], 2760, 2),
            (["ESH26"], 629, 2),
            ([], 0, 0),
        ):
            view.multiselect[0].set_value(contracts).run()
            assert_scope(contracts, rows, bars)
        assert not view.get("plotly_chart")

        session = date(2025, 10, 16)
        view.date_input[0].set_value(session)
        view.date_input[1].set_value(session)
        for contract, rows in (("ESZ25", 1380), ("ESH26", 264)):
            view.multiselect[0].set_value([contract]).run()
            assert_scope([contract], rows, 1, session)
            assert_chart_contract(contract)
    finally:
        app.assess_upload.clear()
