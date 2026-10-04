from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from market_quality.config import AppConfig
from market_quality.ui.analytics import prepare_intraday_plot_data
from market_quality.ui.uploads import process_uploaded_data


def _application() -> None:
    import app

    app.main()


def _analytics(scoped: "pd.DataFrame", daily: "pd.DataFrame", rolling: "pd.DataFrame") -> None:
    from market_quality.config import AppConfig
    from market_quality.ui.analytics import render_analytics

    render_analytics(scoped, daily, rolling, AppConfig())


def test_chart_selections_use_scoped_results_without_reassessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    import app

    contents = (
        b"contract,timestamp,open,high,low,close,volume\n"
        b"ESZ6,2026-01-05 10:31:00,100,101,99,100.5,10\n"
        b"ESZ6,2026-01-06 10:31:00,102,103,101,102.5,20\n"
        b"NQZ6,2026-01-05 10:31:00,200,201,199,200.5,30\n"
        b"NQZ6,2026-01-06 10:31:00,202,203,201,202.5,40\n"
    )
    upload = BytesIO(contents)
    upload.name = "bars.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    process = Mock(wraps=process_uploaded_data)
    monkeypatch.setattr(app, "process_uploaded_data", process)
    app.assess_upload.clear()
    try:
        view = AppTest.from_function(_application).run()
        assert not view.exception
        assert process.call_count == 1
        assert view.selectbox[0].value == "ESZ6"
        assert view.selectbox[1].value == date(2026, 1, 6)
        assert view.dataframe[0].value["contract"].tolist() == ["ESZ6", "ESZ6"]
        assert view.dataframe[1].value["close"].tolist() == [102.5]
        assert len(view.get("plotly_chart")) == 2

        view.selectbox[0].select("NQZ6").run()
        assert not view.exception
        assert view.dataframe[0].value["contract"].tolist() == ["NQZ6", "NQZ6"]
        assert view.dataframe[1].value["close"].tolist() == [202.5]
        assert view.multiselect[0].value == ["ESZ6", "NQZ6"]
        assert process.call_count == 1

        view.selectbox[1].select(date(2026, 1, 5)).run()
        assert not view.exception
        assert view.dataframe[1].value["session_date"].tolist() == [pd.Timestamp("2026-01-05")]
        assert view.dataframe[1].value["close"].tolist() == [200.5]
        assert process.call_count == 1

        view.multiselect[0].set_value(["ESZ6"]).run()
        assert not view.exception
        assert [selector.label for selector in view.selectbox] == ["Trading session"]
        assert view.dataframe[0].value["contract"].tolist() == ["ESZ6", "ESZ6"]

        view.multiselect[0].set_value([]).run()
        assert not view.exception
        assert not view.get("plotly_chart")
        assert "View daily OHLCV data" not in [expander.label for expander in view.expander]
        assert "Data Quality" in [section.value for section in view.subheader]
        assert any("No eligible observations match" in message.value for message in view.info)
        assert process.call_count == 1
    finally:
        app.assess_upload.clear()


@pytest.mark.parametrize("missing_dates", [False, True])
def test_unavailable_analytics_show_informational_states(missing_dates: bool) -> None:
    contents = (
        b"contract,timestamp,open,high,low,close,volume\n"
        b"ESZ6,2026-01-05 10:31:00,100,101,99,100.5,10\n"
    )
    _, result = process_uploaded_data(contents, "bars.csv", AppConfig())
    rolling = result.rolling_vwap.copy() if missing_dates else result.rolling_vwap.iloc[:0].copy()
    if missing_dates:
        rolling["session_date"] = pd.NaT
    original = rolling.copy(deep=True)
    view = AppTest.from_function(
        _analytics, args=(result.scoped_data, result.daily_ohlcv.iloc[:0], rolling)
    ).run()

    assert not view.exception
    assert not view.get("plotly_chart")
    assert not view.selectbox
    assert any("No daily bars" in message.value for message in view.info)
    expected = "No trading session dates" if missing_dates else "No VWAP observations"
    assert any(expected in message.value for message in view.info)
    pd.testing.assert_frame_equal(rolling, original)


def test_intraday_plot_breaks_only_nonconsecutive_observations() -> None:
    timestamps = pd.to_datetime(
        ["2026-01-05T16:31:00Z", "2026-01-05T16:32:00Z", "2026-01-05T16:34:00Z"]
    )
    observations = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "local_timestamp": timestamps.tz_convert("America/Chicago"),
            "close": [100.0, 101.0, 104.0],
            "vwap": [100.0, 100.5, 102.0],
            "volume": [10, 20, 30],
        },
        index=[7, 2, 9],
    )
    original = observations.copy(deep=True)

    plotting = prepare_intraday_plot_data(observations, AppConfig().quality.expected_frequency)

    assert len(plotting) == 4
    assert plotting["timestamp_utc"].iloc[:2].tolist() == timestamps[:2].tolist()
    assert plotting.iloc[2].isna().all()
    assert plotting.loc[3, "timestamp_utc"] == timestamps[2]
    pd.testing.assert_frame_equal(
        plotting.iloc[[0, 1, 3]].reset_index(drop=True),
        original.loc[:, ["timestamp_utc", "local_timestamp", "close", "vwap"]].reset_index(
            drop=True
        ),
    )
    pd.testing.assert_frame_equal(observations, original)
