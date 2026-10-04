import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from market_quality.config import AppConfig


def prepare_intraday_plot_data(
    observations: pd.DataFrame,
    expected_frequency: str,
) -> pd.DataFrame:
    """Insert chart-only separators using elapsed UTC time and upstream-validated cadence.

    The selected-session observations are already ordered by the VWAP backend.
    """
    columns = ["timestamp_utc", "close", "vwap"]
    if "local_timestamp" in observations:
        columns.insert(1, "local_timestamp")
    plotting = observations.loc[:, columns].reset_index(drop=True).copy()
    cadence = pd.Timedelta(pd.tseries.frequencies.to_offset(expected_frequency).nanos, unit="ns")
    breaks = plotting["timestamp_utc"].diff().gt(cadence)
    # Skipped plotting positions become null separators, not additional market bars.
    plotting.index = plotting.index + breaks.cumsum()
    return plotting.reindex(range(len(plotting) + int(breaks.sum())))


@st.fragment
def render_analytics(
    scoped_data: pd.DataFrame,
    daily_ohlcv: pd.DataFrame,
    rolling_vwap: pd.DataFrame,
    config: AppConfig,
) -> None:
    """Chart selectors rerun presentation only, using the already-computed analytics."""
    st.subheader("Analytics")
    contracts = sorted(scoped_data["contract"].dropna().unique().tolist())
    if not contracts:
        st.info("No contracts are available in the current analytics scope.")
        return
    contract = contracts[0] if len(contracts) == 1 else st.selectbox("Chart contract", contracts)
    daily = daily_ohlcv.loc[daily_ohlcv["contract"].eq(contract)]
    intraday = rolling_vwap.loc[rolling_vwap["contract"].eq(contract)]
    daily_tab, intraday_tab = st.tabs(["Daily market view", "Intraday VWAP"])

    with daily_tab:
        if daily.empty:
            st.info("No daily bars are available for this contract in the current scope.")
        else:
            figure = make_subplots(
                rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.06
            )
            figure.add_trace(
                go.Candlestick(
                    x=daily["session_date"],
                    open=daily["open"],
                    high=daily["high"],
                    low=daily["low"],
                    close=daily["close"],
                    name="Session OHLC",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Bar(x=daily["session_date"], y=daily["volume"], name="Volume"), row=2, col=1
            )
            figure.update_layout(
                title=f"{contract} — Daily session OHLC and volume",
                height=600,
                showlegend=False,
                xaxis_rangeslider_visible=False,
            )
            figure.update_yaxes(title_text="Price", row=1, col=1)
            figure.update_yaxes(title_text="Volume", row=2, col=1)
            figure.update_xaxes(title_text="Trading session closing date", row=2, col=1)
            st.plotly_chart(figure)
            st.caption(
                "Bar count describes observed coverage, not guaranteed session completeness."
            )
            with st.expander("View daily OHLCV data"):
                st.dataframe(daily, hide_index=True)

    with intraday_tab:
        if intraday.empty:
            st.info("No VWAP observations are available for this contract in the current scope.")
            return
        sessions = sorted(intraday["session_date"].dropna().dt.date.unique(), reverse=True)
        if not sessions:
            st.info("No trading session dates are available for this contract.")
            return
        session = st.selectbox("Trading session", sessions)
        observations = intraday.loc[intraday["session_date"].eq(pd.Timestamp(session))]
        if observations.empty:
            st.info("No VWAP observations are available for this trading session.")
            return

        time_column = "local_timestamp" if "local_timestamp" in observations else "timestamp_utc"
        timezone = config.session.timezone if time_column == "local_timestamp" else "UTC"
        plotting = prepare_intraday_plot_data(observations, config.quality.expected_frequency)
        # Supply wall clocks to Plotly so the browser cannot shift the displayed market time.
        clock = plotting[time_column].dt.tz_localize(None)
        window = (
            "15-minute" if config.analytics.vwap_window == "15min" else config.analytics.vwap_window
        )
        figure = go.Figure()
        for column, name in (("close", "Close"), ("vwap", f"Rolling {window} VWAP")):
            figure.add_trace(
                go.Scatter(
                    x=clock,
                    y=plotting[column],
                    mode="lines+markers",
                    name=name,
                    connectgaps=False,
                )
            )
        figure.update_layout(
            title=f"{contract} — Close and rolling VWAP — {session}",
            xaxis_title=f"Market time ({timezone})",
            yaxis_title="Price",
        )
        st.plotly_chart(figure)
        st.caption(
            f"Rolling VWAP is a {window} bar-based approximation using Typical Price "
            "(High + Low + Close) / 3 and observed volume. True trade-level VWAP cannot "
            "be recovered from OHLCV bars. Only observed bars are plotted."
        )
        with st.expander("View intraday VWAP data"):
            columns = [
                "contract",
                "session_date",
                time_column,
                "close",
                "volume",
                "representative_price",
                "rolling_volume",
                "vwap",
            ]
            st.dataframe(observations.loc[:, columns], hide_index=True)
