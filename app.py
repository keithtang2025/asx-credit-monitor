"""Streamlit dashboard.  Run with:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from monitor import config as C
from monitor.data import fetch_company
from monitor.engine import run_watchlist
from monitor.models import default_point, rolling_dd

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="ASX Credit Risk Monitor", layout="wide")
st.title("ASX Credit Risk Monitor")
st.caption("Merton/KMV distance to default, sector-aware Altman Z'' and EBITDA "
           "interest cover (normalised for one-offs) for non-financial ASX companies. RED needs DD below 2 or two "
           "independent signals red at once; WATCH = one soft signal. Research tool, not a credit rating.")


@st.cache_data(ttl=6 * 3600, show_spinner="Scoring watchlist...")
def load(tickers):
    return run_watchlist(list(tickers), verbose=False)


with st.sidebar:
    st.header("Settings")
    use_saved = st.checkbox("Use last saved run (data/latest.csv)",
                            value=(DATA / "latest.csv").exists())
    C.RISK_FREE_RATE = st.number_input("Risk-free rate", 0.0, 0.15, C.RISK_FREE_RATE, 0.0025,
                                       format="%.4f")
    extra = st.text_input("Extra tickers (comma separated)", "")

tickers = tuple(C.WATCHLIST + [t.strip().upper() for t in extra.split(",") if t.strip()])
df = pd.read_csv(DATA / "latest.csv") if use_saved else load(tickers)

# --- KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Companies", len(df))
c2.metric("RED", int((df.status == "RED").sum()))
c3.metric("AMBER", int((df.status == "AMBER").sum()))
c4.metric("WATCH", int((df.status == "WATCH").sum()))
c5.metric("Median DD", f"{df.distance_to_default.median():.2f}")

# --- Filters and table
sectors = st.multiselect("Sector", sorted(df.sector.dropna().unique()))
status = st.multiselect("Status", ["RED", "AMBER", "WATCH", "GREEN", "N/A"],
                        default=["RED", "AMBER"])
view = df.copy()
if sectors:
    view = view[view.sector.isin(sectors)]
if status:
    view = view[view.status.isin(status)]

COLORS = {"RED": "background-color:#f8d7da", "AMBER": "background-color:#fff3cd",
          "WATCH": "background-color:#e2e3f3",
          "GREEN": "background-color:#d1e7dd"}
cols = ["ticker", "name", "sector", "status", "reasons", "data_issues",
        "distance_to_default", "pd_1y", "altman_z2", "ebitda_coverage", "ebit_coverage",
        "net_debt_ebitda", "drawdown_52w", "market_cap_bn", "default_point_bn",
        "statement_date"]
st.dataframe(
    view[cols].style.map(lambda v: COLORS.get(v, ""), subset=["status"]).format({
        "distance_to_default": "{:.2f}", "pd_1y": "{:.2%}", "altman_z2": "{:.2f}",
        "ebitda_coverage": "{:.1f}x", "ebit_coverage": "{:.1f}x", "net_debt_ebitda": "{:.1f}x",
        "drawdown_52w": "{:.0%}", "market_cap_bn": "{:.1f}", "default_point_bn": "{:.1f}"},
        na_rep="n/a"),
    use_container_width=True, hide_index=True)

# --- Cross-section chart
fig = px.scatter(df.dropna(subset=["altman_z2"]), x="altman_z2", y="distance_to_default", color="status",
                 size="market_cap_bn", hover_name="ticker",
                 color_discrete_map={"RED": "#c0392b", "AMBER": "#e6a700", "WATCH": "#6c71c4",
                                     "GREEN": "#2e8b57", "N/A": "#999"},
                 labels={"altman_z2": "Altman Z''", "distance_to_default": "Distance to default"})
fig.add_hline(y=C.THRESHOLDS["distance_to_default"][0], line_dash="dot")
fig.add_vline(x=C.THRESHOLDS["altman_z2"][0], line_dash="dot")
st.plotly_chart(fig, use_container_width=True)
st.caption("Companies where Altman Z'' is switched off (retail, airlines, travel, "
           "infrastructure, utilities, property) are left out of this chart.")

# --- Drill-down
st.subheader("Company drill-down")
tk = st.selectbox("Ticker", df.ticker)
if st.button("Compute distance-to-default history"):
    rec = fetch_company(tk, period="5y")
    dp = default_point(rec["short_term_debt"] or 0, rec["long_term_debt"] or 0)
    hist = rolling_dd(rec["market_cap_series"], dp, C.RISK_FREE_RATE, C.HORIZON)
    st.line_chart(hist)
    st.caption("Default point held at the latest balance sheet, so this isolates "
               "market-driven changes in credit risk.")

# --- History of statuses
if (DATA / "history.csv").exists():
    st.subheader("Run history")
    h = pd.read_csv(DATA / "history.csv")
    h = h[h.ticker == tk]
    if len(h) > 1:
        st.line_chart(h.set_index("run_date")["distance_to_default"])
