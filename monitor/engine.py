"""Runs the models over a company record and the full watchlist."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .models import (altman_z2, default_point, interest_coverage, merton_kmv,
                     net_debt_to_ebitda, overall_status, rag)


def score_company(rec: dict) -> dict:
    mcap = rec["market_cap_series"].dropna().iloc[-C.VOL_WINDOW:]
    st, lt = rec["short_term_debt"], rec["long_term_debt"]
    if np.isnan(st) and np.isnan(lt) and not np.isnan(rec["total_debt"]):
        st, lt = 0.0, rec["total_debt"]  # no split available
    dp = default_point(0 if np.isnan(st) else st, 0 if np.isnan(lt) else lt)

    m = merton_kmv(mcap, dp, C.RISK_FREE_RATE, C.HORIZON)
    z = altman_z2(rec["working_capital"], rec["retained_earnings"], rec["ebit"],
                  rec["book_equity"], rec["total_assets"], rec["total_liabilities"])
    icr = interest_coverage(rec["ebit"], rec["interest_expense"])
    lev = net_debt_to_ebitda(rec["total_debt"], rec["cash"], rec["ebitda"])

    px = rec["prices"].iloc[-252:]
    drawdown = 1 - px.iloc[-1] / px.max()
    ret_3m = px.iloc[-1] / px.iloc[-63] - 1 if len(px) > 63 else np.nan

    f_dd = rag(m["distance_to_default"], *C.THRESHOLDS["distance_to_default"])
    f_z = rag(z, *C.THRESHOLDS["altman_z2"])
    f_icr = rag(icr, *C.THRESHOLDS["interest_coverage"])
    f_dd_mkt = "AMBER" if drawdown >= C.DRAWDOWN_ALERT else "GREEN"

    reasons = []
    if f_dd != "GREEN" and f_dd != "N/A":
        reasons.append(f"DD {m['distance_to_default']:.2f}")
    if f_z in ("RED", "AMBER"):
        reasons.append(f"Z'' {z:.2f}")
    if f_icr in ("RED", "AMBER"):
        reasons.append(f"ICR {icr:.1f}x")
    if f_dd_mkt == "AMBER":
        reasons.append(f"drawdown {drawdown:.0%}")

    return {
        "ticker": rec["ticker"], "name": rec["name"], "sector": rec["sector"],
        "market_cap_bn": mcap.iloc[-1] / 1e9,
        "default_point_bn": dp / 1e9,
        "asset_vol": m["asset_vol"],
        "distance_to_default": m["distance_to_default"],
        "pd_1y": m["pd_1y"],
        "altman_z2": z,
        "interest_coverage": icr,
        "net_debt_ebitda": lev,
        "drawdown_52w": drawdown,
        "return_3m": ret_3m,
        "flag_dd": f_dd, "flag_z": f_z, "flag_icr": f_icr, "flag_mkt": f_dd_mkt,
        "status": overall_status([f_dd, f_z, f_icr, f_dd_mkt]),
        "reasons": "; ".join(reasons),
        "statement_date": rec["statement_date"],
        "statement_currency": rec["statement_currency"],
    }


def run_watchlist(tickers=None, fetch=None, verbose=True) -> pd.DataFrame:
    if fetch is None:
        from .data import fetch_company as fetch
    tickers = tickers or C.WATCHLIST
    rows, errors = [], []
    for tk in tickers:
        try:
            rows.append(score_company(fetch(tk)))
            if verbose:
                print(f"  ok   {tk}")
        except Exception as e:  # keep going on bad tickers
            errors.append((tk, str(e)))
            if verbose:
                print(f"  skip {tk}: {e}")
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {"RED": 0, "AMBER": 1, "GREEN": 2, "N/A": 3}
        df = df.sort_values(["status", "distance_to_default"],
                            key=lambda s: s.map(order) if s.name == "status" else s)
    df.attrs["errors"] = errors
    return df.reset_index(drop=True)
