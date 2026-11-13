"""Runs the models over a company record and the full watchlist (v2 scoring)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import config as C
from .models import (altman_z2, coverage, default_point, merton_kmv,
                     net_debt_to_ebitda, rag, status_v2)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def z_applies(rec: dict) -> bool:
    """False for business models Altman Z'' misreads."""
    if rec["ticker"] in C.Z_EXCLUDE_TICKERS:
        return False
    text = f"{rec.get('industry', '')} {rec.get('sector', '')}".lower()
    return not any(k in text for k in C.Z_EXCLUDE_KEYWORDS)


def data_quality(rec: dict, z_on: bool) -> list[str]:
    issues = []
    needed = ["total_assets", "total_liabilities", "ebitda", "interest_expense"]
    if z_on:
        needed += ["working_capital", "retained_earnings", "ebit", "book_equity"]
    missing = [k for k in needed if _nan(rec.get(k))]
    if missing:
        issues.append("missing " + ", ".join(missing))
    if _nan(rec.get("short_term_debt")) and _nan(rec.get("long_term_debt")):
        issues.append("no debt split, used total debt")
    sd = rec.get("statement_date")
    if sd:
        age = (date.today() - date.fromisoformat(sd)).days
        if age > C.STALE_STATEMENT_DAYS:
            issues.append(f"statements {age // 30} months old")
    else:
        issues.append("no statement date")
    if rec.get("statement_currency") not in (None, "AUD"):
        issues.append(f"converted from {rec['statement_currency']}")
    return issues


def score_company(rec: dict) -> dict:
    mcap = rec["market_cap_series"].dropna().iloc[-C.VOL_WINDOW:]
    st, lt = rec["short_term_debt"], rec["long_term_debt"]
    if _nan(st) and _nan(lt) and not _nan(rec["total_debt"]):
        st, lt = 0.0, rec["total_debt"]
    dp = default_point(0 if _nan(st) else st, 0 if _nan(lt) else lt)

    m = merton_kmv(mcap, dp, C.RISK_FREE_RATE, C.HORIZON)

    z_on = z_applies(rec)
    z = altman_z2(rec["working_capital"], rec["retained_earnings"], rec["ebit"],
                  rec["book_equity"], rec["total_assets"], rec["total_liabilities"])
    ebitda_cov = coverage(rec["ebitda"], rec["interest_expense"])
    ebit_cov = coverage(rec["ebit"], rec["interest_expense"])
    lev = net_debt_to_ebitda(rec["total_debt"], rec["cash"], rec["ebitda"])

    # One-off distortion: EBIT negative while EBITDA positive suggests
    # impairments or restructuring charges rather than weak cash earnings.
    ebit_distorted = (not _nan(rec["ebit"]) and not _nan(rec["ebitda"])
                      and rec["ebit"] < 0 < rec["ebitda"])

    px = rec["prices"].iloc[-252:]
    drawdown = 1 - px.iloc[-1] / px.max()
    ret_3m = px.iloc[-1] / px.iloc[-63] - 1 if len(px) > 63 else np.nan

    f_dd = rag(m["distance_to_default"], *C.THRESHOLDS["distance_to_default"])
    f_z = rag(z, *C.THRESHOLDS["altman_z2"]) if z_on else "OFF"
    f_cov = rag(ebitda_cov, *C.THRESHOLDS["ebitda_coverage"])
    f_mkt = "AMBER" if drawdown >= C.DRAWDOWN_ALERT else "GREEN"

    status = status_v2(f_dd, [f_dd, f_z, f_cov], f_mkt, C.RED_SIGNALS_NEEDED)

    reasons = []
    if f_dd in ("RED", "AMBER"):
        reasons.append(f"DD {m['distance_to_default']:.2f} ({f_dd})")
    if f_z in ("RED", "AMBER"):
        reasons.append(f"Z'' {z:.2f} ({f_z})")
    if f_cov in ("RED", "AMBER"):
        reasons.append(f"EBITDA cover {ebitda_cov:.1f}x ({f_cov})")
    if f_mkt == "AMBER":
        reasons.append(f"drawdown {drawdown:.0%}")
    if ebit_distorted:
        reasons.append("EBIT negative, likely one-offs")

    dq = data_quality(rec, z_on)

    return {
        "ticker": rec["ticker"], "name": rec["name"], "sector": rec["sector"],
        "industry": rec.get("industry", ""),
        "status": status,
        "reasons": "; ".join(reasons),
        "data_issues": "; ".join(dq),
        "distance_to_default": m["distance_to_default"],
        "pd_1y": m["pd_1y"],
        "altman_z2": z if z_on else np.nan,
        "z_applies": z_on,
        "ebitda_coverage": ebitda_cov,
        "ebit_coverage": ebit_cov,
        "net_debt_ebitda": lev,
        "drawdown_52w": drawdown,
        "return_3m": ret_3m,
        "market_cap_bn": mcap.iloc[-1] / 1e9,
        "default_point_bn": dp / 1e9,
        "asset_vol": m["asset_vol"],
        "flag_dd": f_dd, "flag_z": f_z, "flag_cov": f_cov, "flag_mkt": f_mkt,
        "red_signals": sum(f == "RED" for f in (f_dd, f_z, f_cov)),
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
        df["_o"] = df["status"].map(order)
        df = df.sort_values(["_o", "distance_to_default"]).drop(columns="_o")
    df.attrs["errors"] = errors
    return df.reset_index(drop=True)
