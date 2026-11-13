"""Runs the models over a company record and the full watchlist (v9 scoring)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import config as C
from .models import (altman_z2, coverage, default_point, merton_kmv,
                     net_debt_to_ebitda, rag, rag_high, status_v8)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def z_applies(rec: dict) -> bool:
    """False for business models Altman Z'' misreads."""
    if rec["ticker"] in C.Z_EXCLUDE_TICKERS:
        return False
    text = f"{rec.get('industry', '')} {rec.get('sector', '')}".lower()
    return not any(k in text for k in C.Z_EXCLUDE_KEYWORDS)


def gearing_thresholds(rec: dict):
    """(red_above, amber_above) if this business is judged on gearing, else None."""
    if rec["ticker"] in C.GEARING_TICKER_OVERRIDES:
        return C.GEARING_TICKER_OVERRIDES[rec["ticker"]]
    text = f"{rec.get('industry', '')} {rec.get('sector', '')}".lower()
    for keywords, thr in C.GEARING_BY_SECTOR:
        if any(k in text for k in keywords):
            return thr
    return None


def coverage_thresholds(rec: dict) -> tuple[float, float]:
    if rec["ticker"] in C.COVERAGE_TICKER_OVERRIDES:
        return C.COVERAGE_TICKER_OVERRIDES[rec["ticker"]]
    text = f"{rec.get('industry', '')} {rec.get('sector', '')}".lower()
    for keywords, thr in C.COVERAGE_BY_SECTOR:
        if any(k in text for k in keywords):
            return thr
    return C.THRESHOLDS["ebitda_coverage"]


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
    issues.extend(rec.get("crosscheck_issues", []))
    return issues


def score_company(rec: dict) -> dict:
    mcap = rec["market_cap_series"].dropna().iloc[-C.VOL_WINDOW:]
    st, lt = rec["short_term_debt"], rec["long_term_debt"]
    if _nan(st) and _nan(lt) and not _nan(rec["total_debt"]):
        st, lt = 0.0, rec["total_debt"]
    dp = default_point(0 if _nan(st) else st, 0 if _nan(lt) else lt)

    m = merton_kmv(mcap, dp, C.RISK_FREE_RATE, C.HORIZON)

    # Net cash: more cash than debt (v6). v8: Z'' is also switched off for
    # these, since accumulated start-up losses say nothing about default risk.
    # v9: Yahoo omits debt lines for debt-free companies; missing = zero
    debt_parts = [rec.get(k) for k in ("total_debt", "short_term_debt", "long_term_debt")]
    if not _nan(rec.get("total_debt")):
        debt = rec["total_debt"]
    elif any(not _nan(v) for v in debt_parts[1:]):
        debt = sum(0.0 if _nan(v) else v for v in debt_parts[1:])
    else:
        debt = 0.0
    cash = rec.get("cash")
    net_cash = not _nan(cash) and cash >= debt
    z_on = z_applies(rec) and not net_cash
    z = altman_z2(rec["working_capital"], rec["retained_earnings"], rec["ebit"],
                  rec["book_equity"], rec["total_assets"], rec["total_liabilities"])
    cov_latest = coverage(rec["ebitda"], rec["interest_expense"])
    ebitda_3y, int_3y = rec.get("ebitda_3y", np.nan), rec.get("interest_3y", np.nan)
    cov_3y = coverage(ebitda_3y, int_3y) if not _nan(ebitda_3y) else np.nan

    # Cyclicals are judged through the cycle; everyone else on the latest year
    ind = str(rec.get("industry", "")).lower()
    cyclical = (rec.get("sector") in C.CYCLICAL_SECTORS
                or rec["ticker"] in C.CYCLICAL_TICKERS
                or any(k in ind for k in C.CYCLICAL_INDUSTRY_KEYWORDS))
    if cyclical and not _nan(cov_3y):
        ebitda_cov, cov_basis = cov_3y, "3y avg"
    else:
        ebitda_cov, cov_basis = cov_latest, "latest"

    # Earnings trend: latest normalised EBITDA versus its 3-year average
    trend = np.nan
    if not _nan(ebitda_3y) and ebitda_3y > 0 and not _nan(rec["ebitda"]):
        trend = rec["ebitda"] / ebitda_3y
    f_trend = ("N/A" if _nan(trend)
               else "AMBER" if trend < C.EARNINGS_FALL_RATIO else "GREEN")
    ebit_cov = coverage(rec["ebit"], rec["interest_expense"])
    lev = net_debt_to_ebitda(rec["total_debt"], rec["cash"], rec["ebitda"])

    # One-off distortion: large unusual items relative to normalised EBITDA,
    # or reported EBIT negative while normalised EBITDA is positive.
    unusual = rec.get("unusual_items", np.nan)
    ebit_rep = rec.get("ebit_reported", rec["ebit"])
    one_off = False
    if not _nan(unusual) and not _nan(rec["ebitda"]) and rec["ebitda"] != 0:
        one_off = abs(unusual) > C.ONE_OFF_SHARE * abs(rec["ebitda"])
    if not _nan(ebit_rep) and not _nan(rec["ebitda"]) and ebit_rep < 0 < rec["ebitda"]:
        one_off = True

    px = rec["prices"].iloc[-252:]
    drawdown = 1 - px.iloc[-1] / px.max()
    ret_3m = px.iloc[-1] / px.iloc[-63] - 1 if len(px) > 63 else np.nan

    f_dd = rag(m["distance_to_default"], *C.THRESHOLDS["distance_to_default"])
    f_z = rag(z, *C.THRESHOLDS["altman_z2"]) if z_on else "OFF"
    cov_thr = coverage_thresholds(rec)

    # v8: one credit metric per business model
    gear_thr = gearing_thresholds(rec)
    loss_making = (not _nan(rec["ebitda"]) and rec["ebitda"] <= 0
                   and not (cyclical and not _nan(cov_3y) and ebitda_3y > 0))
    # v9: net gearing, as REITs and their lenders measure it:
    # (debt - cash) / (total assets - cash), floored at zero
    cash0 = 0.0 if _nan(cash) else cash
    assets = rec.get("total_assets")
    gearing = (max(debt - cash0, 0.0) / (assets - cash0)
               if not _nan(assets) and assets - cash0 > 0 else np.nan)
    runway = np.nan
    f_fund = "N/A"
    if loss_making:
        burn = -rec["ebitda"] + (abs(rec["interest_expense"])
                                 if not _nan(rec.get("interest_expense")) else 0.0)
        if not _nan(cash) and burn > 0:
            runway = 12 * cash / burn
    if gear_thr:
        credit_metric, credit_value = "gearing", gearing
        f_cov = rag_high(gearing, *gear_thr)
    elif loss_making and net_cash:
        # v9: cash covers all debt, so lenders could be repaid today. A short
        # runway means raising equity (dilution), not default: funding risk,
        # capped at WATCH.
        credit_metric, credit_value, f_cov = "funding runway", runway, "OFF"
        f_fund = rag(runway, *C.RUNWAY_MONTHS)
    elif loss_making:
        credit_metric, credit_value = "cash runway", runway
        f_cov = rag(runway, *C.RUNWAY_MONTHS)
    elif net_cash:
        # tiny interest costs make cover meaningless (v6)
        credit_metric, credit_value, f_cov = "net cash", np.nan, "OFF"
    else:
        credit_metric, credit_value = "interest cover", ebitda_cov
        f_cov = rag(ebitda_cov, *cov_thr)
    f_mkt = "AMBER" if drawdown >= C.DRAWDOWN_ALERT else "GREEN"

    # v4: the shared-cause merge only applies when the ratios could NOT be
    # cleaned of one-offs. If they were cleaned and are still red, the
    # weakness is underlying and the two signals are independent.
    normalised = rec.get("normalised", False)
    shared_cause = one_off and not normalised
    status, n_red = status_v8(f_dd, f_z, f_cov, f_mkt, f_trend,
                              shared_cause=shared_cause,
                              red_needed=C.RED_SIGNALS_NEEDED)

    if status == "GREEN" and f_fund in ("RED", "AMBER"):
        status = "WATCH"

    reasons = []
    if f_dd in ("RED", "AMBER"):
        reasons.append(f"DD {m['distance_to_default']:.2f} ({f_dd})")
    if f_z in ("RED", "AMBER"):
        reasons.append(f"Z'' {z:.2f} ({f_z})")
    if f_cov in ("RED", "AMBER"):
        if credit_metric == "gearing":
            reasons.append(f"gearing {gearing:.0%} ({f_cov})")
        elif credit_metric == "cash runway":
            reasons.append(f"loss-making: cash runway {runway:.0f} months ({f_cov})")
        else:
            reasons.append(f"EBITDA cover {ebitda_cov:.1f}x {cov_basis} ({f_cov})")
    if f_fund in ("RED", "AMBER"):
        reasons.append(f"funding risk (cash covers debt): runway {runway:.0f} months")
    dd_val = m["distance_to_default"]
    if (f_cov == "RED" and not _nan(dd_val) and not np.isinf(dd_val)
            and dd_val >= C.MARKET_DISAGREES_DD):
        reasons.append(f"CHECK DATA: market (DD {dd_val:.1f}) sees less risk than the accounts")
    if f_trend == "AMBER":
        if rec["ebitda"] <= 0:
            reasons.append("earnings falling: EBITDA turned negative (3y avg positive)")
        else:
            reasons.append(f"earnings falling: EBITDA {trend:.0%} of 3y avg")
    if f_mkt == "AMBER":
        reasons.append(f"drawdown {drawdown:.0%}")
    if one_off:
        u = f"{unusual / 1e6:,.0f}m" if not _nan(unusual) else "n/a"
        if normalised:
            reasons.append(f"one-off items {u} (ratios normalised)")
        else:
            reasons.append("one-offs suspected but not adjustable (signals merged)")
    if rec.get("source_check") == "MISMATCH" and n_red > 0:
        reasons.append("CHECK DATA: sources disagree")

    dq = data_quality(rec, z_on)

    return {
        "ticker": rec["ticker"], "name": rec["name"], "sector": rec["sector"],
        "industry": rec.get("industry", ""),
        "status": status,
        "reasons": "; ".join(reasons),
        "data_issues": "; ".join(dq),
        "distance_to_default": m["distance_to_default"],
        "dd_display": ("no debt" if np.isinf(m["distance_to_default"])
                       else f"{m['distance_to_default']:.2f}"),
        "credit_metric": credit_metric,
        "credit_metric_value": credit_value,
        "gearing": gearing,
        "cash_runway_months": runway,
        "flag_funding": f_fund,
        "pd_1y": m["pd_1y"],
        "altman_z2": z if z_on else np.nan,
        "z_applies": z_on,
        "ebitda_coverage": ebitda_cov,
        "coverage_basis": cov_basis,
        "ebitda_cover_latest": cov_latest,
        "ebitda_cover_3y": cov_3y,
        "earnings_vs_3y": trend,
        "flag_trend": f_trend,
        "cyclical": cyclical,
        "net_cash": net_cash,
        "ebitda_m": rec["ebitda"] / 1e6 if not _nan(rec["ebitda"]) else np.nan,
        "ebitda_3y_m": ebitda_3y / 1e6 if not _nan(ebitda_3y) else np.nan,
        "ebitda_source": rec.get("ebitda_source", ""),
        "coverage_red_line": cov_thr[0],
        "ebit_coverage": ebit_cov,
        "one_off_affected": one_off,
        "normalised": normalised,
        "source_check": rec.get("source_check", "N/A"),
        "unusual_items_m": unusual / 1e6 if not _nan(unusual) else np.nan,
        "net_debt_ebitda": lev,
        "drawdown_52w": drawdown,
        "return_3m": ret_3m,
        "market_cap_bn": mcap.iloc[-1] / 1e9,
        "default_point_bn": dp / 1e9,
        "asset_vol": m["asset_vol"],
        "flag_dd": f_dd, "flag_z": f_z, "flag_cov": f_cov, "flag_mkt": f_mkt,
        "red_signals": n_red,
        "statement_date": rec["statement_date"],
        "statement_currency": rec["statement_currency"],
    }


def run_watchlist(tickers=None, fetch=None, verbose=True) -> pd.DataFrame:
    if fetch is None:
        from .data import fetch_company as fetch
    import time
    tickers = tickers or C.WATCHLIST
    rows, errors = [], []
    excluded = []
    live = fetch is None or getattr(fetch, "__module__", "") == "monitor.data"
    for tk in tickers:
        rec = None
        for attempt in (1, 2):                      # one retry for flaky requests
            try:
                rec = fetch(tk)
                break
            except Exception as e:
                if attempt == 2 or "No price history" in str(e):
                    errors.append((tk, str(e)))
                    if verbose:
                        print(f"  skip {tk}: {e}")
                elif live:
                    time.sleep(5)
        if live:
            time.sleep(C.FETCH_PAUSE)
        if rec is None:
            continue
        if rec.get("sector") in C.EXCLUDED_SECTORS:
            excluded.append(tk)
            if verbose:
                print(f"  excl {tk}: {rec.get('sector')}")
            continue
        try:
            rows.append(score_company(rec))
            if verbose:
                print(f"  ok   {tk}")
        except Exception as e:  # keep going on bad data
            errors.append((tk, str(e)))
            if verbose:
                print(f"  skip {tk}: {e}")
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {"RED": 0, "AMBER": 1, "WATCH": 2, "GREEN": 3, "N/A": 4}
        df["_o"] = df["status"].map(order)
        df = df.sort_values(["_o", "distance_to_default"]).drop(columns="_o")
    df.attrs["errors"] = errors
    df.attrs["excluded"] = excluded
    return df.reset_index(drop=True)
