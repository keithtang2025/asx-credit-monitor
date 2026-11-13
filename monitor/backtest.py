"""Distance-to-default backtest engine.

Pure functions (testable without internet) plus Yahoo/CSV loaders.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .models import default_point, merton_kmv

PRICE_DIR = Path(__file__).resolve().parent.parent / "backtest" / "prices"


# ------------------------------------------------------------ debt as-of ---

def debt_asof(schedule: list[tuple], when: pd.Timestamp, lag_days: int):
    """Default point known at `when`, using a reporting lag.

    schedule: [(balance_date, short_term, long_term), ...]
    Returns (default_point, assumed) where assumed=True means no balance
    sheet was public yet and the earliest one was used (look-ahead risk).
    """
    rows = sorted((pd.Timestamp(d), st, lt) for d, st, lt in schedule)
    known = [r for r in rows if r[0] + pd.Timedelta(days=lag_days) <= when]
    if known:
        _, st, lt = known[-1]
        return default_point(st, lt), False
    _, st, lt = rows[0]
    return default_point(st, lt), True


# ------------------------------------------------------------- DD path ---

def dd_path(mcap: pd.Series, schedule: list[tuple], r: float,
            start: pd.Timestamp, end: pd.Timestamp,
            window: int = 252, step: int = 5, lag_days: int = 60) -> pd.DataFrame:
    """Weekly distance to default between start and end, no look-ahead."""
    s = mcap.dropna().sort_index()
    s = s[s > 0]
    out = []
    idx = s.index
    positions = [i for i in range(window, len(s) + 1, step)
                 if start <= idx[i - 1] <= end]
    for i in positions:
        t = idx[i - 1]
        dp, assumed = debt_asof(schedule, t, lag_days)
        res = merton_kmv(s.iloc[i - window:i], dp, r, 1.0, max_iter=25)
        out.append({"date": t, "dd": res["distance_to_default"],
                    "pd_1y": res["pd_1y"], "asset_vol": res["asset_vol"],
                    "market_cap": s.iloc[i - 1], "default_point": dp,
                    "debt_assumed": assumed})
    return pd.DataFrame(out).set_index("date") if out else pd.DataFrame()


# ------------------------------------------------------------- metrics ---

@dataclass
class CaseResult:
    name: str
    group: str
    event_date: pd.Timestamp
    first_amber: pd.Timestamp | None
    first_red: pd.Timestamp | None
    lead_amber_days: int | None
    lead_red_days: int | None
    min_dd: float
    min_dd_date: pd.Timestamp | None
    last_dd: float
    pct_weeks_amber_or_worse: float
    pct_weeks_red: float
    observations: int
    debt_assumed_share: float


def first_sustained(series: pd.Series, threshold: float, confirm: int):
    """First date the series stays below threshold for `confirm` observations."""
    below = (series < threshold).astype(int)
    run = below.groupby((below != below.shift()).cumsum()).cumsum() * below
    hits = run[run >= confirm]
    if hits.empty:
        return None
    # report the start of the confirmed run
    pos = series.index.get_loc(hits.index[0]) - (confirm - 1)
    return series.index[pos]


def evaluate(name: str, group: str, path: pd.DataFrame, event_date,
             amber: float | None = None, red: float | None = None,
             confirm: int = 2) -> CaseResult:
    amber = amber if amber is not None else C.THRESHOLDS["distance_to_default"][1]
    red = red if red is not None else C.THRESHOLDS["distance_to_default"][0]
    ev = pd.Timestamp(event_date)
    dd = path["dd"].dropna()
    fa = first_sustained(dd, amber, confirm)
    fr = first_sustained(dd, red, confirm)
    lead = lambda d: int((ev - d).days) if d is not None else None
    return CaseResult(
        name=name, group=group, event_date=ev,
        first_amber=fa, first_red=fr,
        lead_amber_days=lead(fa), lead_red_days=lead(fr),
        min_dd=float(dd.min()) if len(dd) else np.nan,
        min_dd_date=dd.idxmin() if len(dd) else None,
        last_dd=float(dd.iloc[-1]) if len(dd) else np.nan,
        pct_weeks_amber_or_worse=float((dd < amber).mean()) if len(dd) else np.nan,
        pct_weeks_red=float((dd < red).mean()) if len(dd) else np.nan,
        observations=len(dd),
        debt_assumed_share=float(path["debt_assumed"].mean()) if len(path) else np.nan,
    )


# ------------------------------------------------------------- loaders ---

def load_prices(ticker: str, start, end) -> tuple[pd.DataFrame, str]:
    """Daily Close (and optional Shares/MarketCap). Yahoo first, then CSV."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(start=start, end=end + pd.Timedelta(days=1),
                                      auto_adjust=False)
        if len(h) > 260:
            h.index = h.index.tz_localize(None)
            return h[["Close"]], "Yahoo"
    except Exception:
        pass
    code = ticker.split(".")[0]
    for p in (PRICE_DIR / f"{code}.csv", PRICE_DIR / f"{ticker}.csv"):
        if p.exists():
            df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
            return df.loc[(df.index >= start) & (df.index <= end)], f"CSV {p.name}"
    raise FileNotFoundError(
        f"No price data for {ticker}. Yahoo has no history for this window "
        f"(common for delisted companies). Add backtest/prices/{code}.csv "
        f"with columns Date,Close[,Shares|MarketCap].")


def load_shares(ticker: str, prices: pd.DataFrame, fixed: float | None):
    """Shares outstanding aligned to the price dates."""
    if "MarketCap" in prices:
        return None
    if "Shares" in prices:
        return prices["Shares"].ffill()
    if fixed:
        return pd.Series(fixed, index=prices.index)
    try:
        import yfinance as yf
        sh = yf.Ticker(ticker).get_shares_full(start=prices.index[0] - pd.Timedelta(days=400),
                                               end=prices.index[-1])
        if sh is not None and len(sh):
            sh.index = sh.index.tz_localize(None)
            sh = sh[~sh.index.duplicated(keep="last")].sort_index()
            return sh.reindex(prices.index, method="ffill").bfill()
    except Exception:
        pass
    raise ValueError(f"No share count for {ticker}: set 'shares' in backtest_cases.py "
                     f"or add a Shares/MarketCap column to the CSV.")


def yahoo_debt_schedule(ticker: str) -> list[tuple]:
    """Annual (date, short-term, long-term) debt from Yahoo balance sheets."""
    import yfinance as yf
    bs = yf.Ticker(ticker).balance_sheet
    if bs is None or bs.empty:
        raise ValueError(f"No Yahoo balance sheets for {ticker}; add a 'debt' schedule.")

    def pick(names, col):
        for n in names:
            if n in bs.index and pd.notna(bs.loc[n, col]):
                return float(bs.loc[n, col])
        return np.nan

    out = []
    for col in bs.columns:
        st = pick(["Current Debt And Capital Lease Obligation", "Current Debt"], col)
        lt = pick(["Long Term Debt And Capital Lease Obligation", "Long Term Debt"], col)
        tot = pick(["Total Debt"], col)
        if np.isnan(st) and np.isnan(lt) and not np.isnan(tot):
            st, lt = 0.0, tot
        if not (np.isnan(st) and np.isnan(lt)):
            out.append((str(col.date()), 0.0 if np.isnan(st) else st,
                        0.0 if np.isnan(lt) else lt))
    if not out:
        raise ValueError(f"Yahoo balance sheets for {ticker} have no debt lines.")
    return out
