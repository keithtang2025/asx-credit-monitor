"""Data access via yfinance (free, delayed, good enough for an MVP).

Swap this module for Bloomberg, Refinitiv or Morningstar later without
touching the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

_FX_CACHE: dict[str, float] = {}


def _first(df: pd.DataFrame, names: list[str]):
    """Latest value of the first matching row label in a yfinance statement."""
    if df is None or df.empty:
        return np.nan
    for n in names:
        if n in df.index:
            row = df.loc[n].dropna()
            if len(row):
                return float(row.iloc[0])  # columns are newest first
    return np.nan


def _rows(df: pd.DataFrame, names: list[str], n: int = 4) -> list[float]:
    """Up to n most recent annual values (newest first) of the first matching row.

    Keeps NaN gaps so the years stay aligned across different line items.
    """
    if df is None or df.empty:
        return [np.nan] * n
    for name in names:
        if name in df.index:
            vals = pd.to_numeric(df.loc[name].iloc[:n], errors="coerce").astype(float).tolist()
            return vals + [np.nan] * (n - len(vals))
    return [np.nan] * n


def fx_rate(from_ccy: str, to_ccy: str = "AUD") -> float:
    """Multiply an amount in from_ccy by this to get to_ccy."""
    if not from_ccy or from_ccy == to_ccy:
        return 1.0
    key = f"{from_ccy}{to_ccy}"
    if key not in _FX_CACHE:
        hist = yf.Ticker(f"{key}=X").history(period="5d")
        _FX_CACHE[key] = float(hist["Close"].iloc[-1]) if len(hist) else np.nan
    return _FX_CACHE[key]


def fetch_company(ticker: str, period: str = "2y") -> dict:
    from .config import TICKER_ALIASES
    requested = ticker
    ticker = TICKER_ALIASES.get(ticker, ticker)
    t = yf.Ticker(ticker)
    prices = t.history(period=period, auto_adjust=False)["Close"]
    if prices.empty:
        raise ValueError(f"No price history for {ticker}; it may be delisted or "
                         f"renamed (add the new code to TICKER_ALIASES)")
    info = t.info or {}
    prices.index = prices.index.tz_localize(None)

    bs, inc, cf = t.balance_sheet, t.income_stmt, t.cashflow
    fin_ccy = info.get("financialCurrency") or "AUD"
    px_ccy = info.get("currency") or "AUD"
    fx = fx_rate(fin_ccy, px_ccy)  # convert statements into price currency

    g = lambda df, names: _first(df, names) * fx
    shares = info.get("sharesOutstanding") or g(bs, ["Ordinary Shares Number", "Share Issued"]) / fx

    rec = {
        "ticker": ticker,
        "requested_ticker": requested,
        "name": info.get("shortName", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", ""),
        "currency": px_ccy,
        "statement_currency": fin_ccy,
        "prices": prices,
        "shares": shares,
        "market_cap_series": prices * shares,
        "short_term_debt": g(bs, ["Current Debt And Capital Lease Obligation", "Current Debt",
                                  "Current Capital Lease Obligation"]),
        "long_term_debt": g(bs, ["Long Term Debt And Capital Lease Obligation", "Long Term Debt"]),
        "total_debt": g(bs, ["Total Debt"]),
        "cash": g(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
        "total_assets": g(bs, ["Total Assets"]),
        "total_liabilities": g(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"]),
        "working_capital": g(bs, ["Working Capital"]),
        "retained_earnings": g(bs, ["Retained Earnings"]),
        "book_equity": g(bs, ["Stockholders Equity", "Common Stock Equity"]),
        "ebit_reported": g(inc, ["EBIT"]),
        "operating_income": g(inc, ["Operating Income", "Total Operating Income As Reported"]),
        "ebitda_reported": g(inc, ["EBITDA"]),
        "ebitda_normalized_yahoo": g(inc, ["Normalized EBITDA"]),
        "unusual_items": g(inc, ["Total Unusual Items", "Total Unusual Items Excluding Goodwill"]),
        "interest_expense": g(inc, ["Interest Expense", "Interest Expense Non Operating"]),
        "statement_date": str(bs.columns[0].date()) if bs is not None and not bs.empty else None,
    }

    # Multi-year history (newest first, statement currency converted)
    h = lambda df, names: [v * fx for v in _rows(df, names)]
    da = h(inc, ["Reconciled Depreciation"])
    if all(np.isnan(da)):
        da = h(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion"])
    rec["history"] = {
        "ebit": h(inc, ["EBIT"]),
        "opinc": h(inc, ["Operating Income", "Total Operating Income As Reported"]),
        "da": da,
        "unusual": h(inc, ["Total Unusual Items", "Total Unusual Items Excluding Goodwill"]),
        "interest": h(inc, ["Interest Expense", "Interest Expense Non Operating"]),
        "ebitda_yahoo": h(inc, ["EBITDA"]),
        "ebitda_norm_yahoo": h(inc, ["Normalized EBITDA"]),
    }

    # Second source: cross-check and fill gaps BEFORE normalising
    from .config import USE_SECOND_SOURCE
    from .crosscheck import compare_and_fill, fetch_fmp
    fmp = fetch_fmp(ticker) if USE_SECOND_SOURCE else None
    fmp_fx = fx_rate(fmp.get("currency"), px_ccy) if fmp and fmp.get("currency") else fx
    compare_and_fill(rec, fmp, fmp_fx)

    return normalise(rec)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _gap(a, b, tol) -> bool:
    """True when a and b differ by more than tol (relative) and 5m (absolute)."""
    if _nan(a) or _nan(b):
        return False
    d = abs(a - b)
    return d > 5e6 and d > tol * max(abs(a), abs(b))


def _year_ebitda(opinc, ebit, da, unusual, yahoo_norm, yahoo_rep, tol=0.10):
    """Normalised EBITDA for one year, where it came from, and whether
    one-offs sit below the operating line.

    Base is OPERATING INCOME, not Yahoo's "EBIT". Yahoo's EBIT is pre-tax
    profit plus interest, so it includes non-operating items such as large
    impairments booked below the operating line (Treasury Wine FY26: EBIT
    -673m vs operating profit +57.5m).

    Unusual items are added back only when they sit inside operating income
    (EBIT and operating income agree). When EBIT differs, the one-offs are
    below the line and operating income already excludes them, so adding
    them back would double count.
    Yahoo reports unusual items with their P&L sign (losses negative).
    """
    base = opinc if not _nan(opinc) else ebit
    below_line = _gap(ebit, opinc, tol)
    adj = 0.0 if (below_line or _nan(unusual)) else unusual
    if not _nan(base) and not _nan(da):
        src = "operating income + D&A" if not _nan(opinc) else "EBIT + D&A"
        return base + abs(da) - adj, src, below_line, adj
    if not _nan(yahoo_norm):
        return yahoo_norm, "Yahoo normalized EBITDA", below_line, adj
    if not _nan(yahoo_rep):
        return yahoo_rep - adj, "Yahoo EBITDA", below_line, adj
    return np.nan, "missing", below_line, adj


def normalise(rec: dict) -> dict:
    """Latest and 3-year normalised EBIT/EBITDA from operating income."""
    from .config import EBIT_GAP_TOLERANCE, MIN_YEARS_FOR_AVERAGE

    unusual = rec.get("unusual_items", np.nan)
    ebit_rep = rec.get("ebit_reported", np.nan)
    opinc = rec.get("operating_income", np.nan)
    hist = rec.get("history")
    issues = rec.setdefault("crosscheck_issues", [])

    if hist:
        n = len(hist["ebit"])
        hist.setdefault("opinc", [np.nan] * n)
        if not _nan(ebit_rep):
            hist["ebit"][0] = ebit_rep
        if not _nan(opinc):
            hist["opinc"][0] = opinc
        years = [_year_ebitda(hist["opinc"][i], hist["ebit"][i], hist["da"][i],
                              hist["unusual"][i], hist["ebitda_norm_yahoo"][i],
                              hist["ebitda_yahoo"][i], EBIT_GAP_TOLERANCE)
                 for i in range(n)]
        ebitda_years = [y[0] for y in years]
        rec["ebitda"], rec["ebitda_source"], below_line, adj = years[0]
        if _nan(rec["ebitda"]) and not _nan(rec.get("ebitda_reported", np.nan)):
            rec["ebitda"], rec["ebitda_source"] = rec["ebitda_reported"], "Yahoo EBITDA"

        last3 = [v for v in ebitda_years[:3] if not _nan(v)]
        ints = [abs(v) for v in hist["interest"][:3] if not _nan(v)]
        enough = len(last3) >= MIN_YEARS_FOR_AVERAGE and len(ints) >= MIN_YEARS_FOR_AVERAGE
        rec["ebitda_3y"] = float(np.mean(last3)) if enough else np.nan
        rec["interest_3y"] = float(np.mean(ints)) if enough else np.nan
        rec["ebitda_history"] = ebitda_years
        op0, ebit0 = hist["opinc"][0], hist["ebit"][0]
    else:
        op0, ebit0 = opinc, ebit_rep
        below_line = _gap(ebit0, op0, EBIT_GAP_TOLERANCE)
        adj = 0.0 if (below_line or _nan(unusual)) else unusual
        ebitda_norm = rec.get("ebitda_normalized_yahoo", np.nan)
        if _nan(ebitda_norm):
            ebitda_rep = rec.get("ebitda_reported", np.nan)
            ebitda_norm = ebitda_rep - adj if not _nan(ebitda_rep) else np.nan
        rec["ebitda"], rec["ebitda_source"] = ebitda_norm, "Yahoo EBITDA"
        rec["ebitda_3y"] = rec["interest_3y"] = np.nan

    if below_line:
        issues.append(f"Yahoo EBIT {ebit0 / 1e6:,.0f}m includes "
                      f"{(ebit0 - op0) / 1e6:,.0f}m of non-operating items; "
                      f"used operating income {op0 / 1e6:,.0f}m")

    base = op0 if not _nan(op0) else ebit0
    rec["ebit"] = base - adj if not _nan(base) else np.nan
    # Ratios count as one-off adjusted when unusual items were added back, or
    # when operating income already excludes one-offs sitting below the line.
    rec["normalised"] = bool(below_line or (not _nan(unusual) and adj != 0.0))
    return rec
