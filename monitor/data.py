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
        "ebit_reported": g(inc, ["EBIT", "Operating Income"]),
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
        "ebit": h(inc, ["EBIT", "Operating Income"]),
        "da": da,
        "unusual": h(inc, ["Total Unusual Items", "Total Unusual Items Excluding Goodwill"]),
        "interest": h(inc, ["Interest Expense", "Interest Expense Non Operating"]),
        "ebitda_yahoo": h(inc, ["EBITDA"]),
        "ebitda_norm_yahoo": h(inc, ["Normalized EBITDA"]),
    }

    # Second source: cross-check and fill gaps BEFORE normalising
    from .crosscheck import compare_and_fill, fetch_fmp
    fmp = fetch_fmp(ticker)
    fmp_fx = fx_rate(fmp.get("currency"), px_ccy) if fmp and fmp.get("currency") else fx
    compare_and_fill(rec, fmp, fmp_fx)

    return normalise(rec)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _year_ebitda(ebit, da, unusual, yahoo_norm, yahoo_rep):
    """Normalised EBITDA for one year and where it came from.

    Preferred: EBIT + D&A with unusual items added back. Yahoo's own EBITDA
    fields are fallbacks because they proved unreliable (e.g. TWE, CSL).
    Yahoo reports unusual items with their P&L sign (losses negative), so
    subtracting them adds back impairments and restructuring charges.
    """
    adj = 0.0 if _nan(unusual) else unusual
    if not _nan(ebit) and not _nan(da):
        return ebit + abs(da) - adj, "EBIT + D&A"
    if not _nan(yahoo_norm):
        return yahoo_norm, "Yahoo normalized EBITDA"
    if not _nan(yahoo_rep):
        return yahoo_rep - adj, "Yahoo EBITDA"
    return np.nan, "missing"


def normalise(rec: dict) -> dict:
    """Latest and 3-year normalised EBIT/EBITDA, plus a source check on EBITDA."""
    from .config import EBITDA_FIELD_TOLERANCE, MIN_YEARS_FOR_AVERAGE

    unusual = rec.get("unusual_items", np.nan)
    ebit_rep = rec.get("ebit_reported", np.nan)
    hist = rec.get("history")

    if hist:
        # Latest year uses the (possibly FMP-filled) headline fields
        hist["ebit"][0] = ebit_rep if not _nan(ebit_rep) else hist["ebit"][0]
        years = [_year_ebitda(hist["ebit"][i], hist["da"][i], hist["unusual"][i],
                              hist["ebitda_norm_yahoo"][i], hist["ebitda_yahoo"][i])
                 for i in range(len(hist["ebit"]))]
        ebitda_years = [v for v, _ in years]
        rec["ebitda"], rec["ebitda_source"] = years[0]
        if _nan(rec["ebitda"]) and not _nan(rec.get("ebitda_reported", np.nan)):
            rec["ebitda"], rec["ebitda_source"] = rec["ebitda_reported"], "FMP/Yahoo EBITDA"

        last3 = [v for v in ebitda_years[:3] if not _nan(v)]
        ints = [abs(v) for v in hist["interest"][:3] if not _nan(v)]
        enough = len(last3) >= MIN_YEARS_FOR_AVERAGE and len(ints) >= MIN_YEARS_FOR_AVERAGE
        rec["ebitda_3y"] = float(np.mean(last3)) if enough else np.nan
        rec["interest_3y"] = float(np.mean(ints)) if enough else np.nan
        rec["ebitda_history"] = ebitda_years

        # Source check: Yahoo's EBITDA field vs our EBIT + D&A build
        y = hist["ebitda_yahoo"][0]
        built_raw = (hist["ebit"][0] + abs(hist["da"][0])
                     if not _nan(hist["ebit"][0]) and not _nan(hist["da"][0]) else np.nan)
        if not _nan(y) and not _nan(built_raw):
            diff = abs(y - built_raw)
            if diff > 5e6 and diff > EBITDA_FIELD_TOLERANCE * max(abs(y), abs(built_raw)):
                rec.setdefault("crosscheck_issues", []).append(
                    f"Yahoo EBITDA field {y / 1e6:,.0f}m differs from EBIT + D&A "
                    f"{built_raw / 1e6:,.0f}m; used EBIT + D&A")
    else:
        ebitda_norm = rec.get("ebitda_normalized_yahoo", np.nan)
        if _nan(ebitda_norm):
            ebitda_rep = rec.get("ebitda_reported", np.nan)
            ebitda_norm = ebitda_rep - unusual if not _nan(unusual) else ebitda_rep
        rec["ebitda"], rec["ebitda_source"] = ebitda_norm, "Yahoo/FMP EBITDA"
        rec["ebitda_3y"] = rec["interest_3y"] = np.nan

    rec["ebit"] = ebit_rep - unusual if not _nan(unusual) else ebit_rep
    # True when the ratios really are one-off adjusted
    rec["normalised"] = (not _nan(unusual)
                         or not _nan(rec.get("ebitda_normalized_yahoo", np.nan)))
    return rec
