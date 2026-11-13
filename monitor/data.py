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

    bs, inc = t.balance_sheet, t.income_stmt
    fin_ccy = info.get("financialCurrency") or "AUD"
    px_ccy = info.get("currency") or "AUD"
    fx = fx_rate(fin_ccy, px_ccy)  # convert statements into price currency

    g = lambda df, names: _first(df, names) * fx
    shares = info.get("sharesOutstanding") or g(bs, ["Ordinary Shares Number", "Share Issued"]) / fx

    ebit_rep = g(inc, ["EBIT", "Operating Income"])
    ebitda_rep = g(inc, ["EBITDA"])
    unusual = g(inc, ["Total Unusual Items", "Total Unusual Items Excluding Goodwill"])
    ebitda_norm = g(inc, ["Normalized EBITDA"])
    if np.isnan(ebitda_norm):
        ebitda_norm = ebitda_rep - unusual if not np.isnan(unusual) else ebitda_rep
    # Yahoo reports unusual items with their P&L sign (losses negative),
    # so subtracting them adds back impairments and restructuring charges.
    ebit_norm = ebit_rep - unusual if not np.isnan(unusual) else ebit_rep

    return {
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
        "ebit": ebit_norm,
        "ebitda": ebitda_norm,
        "ebit_reported": ebit_rep,
        "ebitda_reported": ebitda_rep,
        "unusual_items": unusual,
        "interest_expense": g(inc, ["Interest Expense", "Interest Expense Non Operating"]),
        "statement_date": str(bs.columns[0].date()) if bs is not None and not bs.empty else None,
    }
