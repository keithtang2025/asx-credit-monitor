"""Second data source (Financial Modeling Prep) and cross-check against Yahoo.

Set the FMP_API_KEY environment variable (a GitHub secret in Actions).
Without a key, or if FMP has no data for a ticker, the monitor runs on
Yahoo alone and records "N/A" in the source_check column.

Free tier: 250 requests a day. This module makes 2 per company.
"""
from __future__ import annotations

import os

import numpy as np
import requests

BASE = "https://financialmodelingprep.com/stable"

# Our field name -> FMP field name, and which statement it comes from
FIELDS = {
    "total_debt": ("balance", "totalDebt"),
    "short_term_debt": ("balance", "shortTermDebt"),
    "long_term_debt": ("balance", "longTermDebt"),
    "cash": ("balance", "cashAndCashEquivalents"),
    "total_assets": ("balance", "totalAssets"),
    "total_liabilities": ("balance", "totalLiabilities"),
    "retained_earnings": ("balance", "retainedEarnings"),
    "book_equity": ("balance", "totalStockholdersEquity"),
    "ebit_reported": ("income", "operatingIncome"),
    "ebitda_reported": ("income", "ebitda"),
    "interest_expense": ("income", "interestExpense"),
}

# Fields compared between sources (the ones that drive the scores)
COMPARE = ["total_debt", "total_assets", "ebitda_reported", "ebit_reported",
           "interest_expense"]

TOLERANCE = 0.10        # flag differences larger than 10%
MIN_ABS = 5e6           # ignore differences smaller than 5m (rounding noise)


def _get(endpoint: str, symbol: str, key: str) -> dict | None:
    try:
        r = requests.get(f"{BASE}/{endpoint}",
                         params={"symbol": symbol, "limit": 1, "apikey": key},
                         timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        return data[0] if isinstance(data, list) and data else None
    except (requests.RequestException, ValueError):
        return None


def fetch_fmp(symbol: str, key: str | None = None) -> dict | None:
    """Latest annual statements from FMP, as {our_field: value, ...}."""
    key = key or os.environ.get("FMP_API_KEY")
    if not key:
        return None
    bal = _get("balance-sheet-statement", symbol, key)
    inc = _get("income-statement", symbol, key)
    if not bal and not inc:
        return None
    src = {"balance": bal or {}, "income": inc or {}}
    out = {}
    for ours, (stmt, theirs) in FIELDS.items():
        v = src[stmt].get(theirs)
        out[ours] = float(v) if v is not None else np.nan
    out["date"] = (bal or inc).get("date")
    out["currency"] = (bal or inc).get("reportedCurrency")
    return out


def _close(a: float, b: float) -> bool:
    if np.isnan(a) or np.isnan(b):
        return True
    a, b = abs(a), abs(b)            # sign conventions differ for interest
    diff = abs(a - b)
    return diff <= MIN_ABS or diff <= TOLERANCE * max(a, b)


def compare_and_fill(rec: dict, fmp: dict | None, fx: float = 1.0) -> dict:
    """Cross-check a Yahoo record against FMP and fill Yahoo gaps.

    fx converts FMP values into the record's price currency.
    Adds rec["source_check"] ("OK", "MISMATCH", "PERIOD DIFFERS", "N/A")
    and rec["crosscheck_issues"] (list of strings). Returns rec.
    """
    rec.setdefault("crosscheck_issues", [])
    if not fmp:
        rec["source_check"] = "N/A"
        return rec

    f = {k: (v * fx if isinstance(v, float) else v) for k, v in fmp.items()}

    # Fill gaps in Yahoo from FMP
    filled = []
    for k in FIELDS:
        if k in rec and np.isnan(rec.get(k, np.nan)) and not np.isnan(f.get(k, np.nan)):
            rec[k] = f[k]
            filled.append(k)
    if filled:
        rec["crosscheck_issues"].append("filled from FMP: " + ", ".join(filled))

    # Only compare like with like
    if rec.get("statement_date") and f.get("date") and \
            str(rec["statement_date"])[:7] != str(f["date"])[:7]:
        rec["source_check"] = "PERIOD DIFFERS"
        rec["crosscheck_issues"].append(
            f"Yahoo {rec['statement_date']} vs FMP {f['date']}, not compared")
        return rec

    mismatches = []
    for k in COMPARE:
        y, m = rec.get(k, np.nan), f.get(k, np.nan)
        if not _close(y, m):
            mismatches.append(f"{k} Yahoo {y / 1e6:,.0f}m vs FMP {m / 1e6:,.0f}m")
        elif not np.isnan(y) and not np.isnan(m) and k.startswith("ebit") \
                and np.sign(y) != np.sign(m) and abs(y - m) > MIN_ABS:
            mismatches.append(f"{k} sign differs")
    if mismatches:
        rec["source_check"] = "MISMATCH"
        rec["crosscheck_issues"].append("sources disagree: " + "; ".join(mismatches))
    else:
        rec["source_check"] = "OK"
    return rec
