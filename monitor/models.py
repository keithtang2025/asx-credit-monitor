"""Credit risk models: Merton/KMV distance to default, Altman Z'', coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


# ---------------------------------------------------------------- Merton ---

def default_point(short_term_debt: float, long_term_debt: float) -> float:
    """KMV convention: short-term debt plus half of long-term debt."""
    return float(short_term_debt or 0.0) + 0.5 * float(long_term_debt or 0.0)


def _bs_equity(V, D, r, sigma, T):
    d1 = (np.log(V / D) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)


def _implied_asset_value(E, D, r, sigma, T):
    """Solve E = BS call on assets V with strike D for V."""
    f = lambda V: _bs_equity(V, D, r, sigma, T) - E
    lo, hi = max(E, 1e-9), E + D * 2 + 1.0
    return brentq(f, lo, hi, xtol=1e-6 * (E + D))


def merton_kmv(equity_values: pd.Series, debt: float, r: float = 0.04,
               T: float = 1.0, max_iter: int = 100, tol: float = 1e-4) -> dict:
    """Iterative KMV estimation of asset value and asset volatility.

    equity_values: daily market capitalisation series (same currency as debt).
    debt: default point.
    Returns asset value, asset volatility, distance to default and PD.
    The drift used is the risk-free rate, so PD is risk-neutral (conservative).
    """
    E = equity_values.dropna().astype(float).values
    if len(E) < 60 or debt <= 0:
        return {"asset_value": np.nan, "asset_vol": np.nan,
                "distance_to_default": np.nan if debt > 0 else np.inf,
                "pd_1y": np.nan if debt > 0 else 0.0, "iterations": 0}

    V = E + debt
    sigma = np.std(np.diff(np.log(V)), ddof=1) * np.sqrt(252)
    sigma = max(sigma, 1e-3)
    it = 0
    for it in range(1, max_iter + 1):
        V = np.array([_implied_asset_value(e, debt, r, sigma, T) for e in E])
        new_sigma = max(np.std(np.diff(np.log(V)), ddof=1) * np.sqrt(252), 1e-3)
        if abs(new_sigma - sigma) < tol:
            sigma = new_sigma
            break
        sigma = new_sigma

    V_t = V[-1]
    dd = (np.log(V_t / debt) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return {"asset_value": V_t, "asset_vol": sigma,
            "distance_to_default": dd, "pd_1y": norm.cdf(-dd), "iterations": it}


def rolling_dd(equity_values: pd.Series, debt: float, r: float, T: float,
               window: int = 252, step: int = 5) -> pd.Series:
    """Distance to default through time (holding the default point fixed).

    Fast approximation: one KMV calibration per step using the trailing window.
    """
    s = equity_values.dropna()
    out = {}
    for end in range(window, len(s) + 1, step):
        res = merton_kmv(s.iloc[end - window:end], debt, r, T, max_iter=30)
        out[s.index[end - 1]] = res["distance_to_default"]
    return pd.Series(out, name="distance_to_default")


# ---------------------------------------------------------- Accounting ---

def altman_z2(working_capital, retained_earnings, ebit, book_equity,
              total_assets, total_liabilities) -> float:
    """Altman Z'' (emerging-market / non-manufacturer version).

    Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4
    Distress < 1.1, grey 1.1 to 2.6, safe > 2.6.
    """
    try:
        if not total_assets or not total_liabilities:
            return np.nan
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = book_equity / total_liabilities
        return 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    except (TypeError, ZeroDivisionError):
        return np.nan


def interest_coverage(ebit, interest_expense) -> float:
    if interest_expense is None or pd.isna(interest_expense) or interest_expense == 0:
        return np.inf if (ebit or 0) > 0 else np.nan
    return ebit / abs(interest_expense)


def net_debt_to_ebitda(total_debt, cash, ebitda) -> float:
    if ebitda is None or pd.isna(ebitda) or ebitda <= 0:
        return np.nan
    return ((total_debt or 0) - (cash or 0)) / ebitda


# --------------------------------------------------------------- Flags ---

def rag(value, red, amber) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if value < red:
        return "RED"
    if value < amber:
        return "AMBER"
    return "GREEN"


def overall_status(flags: list[str]) -> str:
    if "RED" in flags:
        return "RED"
    if "AMBER" in flags:
        return "AMBER"
    if all(f == "N/A" for f in flags):
        return "N/A"
    return "GREEN"
