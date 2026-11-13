"""Model checks on synthetic data (no internet needed).  Run: python -m pytest -q"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from monitor.engine import coverage_thresholds, run_watchlist, score_company, z_applies
from monitor.models import _bs_equity, altman_z2, merton_kmv, rag, status_v2, status_v3


def simulate_equity(V0, D, sigma_v, r=0.04, T=1.0, n=252, seed=0):
    """Simulate asset paths, convert to equity via Black-Scholes."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal((r - 0.5 * sigma_v**2) * dt, sigma_v * np.sqrt(dt), n)
    V = V0 * np.exp(np.cumsum(shocks))
    E = _bs_equity(V, D, r, sigma_v, T)
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.Series(E, idx), V


def test_kmv_recovers_true_asset_vol_and_dd():
    D, sigma_true, r = 60.0, 0.25, 0.04
    E, V = simulate_equity(100.0, D, sigma_true, r)
    res = merton_kmv(E, D, r)
    assert abs(res["asset_vol"] - sigma_true) < 0.03
    true_dd = (np.log(V[-1] / D) + (r - 0.5 * sigma_true**2)) / sigma_true
    assert abs(res["distance_to_default"] - true_dd) < 0.3
    assert abs(res["pd_1y"] - norm.cdf(-true_dd)) < 0.02


def test_more_leverage_means_lower_dd():
    safe = merton_kmv(simulate_equity(100, 20, 0.3)[0], 20)
    risky = merton_kmv(simulate_equity(100, 85, 0.3)[0], 85)
    assert risky["distance_to_default"] < safe["distance_to_default"]
    assert risky["pd_1y"] > safe["pd_1y"]


def test_no_debt_is_zero_pd():
    res = merton_kmv(simulate_equity(100, 1, 0.3)[0], 0.0)
    assert res["pd_1y"] == 0.0


def test_altman_and_rag():
    z = altman_z2(10, 20, 8, 50, 100, 50)
    assert np.isclose(z, 6.56 * .1 + 3.26 * .2 + 6.72 * .08 + 1.05 * 1.0)
    assert rag(0.5, 1.1, 2.6) == "RED"
    assert rag(2.0, 1.1, 2.6) == "AMBER"
    assert rag(3.0, 1.1, 2.6) == "GREEN"
    assert rag(np.nan, 1, 2) == "N/A"


def test_engine_end_to_end_with_fake_fetch():
    def fake(tk):
        D = {"SAFE.AX": 10.0, "RISKY.AX": 90.0}[tk]
        E, _ = simulate_equity(100, D, 0.35, seed=1)
        return {"ticker": tk, "name": tk, "sector": "Test", "industry": "Widgets",
                "prices": E / 10,
                "market_cap_series": E, "short_term_debt": D, "long_term_debt": 0.0,
                "total_debt": D, "cash": 1.0, "total_assets": 100.0,
                "total_liabilities": D, "working_capital": 5.0, "retained_earnings": 5.0,
                "book_equity": 100 - D, "ebit": 6.0, "ebitda": 9.0,
                "interest_expense": D * 0.06, "statement_date": "2025-06-30",
                "statement_currency": "AUD"}
    df = run_watchlist(["SAFE.AX", "RISKY.AX"], fetch=fake, verbose=False)
    assert df.iloc[0]["ticker"] == "RISKY.AX"  # worst sorted first
    assert df.set_index("ticker").loc["RISKY.AX", "status"] == "RED"
    safe = df.set_index("ticker").loc["SAFE.AX"]
    assert safe["flag_dd"] == "GREEN" and safe["flag_cov"] == "GREEN"


def test_status_v2_rules():
    # One weak accounting ratio alone is only AMBER
    assert status_v2("GREEN", ["GREEN", "RED", "GREEN"], "GREEN") == "AMBER"
    # Two independent reds escalate to RED
    assert status_v2("GREEN", ["GREEN", "RED", "RED"], "GREEN") == "RED"
    # DD red on its own is RED
    assert status_v2("RED", ["RED", "GREEN", "GREEN"], "GREEN") == "RED"
    # Switched-off Z'' does not count
    assert status_v2("GREEN", ["GREEN", "OFF", "GREEN"], "GREEN") == "GREEN"
    assert status_v2("GREEN", ["GREEN", "OFF", "GREEN"], "AMBER") == "AMBER"


def test_z_switched_off_for_supermarkets_and_toll_roads():
    assert not z_applies({"ticker": "XYZ.AX", "industry": "Grocery Stores", "sector": ""})
    assert not z_applies({"ticker": "TCL.AX", "industry": "", "sector": ""})
    assert z_applies({"ticker": "BSL.AX", "industry": "Steel", "sector": "Basic Materials"})


def test_one_off_ebit_flag_and_data_issues():
    E, _ = simulate_equity(100, 10, 0.25, seed=3)
    rec = {"ticker": "ONE.AX", "name": "One", "sector": "Healthcare", "industry": "Biotech",
           "prices": E / 10, "market_cap_series": E, "short_term_debt": 5.0,
           "long_term_debt": 10.0, "total_debt": 15.0, "cash": 3.0, "total_assets": 120.0,
           "total_liabilities": 40.0, "working_capital": 10.0, "book_equity": 80.0,
           "retained_earnings": float("nan"), "ebit": -8.0, "ebitda": 25.0,
           "interest_expense": 2.0, "statement_date": "2020-06-30",
           "statement_currency": "USD"}
    out = score_company(rec)
    assert "one-off items" in out["reasons"]
    assert out["one_off_affected"]
    assert out["flag_cov"] == "GREEN"          # 12.5x EBITDA cover
    assert "retained_earnings" in out["data_issues"]
    assert "months old" in out["data_issues"]
    assert "converted from USD" in out["data_issues"]


# ---- v3 rules, using patterns from the first live runs ----

def test_v3_single_grey_zone_is_watch():          # e.g. Xero: Z'' 2.13 only
    assert status_v3("GREEN", "AMBER", "GREEN", "GREEN")[0] == "WATCH"


def test_v3_drawdown_alone_is_watch():            # e.g. Pilbara: drawdown 58% only
    assert status_v3("GREEN", "GREEN", "GREEN", "AMBER")[0] == "WATCH"


def test_v3_drawdown_plus_accounting_is_amber():  # e.g. Nickel Industries
    assert status_v3("GREEN", "AMBER", "GREEN", "AMBER")[0] == "AMBER"


def test_v3_dd_amber_is_amber():                  # e.g. NEXTDC, Lendlease
    assert status_v3("AMBER", "GREEN", "GREEN", "GREEN")[0] == "AMBER"


def test_v3_shared_cause_counts_once():           # e.g. Treasury Wine impairment
    assert status_v3("GREEN", "RED", "RED", "GREEN", shared_cause=False)[0] == "RED"
    status, n = status_v3("GREEN", "RED", "RED", "GREEN", shared_cause=True)
    assert status == "AMBER" and n == 1


def test_v3_independent_reds_still_red():         # e.g. Star: DD amber, Z'' and cover red
    assert status_v3("AMBER", "RED", "RED", "GREEN")[0] == "RED"


def test_sector_coverage_thresholds():
    assert coverage_thresholds({"ticker": "TCL.AX", "industry": "", "sector": ""}) == (1.5, 2.5)
    assert coverage_thresholds({"ticker": "X.AX", "industry": "REIT - Retail",
                                "sector": "Real Estate"}) == (1.75, 2.75)
    assert coverage_thresholds({"ticker": "X.AX", "industry": "Steel",
                                "sector": "Basic Materials"}) == (2.0, 4.0)


def test_normalised_coverage_avoids_false_red():
    """Reported EBIT/EBITDA hit by an impairment; normalised figures are healthy."""
    E, _ = simulate_equity(100, 10, 0.2, seed=5)
    rec = {"ticker": "IMP.AX", "name": "Imp", "sector": "Consumer Defensive",
           "industry": "Beverages", "prices": E / 10, "market_cap_series": E,
           "short_term_debt": 5.0, "long_term_debt": 20.0, "total_debt": 25.0,
           "cash": 2.0, "total_assets": 150.0, "total_liabilities": 50.0,
           "working_capital": 15.0, "retained_earnings": 30.0, "book_equity": 100.0,
           "ebit": 18.0, "ebitda": 24.0,                  # normalised
           "ebit_reported": -12.0, "ebitda_reported": -6.0, "unusual_items": -30.0,
           "interest_expense": 2.0, "statement_date": "2026-06-30",
           "statement_currency": "AUD"}
    out = score_company(rec)
    assert out["flag_cov"] == "GREEN"        # 12x on normalised EBITDA
    assert out["one_off_affected"]
    assert out["status"] in ("GREEN", "WATCH")
