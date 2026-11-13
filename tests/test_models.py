"""Model checks on synthetic data (no internet needed).  Run: python -m pytest -q"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from monitor.engine import coverage_thresholds, run_watchlist, score_company, z_applies
from monitor.models import _bs_equity, altman_z2, merton_kmv, rag, status_v2, status_v3, status_v5, status_v8


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
    assert "one-offs suspected" in out["reasons"]   # no unusual-items data
    assert out["one_off_affected"] and not out["normalised"]
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
           "statement_currency": "AUD", "normalised": True}
    out = score_company(rec)
    assert out["flag_cov"] == "GREEN"        # 12x on normalised EBITDA
    assert out["one_off_affected"]
    assert out["status"] in ("GREEN", "WATCH")


# ---- v4: Star fix and second-source cross-check ----

def _star_like(normalised: bool):
    """Big one-offs, but normalised ratios still red (the Star pattern)."""
    E, _ = simulate_equity(100, 70, 0.30, seed=11)
    return {"ticker": "STR.AX", "name": "Star-like", "sector": "Consumer Cyclical",
            "industry": "Resorts & Casinos", "prices": E / 10, "market_cap_series": E,
            "short_term_debt": 30.0, "long_term_debt": 40.0, "total_debt": 70.0,
            "cash": 5.0, "total_assets": 150.0, "total_liabilities": 120.0,
            "working_capital": -20.0, "retained_earnings": -90.0, "book_equity": 30.0,
            "ebit": -15.0, "ebitda": -6.0,                  # still negative after add-back
            "ebit_reported": -160.0, "ebitda_reported": -150.0, "unusual_items": -144.0,
            "interest_expense": 5.0, "statement_date": "2026-06-30",
            "statement_currency": "AUD", "normalised": normalised}


def test_v4_star_stays_red_when_normalised_ratios_are_still_red():
    out = score_company(_star_like(normalised=True))
    assert out["flag_z"] == "RED" and out["flag_cov"] == "RED"
    assert out["status"] == "RED"
    assert out["red_signals"] >= 2


def test_v4_merge_only_when_not_normalisable():
    out = score_company(_star_like(normalised=False))
    assert "signals merged" in out["reasons"]


from monitor.crosscheck import compare_and_fill  # noqa: E402
from monitor.data import normalise  # noqa: E402


def _yahoo_rec(**over):
    rec = {"ticker": "CSL.AX", "statement_date": "2026-06-30",
           "total_debt": 1.0e10, "short_term_debt": 1.0e9, "long_term_debt": 9.0e9,
           "cash": 1.5e9, "total_assets": 3.8e10, "total_liabilities": 1.8e10,
           "retained_earnings": 1.2e10, "book_equity": 2.0e10,
           "ebit_reported": 4.0e9, "ebitda_reported": -3.5e9,   # bad Yahoo value
           "interest_expense": 5.0e8}
    rec.update(over)
    return rec


def _fmp(**over):
    f = {"date": "2026-06-30", "currency": "AUD",
         "total_debt": 1.0e10, "short_term_debt": 1.0e9, "long_term_debt": 9.0e9,
         "cash": 1.5e9, "total_assets": 3.8e10, "total_liabilities": 1.8e10,
         "retained_earnings": 1.2e10, "book_equity": 2.0e10,
         "ebit_reported": 4.0e9, "ebitda_reported": 5.2e9, "interest_expense": -5.0e8}
    f.update(over)
    return f


def test_crosscheck_flags_csl_style_bad_ebitda():
    rec = compare_and_fill(_yahoo_rec(), _fmp())
    assert rec["source_check"] == "MISMATCH"
    assert "ebitda_reported" in rec["crosscheck_issues"][-1]
    assert "interest_expense" not in rec["crosscheck_issues"][-1]   # sign convention ignored


def test_crosscheck_ok_when_sources_agree():
    rec = compare_and_fill(_yahoo_rec(ebitda_reported=5.1e9), _fmp())
    assert rec["source_check"] == "OK"


def test_crosscheck_fills_gaps_then_normalises():
    rec = _yahoo_rec(ebitda_reported=float("nan"), interest_expense=float("nan"))
    rec["unusual_items"] = float("nan")
    rec = normalise(compare_and_fill(rec, _fmp()))
    assert rec["ebitda"] == 5.2e9
    assert any("filled from FMP" in i for i in rec["crosscheck_issues"])


def test_crosscheck_different_periods_not_compared():
    rec = compare_and_fill(_yahoo_rec(), _fmp(date="2025-06-30"))
    assert rec["source_check"] == "PERIOD DIFFERS"


def test_crosscheck_without_key_is_na():
    assert compare_and_fill(_yahoo_rec(), None)["source_check"] == "N/A"


# ---- v5: EBITDA built from EBIT + D&A, 3-year averages, earnings trend ----

M = 1e6


def _company_with_history(sector, ebit, da, interest, ebitda_yahoo, seed=21,
                          opinc=None, cash=300 * M, debt=2200 * M):
    E, _ = simulate_equity(100, 12, 0.25, seed=seed)
    rec = {"ticker": "HIS.AX", "name": "Hist", "sector": sector, "industry": "",
           "prices": E, "market_cap_series": E * 1e8,
           "short_term_debt": 200 * M, "long_term_debt": 2000 * M, "total_debt": debt,
           "cash": cash, "total_assets": 8000 * M, "total_liabilities": 5000 * M,
           "working_capital": 0.0, "retained_earnings": -500 * M, "book_equity": 3000 * M,
           "ebit_reported": ebit[0], "ebitda_reported": ebitda_yahoo[0],
           "operating_income": opinc[0] if opinc else np.nan,
           "ebitda_normalized_yahoo": np.nan, "unusual_items": np.nan,
           "interest_expense": interest[0], "statement_date": "2026-06-30",
           "statement_currency": "AUD",
           "history": {"ebit": list(ebit), "da": list(da),
                       "opinc": list(opinc) if opinc else [np.nan] * 4,
                       "unusual": [np.nan] * 4, "interest": list(interest),
                       "ebitda_yahoo": list(ebitda_yahoo),
                       "ebitda_norm_yahoo": [np.nan] * 4}}
    return normalise(rec)


def _twe():
    """Treasury Wine FY26-FY23 from the user's screenshot (AUD, newest first)."""
    return _company_with_history(
        "Consumer Defensive",
        ebit=[57.5 * M, 518.1 * M, 42.6 * M, 182.7 * M],
        da=[165.3 * M, 166.5 * M, 155.5 * M, 161.7 * M],
        interest=[108.1 * M, 109.9 * M, 98.3 * M, 72.7 * M],
        ebitda_yahoo=[-508 * M, np.nan, np.nan, np.nan])   # the bad Yahoo value


def test_v5_twe_ebitda_built_from_ebit_plus_da():
    rec = _twe()
    assert abs(rec["ebitda"] - 222.8 * M) < 0.1 * M
    assert rec["ebitda_source"] == "EBIT + D&A"   # no operating-income row given


def test_v5_twe_cover_and_average_match_hand_calculation():
    out = score_company(_twe())
    assert abs(out["ebitda_cover_latest"] - 222.8 / 108.1) < 0.01      # 2.06x
    assert abs(out["ebitda_cover_3y"] - 368.5 / 105.43) < 0.02         # 3.50x
    assert out["coverage_basis"] == "latest"                            # not cyclical
    assert out["flag_cov"] == "AMBER"
    assert abs(out["earnings_vs_3y"] - 0.6046) < 0.001                  # just above 60%


def test_v5_twe_no_longer_red():
    out = score_company(_twe())
    assert out["status"] == "AMBER"


def _miner(sector):
    """Commodity downturn: one terrible year after two strong ones."""
    return _company_with_history(
        sector,
        ebit=[50 * M, 900 * M, 800 * M, 700 * M],
        da=[100 * M] * 4,
        interest=[100 * M] * 4,
        ebitda_yahoo=[150 * M, 1000 * M, 900 * M, 800 * M])


def test_v5_cyclical_uses_three_year_average():
    out = score_company(_miner("Basic Materials"))
    assert out["coverage_basis"] == "3y avg"
    assert out["flag_cov"] == "GREEN"          # 6.8x through the cycle
    assert out["flag_trend"] == "AMBER"        # but the fall is still flagged
    assert out["status"] in ("WATCH", "AMBER")
    assert "earnings falling" in out["reasons"]


def test_v5_same_numbers_non_cyclical_judged_on_latest_year():
    out = score_company(_miner("Healthcare"))
    assert out["coverage_basis"] == "latest"
    assert out["flag_cov"] == "RED"            # 1.5x latest
    assert out["status"] in ("AMBER", "RED")


def test_v5_trend_never_red_on_its_own():
    assert status_v5("GREEN", "GREEN", "GREEN", "GREEN", "AMBER")[0] == "WATCH"
    assert status_v5("GREEN", "AMBER", "GREEN", "GREEN", "AMBER")[0] == "AMBER"



# ---- v6: operating income instead of Yahoo EBIT, net cash, wording ----

def _twe_real():
    """TWE as Yahoo actually serves it: EBIT -673m (impairment below the
    operating line) while operating profit is +57.5m (user's screenshot)."""
    return _company_with_history(
        "Consumer Defensive",
        ebit=[-673 * M, 518.1 * M, 42.6 * M, 182.7 * M],
        opinc=[57.5 * M, 518.1 * M, 42.6 * M, 182.7 * M],
        da=[165.3 * M, 166.5 * M, 155.5 * M, 161.7 * M],
        interest=[108.1 * M, 109.9 * M, 98.3 * M, 72.7 * M],
        ebitda_yahoo=[-508 * M, 684.6 * M, 198.1 * M, 344.4 * M])


def test_v6_twe_uses_operating_income_not_yahoo_ebit():
    rec = _twe_real()
    assert abs(rec["ebitda"] - 222.8 * M) < 0.1 * M
    assert rec["ebitda_source"] == "operating income + D&A"
    assert any("non-operating items" in i and "-730m" in i
               for i in rec["crosscheck_issues"])
    assert rec["normalised"]


def test_v6_twe_real_pattern_is_amber_not_red():
    out = score_company(_twe_real())
    assert abs(out["ebitda_cover_latest"] - 2.06) < 0.01
    assert out["status"] == "AMBER"
    assert "-4.7x" not in out["reasons"]


def test_v6_unusual_items_not_double_counted_when_below_the_line():
    rec = _company_with_history(
        "Industrials", ebit=[-400 * M] * 4, opinc=[100 * M] * 4,
        da=[50 * M] * 4, interest=[20 * M] * 4, ebitda_yahoo=[np.nan] * 4)
    rec["history"]["unusual"] = [-500 * M] * 4
    rec["unusual_items"] = -500 * M
    rec = normalise(rec)
    assert rec["ebitda"] == 150 * M          # 100 + 50, the -500 is NOT added back


def test_v6_unusual_items_added_back_when_inside_operating_income():
    rec = _company_with_history(
        "Industrials", ebit=[-400 * M] * 4, opinc=[-400 * M] * 4,
        da=[50 * M] * 4, interest=[20 * M] * 4, ebitda_yahoo=[np.nan] * 4)
    rec["history"]["unusual"] = [-500 * M] * 4
    rec["unusual_items"] = -500 * M
    rec = normalise(rec)
    assert rec["ebitda"] == 150 * M          # -400 + 50 + 500 added back


def test_v6_net_cash_switches_cover_off():
    """IGO-style: earnings mostly below the line, more cash than debt."""
    rec = _company_with_history(
        "Basic Materials", ebit=[300 * M] * 4, opinc=[-60 * M] * 4,
        da=[20 * M] * 4, interest=[8 * M] * 4, ebitda_yahoo=[np.nan] * 4,
        cash=900 * M, debt=50 * M)
    out = score_company(rec)
    # v9: loss-making but cash covers debt -> funding runway, not a credit flag
    assert out["net_cash"] and out["credit_metric"] == "funding runway"
    assert out["flag_cov"] == "OFF" and out["flag_funding"] == "GREEN"
    assert "EBITDA cover" not in out["reasons"]


def test_v6_negative_earnings_wording():
    rec = _company_with_history(
        "Healthcare", ebit=[-50 * M, 400 * M, 400 * M, 400 * M],
        opinc=[-50 * M, 400 * M, 400 * M, 400 * M],
        da=[20 * M] * 4, interest=[30 * M] * 4, ebitda_yahoo=[np.nan] * 4)
    out = score_company(rec)
    assert "EBITDA turned negative" in out["reasons"]
    assert "%" not in out["reasons"].split("earnings falling")[1].split(";")[0]


# ---- ASX 300 universe ----

from monitor import config as CFG  # noqa: E402
from monitor.data import reconcile_shares  # noqa: E402


def test_universe_excludes_banks_and_insurers():
    for code in ["CBA", "NAB", "WBC", "ANZ", "MQG", "QBE", "IAG", "SUN", "ZIP"]:
        assert f"{code}.AX" not in CFG.WATCHLIST
    for code in ["BHP", "CSL", "WOW", "TCL", "NXT", "SGR"]:
        assert f"{code}.AX" in CFG.WATCHLIST
    assert 240 <= len(CFG.WATCHLIST) <= 270
    assert len(set(CFG.WATCHLIST)) == len(CFG.WATCHLIST)


def test_runtime_sector_exclusion():
    def fake(tk):
        E, _ = simulate_equity(100, 10, 0.2, seed=2)
        sector = "Financial Services" if tk == "FIN.AX" else "Industrials"
        return {"ticker": tk, "name": tk, "sector": sector, "industry": "",
                "prices": E / 10, "market_cap_series": E, "short_term_debt": 5.0,
                "long_term_debt": 5.0, "total_debt": 10.0, "cash": 1.0,
                "total_assets": 100.0, "total_liabilities": 30.0, "working_capital": 5.0,
                "retained_earnings": 20.0, "book_equity": 70.0, "ebit": 10.0,
                "ebitda": 14.0, "interest_expense": 1.0, "statement_date": "2026-06-30",
                "statement_currency": "AUD"}
    df = run_watchlist(["FIN.AX", "IND.AX"], fetch=fake, verbose=False)
    assert list(df["ticker"]) == ["IND.AX"]
    assert df.attrs["excluded"] == ["FIN.AX"]


def test_cdi_share_count_reconciled_to_market_cap():
    # ResMed-style CDI: Yahoo share count is underlying shares, 10 CDIs each
    shares, note = reconcile_shares(147e6, 40.0, 58.8e9)
    assert abs(shares - 1.47e9) < 1e6 and "CDI" in note
    shares, note = reconcile_shares(1.0e9, 10.0, 10.2e9)   # within 20%: unchanged
    assert shares == 1.0e9 and note is None



# ---- v8: right metric per business model; Z'' needs support ----

def _base(**over):
    E, _ = simulate_equity(100, 10, 0.20, seed=31)
    rec = {"ticker": "BAS.AX", "name": "Base", "sector": "Industrials", "industry": "",
           "prices": E, "market_cap_series": E * 1e8,
           "short_term_debt": 100 * M, "long_term_debt": 900 * M, "total_debt": 1000 * M,
           "cash": 100 * M, "total_assets": 5000 * M, "total_liabilities": 3000 * M,
           "working_capital": 50 * M, "retained_earnings": 500 * M, "book_equity": 2000 * M,
           "ebit": 400 * M, "ebitda": 600 * M, "interest_expense": 60 * M,
           "statement_date": "2026-06-30", "statement_currency": "AUD"}
    rec.update(over)
    return rec


def test_v8_z_red_alone_is_watch_not_amber():
    """Ramsay/Telstra pattern: weak Z'' (negative retained earnings), healthy otherwise."""
    out = score_company(_base(retained_earnings=-3000 * M, working_capital=-500 * M))
    assert out["flag_z"] == "RED" and out["flag_cov"] == "GREEN"
    assert out["status"] == "WATCH"


def test_v8_z_red_still_supports_other_warnings():
    out = score_company(_base(retained_earnings=-3000 * M, working_capital=-500 * M,
                              ebitda=180 * M))       # cover 3.0x -> AMBER
    assert out["status"] == "AMBER"


def test_v8_loss_maker_judged_on_cash_runway():
    """Imricor pattern: negative EBITDA, some debt. Cover would be absurd."""
    out = score_company(_base(ebitda=-40 * M, ebit=-45 * M, cash=25 * M,
                              total_debt=30 * M, short_term_debt=10 * M,
                              long_term_debt=20 * M, interest_expense=2 * M))
    assert out["credit_metric"] == "cash runway"
    assert abs(out["cash_runway_months"] - 12 * 25 / 42) < 0.1     # ~7 months
    assert out["flag_cov"] == "RED"
    assert "cash runway 7 months" in out["reasons"]
    assert "EBITDA cover" not in out["reasons"]


def test_v8_reit_judged_on_gearing_not_cover():
    """Dexus pattern: cover looked like 1.1x, but gearing is the lender's test."""
    reit = dict(sector="Real Estate", industry="REIT - Diversified",
                ebitda=70 * M, interest_expense=60 * M)          # cover 1.2x
    out = score_company(_base(total_debt=1500 * M, **reit))      # gearing 30%
    assert out["credit_metric"] == "gearing" and out["flag_cov"] == "GREEN"
    out = score_company(_base(total_debt=2400 * M, **reit))      # net gearing 47%
    assert out["flag_cov"] == "RED" and "gearing 47%" in out["reasons"]


def test_v8_toll_road_override_uses_higher_gearing_line():
    out = score_company(_base(ticker="ALX.AX", total_debt=3000 * M))   # 60%
    assert out["credit_metric"] == "gearing" and out["flag_cov"] == "AMBER"


def test_v8_net_cash_switches_z_off():
    """Explorer/biotech pattern: big accumulated losses, more cash than debt."""
    out = score_company(_base(cash=800 * M, total_debt=20 * M, short_term_debt=5 * M,
                              long_term_debt=15 * M, retained_earnings=-4000 * M,
                              ebitda=300 * M))
    assert out["flag_z"] == "OFF" and out["status"] in ("GREEN", "WATCH")


def test_v8_no_debt_display():
    out = score_company(_base(short_term_debt=0.0, long_term_debt=0.0, total_debt=0.0))
    assert out["dd_display"] == "no debt"


def test_v8_multi_signal_still_red():
    """Star pattern survives the softer Z'' rule: Z'' red + hard red = RED."""
    assert status_v8("AMBER", "RED", "RED", "GREEN")[0] == "RED"
    assert status_v8("GREEN", "RED", "GREEN", "AMBER")[0] == "AMBER"   # Z'' + drawdown
    assert status_v8("GREEN", "GREEN", "RED", "GREEN")[0] == "AMBER"   # hard red alone



# ---- v9 ----

def test_v9_missing_debt_lines_mean_no_debt():
    """EchoIQ pattern: Yahoo omits debt rows; must count as net cash, Z'' off."""
    nan = float("nan")
    out = score_company(_base(short_term_debt=nan, long_term_debt=nan, total_debt=nan,
                              cash=30 * M, retained_earnings=-4000 * M,
                              ebitda=-20 * M, ebit=-21 * M, interest_expense=nan))
    assert out["net_cash"] and out["flag_z"] == "OFF"


def test_v9_no_debt_short_runway_is_funding_watch():
    """PYC/Clarity pattern: 8 months of cash, no debt -> WATCH, not AMBER/RED."""
    out = score_company(_base(short_term_debt=0.0, long_term_debt=0.0, total_debt=0.0,
                              cash=40 * M, ebitda=-60 * M, ebit=-61 * M,
                              interest_expense=0.0))
    assert out["credit_metric"] == "funding runway" and out["flag_funding"] == "RED"
    assert out["status"] == "WATCH"
    assert "funding risk" in out["reasons"]


def test_v9_loss_maker_with_net_debt_keeps_credit_flag():
    out = score_company(_base(ebitda=-40 * M, ebit=-45 * M, cash=25 * M, total_debt=30 * M,
                              short_term_debt=10 * M, long_term_debt=20 * M,
                              interest_expense=2 * M))
    assert out["credit_metric"] == "cash runway" and out["flag_cov"] == "RED"
    assert out["status"] in ("AMBER", "RED")


def test_v9_infratil_on_infrastructure_gearing():
    out = score_company(_base(ticker="IFT.AX", ebitda=100 * M, interest_expense=60 * M))
    assert out["credit_metric"] == "gearing"


def test_v9_agribusiness_uses_three_year_average():
    rec = _company_with_history(
        "Consumer Defensive", ebit=[20 * M, 400 * M, 380 * M, 350 * M],
        opinc=[20 * M, 400 * M, 380 * M, 350 * M],
        da=[30 * M] * 4, interest=[60 * M] * 4, ebitda_yahoo=[np.nan] * 4)
    rec["ticker"] = "GNC.AX"
    out = score_company(rec)
    assert out["cyclical"] and out["coverage_basis"] == "3y avg"
    assert out["flag_cov"] != "RED"
    assert out["flag_trend"] == "AMBER"          # the bad harvest is still flagged


def test_v9_market_disagrees_note():
    """Nine pattern: accounts say crisis, market says fine -> check data note."""
    out = score_company(_base(ebitda=50 * M, interest_expense=60 * M))   # cover 0.8x
    if out["distance_to_default"] >= 4:
        assert "CHECK DATA: market" in out["reasons"]
