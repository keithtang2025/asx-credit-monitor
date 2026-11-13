"""Backtest engine checks on synthetic companies (no internet needed)."""
import numpy as np
import pandas as pd

from monitor.backtest import debt_asof, dd_path, evaluate, first_sustained
from monitor.models import _bs_equity


def _company(V_path, D, sigma=0.30, r=0.03):
    idx = pd.bdate_range("2019-01-01", periods=len(V_path))
    return pd.Series(_bs_equity(np.asarray(V_path), D, r, sigma, 1.0), idx)


def _sliding_to_default(n=1000, seed=0):
    """Assets drift from 3x debt down to below debt over ~4 years."""
    rng = np.random.default_rng(seed)
    drift = np.linspace(np.log(300), np.log(90), n)
    noise = np.cumsum(rng.normal(0, 0.30 / np.sqrt(252), n)) * 0.3
    return np.exp(drift + noise)


def test_debt_asof_respects_reporting_lag():
    sched = [("2023-06-30", 10, 20), ("2024-06-30", 30, 40)]
    # 1 Aug 2024: FY24 accounts not public yet (60-day lag) -> FY23 figures
    dp, assumed = debt_asof(sched, pd.Timestamp("2024-08-01"), 60)
    assert dp == 10 + 0.5 * 20 and not assumed
    dp, _ = debt_asof(sched, pd.Timestamp("2024-09-01"), 60)
    assert dp == 30 + 0.5 * 40


def test_debt_asof_flags_look_ahead_when_nothing_public_yet():
    dp, assumed = debt_asof([("2024-06-30", 10, 20)], pd.Timestamp("2023-01-01"), 60)
    assert assumed


def test_first_sustained_needs_consecutive_breaches():
    s = pd.Series([5, 1.9, 4, 1.8, 1.7, 1.5],
                  index=pd.date_range("2024-01-01", periods=6, freq="W"))
    hit = first_sustained(s, 2.0, 2)
    assert hit == s.index[3]          # the single dip at index 1 is ignored


def test_backtest_warns_before_a_synthetic_default():
    V = _sliding_to_default()
    mcap = _company(V, D=100)
    event = mcap.index[-1]
    path = dd_path(mcap, [("2018-12-31", 100, 0)], 0.03,
                   start=mcap.index[300], end=event, window=252, step=5, lag_days=60)
    res = evaluate("Synthetic", "failure", path, event, amber=3.5, red=2.0, confirm=2)
    assert res.first_amber is not None and res.first_red is not None
    assert res.first_amber <= res.first_red < event
    assert res.lead_amber_days > res.lead_red_days > 0


def test_backtest_quiet_for_a_healthy_control():
    rng = np.random.default_rng(3)
    V = 400 * np.exp(np.cumsum(rng.normal(0, 0.15 / np.sqrt(252), 900)))
    mcap = _company(V, D=60, sigma=0.15)
    path = dd_path(mcap, [("2018-12-31", 60, 0)], 0.03,
                   start=mcap.index[300], end=mcap.index[-1], window=252, step=5)
    res = evaluate("Healthy", "control", path, mcap.index[-1])
    assert res.first_red is None and res.pct_weeks_red == 0
