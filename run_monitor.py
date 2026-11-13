"""Run the monitor, save a snapshot, and flag changes since the last run.

Usage:
    python run_monitor.py                  # full watchlist
    python run_monitor.py QAN.AX SGR.AX    # specific tickers
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from monitor.engine import run_watchlist

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
HISTORY = DATA / "history.csv"
HISTORY_COLS = ["ticker", "name", "status", "reasons", "distance_to_default", "pd_1y",
                "altman_z2", "ebitda_coverage", "earnings_vs_3y", "drawdown_52w",
                "market_cap_bn"]


def compare_with_previous(today: pd.DataFrame) -> pd.DataFrame:
    """Status changes and DD moves versus the most recent earlier run."""
    if not HISTORY.exists():
        return pd.DataFrame()
    hist = pd.read_csv(HISTORY)
    prev_dates = sorted(d for d in hist["run_date"].unique() if d < str(date.today()))
    if not prev_dates:
        return pd.DataFrame()
    prev = hist[hist["run_date"] == prev_dates[-1]][["ticker", "status", "distance_to_default"]]
    m = today.merge(prev, on="ticker", suffixes=("", "_prev"))
    m["dd_change"] = m["distance_to_default"] - m["distance_to_default_prev"]
    changed = m[(m["status"] != m["status_prev"]) | (m["dd_change"].abs() >= 1.0)]
    return changed[["ticker", "name", "status_prev", "status",
                    "distance_to_default_prev", "distance_to_default", "dd_change"]]


def main():
    tickers = sys.argv[1:] or None
    print("Running ASX credit monitor...")
    df = run_watchlist(tickers)
    if df.empty:
        print("No results.")
        return

    changes = compare_with_previous(df)

    # History keeps the key columns only (the full table is in latest.csv)
    keep = [c for c in HISTORY_COLS if c in df.columns]
    snap = df[keep].copy()
    snap.insert(0, "run_date", str(date.today()))
    if HISTORY.exists():
        hist = pd.read_csv(HISTORY)
        hist = hist[hist["run_date"] != str(date.today())]  # rerun same day replaces
        hist = hist[[c for c in ["run_date"] + HISTORY_COLS if c in hist.columns]]
        snap = pd.concat([hist, snap], ignore_index=True)
    snap.to_csv(HISTORY, index=False)
    df.to_csv(DATA / "latest.csv", index=False)

    print("\n=== Watchlist status ===")
    print(df["status"].value_counts().to_string())
    if "source_check" in df and (df["source_check"] != "N/A").any():
        print("\n=== Data source cross-check (Yahoo vs FMP) ===")
        print(df["source_check"].value_counts().to_string())
    alerts = df[df["status"].isin(["RED", "AMBER"])]
    watch = df[df["status"] == "WATCH"]
    if len(alerts):
        print("\n=== Alerts ===")
        print(alerts[["ticker", "name", "status", "dd_display", "pd_1y",
                      "reasons"]].rename(columns={"dd_display": "DD"})
              .to_string(index=False, float_format="%.3f"))
    if len(watch):
        print("\n=== Watch list (single soft signal) ===")
        print(watch[["ticker", "name", "dd_display", "reasons"]]
              .rename(columns={"dd_display": "DD"}).to_string(index=False))
    dq = df[df["data_issues"].astype(str).str.len() > 0]
    if len(dq):
        print("\n=== Data quality checks (verify against annual reports) ===")
        print(dq[["ticker", "data_issues"]].to_string(index=False))
    if len(changes):
        print("\n=== Changes since last run ===")
        print(changes.to_string(index=False, float_format="%.2f"))
    if df.attrs.get("excluded"):
        print(f"\nExcluded at run time as financials: {', '.join(df.attrs['excluded'])}")
    if df.attrs.get("errors"):
        print(f"\nSkipped {len(df.attrs['errors'])} tickers:")
        for tk, why in df.attrs["errors"]:
            print(f"  {tk}: {why[:100]}")
    print(f"\nSaved {DATA / 'latest.csv'} and appended to {HISTORY}")


if __name__ == "__main__":
    main()
