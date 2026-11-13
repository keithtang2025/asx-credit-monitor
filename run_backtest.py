"""Backtest distance to default on real credit events.

Usage:
    python run_backtest.py                 # all cases in backtest_cases.py
    python run_backtest.py "Star Entertainment"

Writes backtest/results/: one CSV and one PNG per case, plus summary.md.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import backtest_cases as B
from monitor import config as C
from monitor.backtest import (dd_path, evaluate, load_prices, load_shares,
                              yahoo_debt_schedule)

OUT = Path(__file__).parent / "backtest" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RED, AMBER = C.THRESHOLDS["distance_to_default"]


def run_case(case: dict):
    ev = pd.Timestamp(case["event_date"])
    start = ev - pd.DateOffset(years=case["lookback_years"])
    prices, src = load_prices(case["ticker"], start - pd.Timedelta(days=420), ev)

    if "MarketCap" in prices:
        mcap = prices["MarketCap"]
    else:
        mcap = prices["Close"] * load_shares(case["ticker"], prices, case.get("shares"))

    schedules = []
    debt = case.get("debt") or yahoo_debt_schedule(case["ticker"])
    schedules.append(("financial debt" if case.get("debt_alt") else "debt", debt))
    if case.get("debt_alt"):
        schedules.append((case.get("alt_label", "alternative"), case["debt_alt"]))

    results, paths = [], {}
    for label, sched in schedules:
        path = dd_path(mcap, sched, case["risk_free"], start, ev,
                       window=B.WINDOW_DAYS, step=B.STEP_DAYS,
                       lag_days=B.REPORTING_LAG_DAYS)
        if path.empty:
            raise ValueError("not enough price history inside the window")
        name = case["name"] + ("" if len(schedules) == 1 else f" ({label})")
        res = evaluate(name, case["group"], path, ev, AMBER, RED, B.CONFIRM_OBS)
        results.append(res)
        paths[label] = path
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path.to_csv(OUT / f"{safe}.csv")

    chart(case, paths, src)
    return results


def chart(case, paths, src):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    styles = ["-", "--"]
    for (label, p), ls in zip(paths.items(), styles):
        ax.plot(p.index, p["dd"], ls, color="#2b4c7e", lw=1.6,
                label=f"Distance to default ({label})" if len(paths) > 1 else
                "Distance to default")
    ax.axhspan(-10, RED, color="#c0392b", alpha=0.08)
    ax.axhspan(RED, AMBER, color="#e6a700", alpha=0.10)
    ax.axhline(RED, color="#c0392b", lw=0.8, ls=":")
    ax.axhline(AMBER, color="#e6a700", lw=0.8, ls=":")
    lo = min(p["dd"].min() for p in paths.values())
    hi = max(p["dd"].max() for p in paths.values())
    ax.set_ylim(min(lo - 0.5, 0), max(hi + 0.5, AMBER + 1))
    ev = pd.Timestamp(case["event_date"])
    ax.axvline(ev, color="black", lw=1)
    ax.annotate(case["event"].split("(")[0].strip(), xy=(ev, ax.get_ylim()[1]),
                xytext=(-6, -14), textcoords="offset points", ha="right",
                fontsize=8)
    ax.set_title(f"{case['name']} ({case['ticker']}): {case['group'].replace('_', ' ')}",
                 fontsize=11, loc="left")
    ax.set_ylabel("Distance to default")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.text(0.99, 0.02, f"Prices: {src}", transform=ax.transAxes, ha="right",
            fontsize=7, color="grey")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"{case['name'].replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def fmt_date(d):
    return d.strftime("%d %b %Y") if d is not None and not pd.isna(d) else "never"


def months(days):
    return f"{days / 30.4:.0f} months" if days is not None else "n/a"


def write_summary(all_results, skipped):
    lines = ["# Distance-to-default backtest", "",
             f"Thresholds: RED below {RED}, AMBER below {AMBER}. A breach counts "
             f"only if it lasts {B.CONFIRM_OBS} consecutive weekly observations. "
             f"Balance sheets are used {B.REPORTING_LAG_DAYS} days after their date "
             "to avoid look-ahead.", ""]
    lines += ["## Failures and near misses (did it warn, and how early?)", "",
              "| Case | Event | First AMBER | Lead | First RED | Lead | Lowest DD |",
              "|---|---|---|---|---|---|---|"]
    for r in all_results:
        if r.group in ("failure", "near_miss"):
            lines.append(
                f"| {r.name} | {fmt_date(r.event_date)} | {fmt_date(r.first_amber)} | "
                f"{months(r.lead_amber_days)} | {fmt_date(r.first_red)} | "
                f"{months(r.lead_red_days)} | {r.min_dd + 0.0:.2f} |")
    lines += ["", "## Controls (how often did it cry wolf?)", "",
              "| Case | Weeks AMBER or worse | Weeks RED | Lowest DD | Lowest on |",
              "|---|---|---|---|---|"]
    for r in all_results:
        if r.group == "control":
            lines.append(
                f"| {r.name} | {r.pct_weeks_amber_or_worse:.0%} | {r.pct_weeks_red:.0%} | "
                f"{r.min_dd:.2f} | {fmt_date(r.min_dd_date)} |")
    caveats = [r for r in all_results if r.debt_assumed_share > 0]
    if caveats:
        lines += ["", "## Look-ahead caveat", ""]
        for r in caveats:
            lines.append(f"- {r.name}: {r.debt_assumed_share:.0%} of observations used a "
                         "balance sheet that was not yet public (no earlier figures "
                         "supplied). Add earlier debt figures to remove this.")
    if skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"- {n}: {why}" for n, why in skipped]
    lines += ["", "Charts and weekly data: one PNG and CSV per case in this folder."]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    only = set(sys.argv[1:])
    results, skipped = [], []
    for case in B.CASES:
        if only and case["name"] not in only:
            continue
        print(f"Running {case['name']}...")
        try:
            results += run_case(case)
        except Exception as e:
            print(f"  skipped: {e}")
            skipped.append((case["name"], str(e)))
    write_summary(results, skipped)


if __name__ == "__main__":
    main()
