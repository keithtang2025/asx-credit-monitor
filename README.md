# ASX Credit Risk Monitor (MVP)

An early-warning monitor for non-financial ASX companies. It combines a
market-implied signal (Merton/KMV distance to default) with accounting
signals (Altman Z'', interest coverage, net debt / EBITDA) and a
market-stress flag (52-week drawdown). Each company gets a RED / AMBER /
GREEN status with the reasons listed, and every run is saved so you can
see what changed since last time.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q             # model checks on synthetic data
python run_monitor.py           # score the watchlist, save data/latest.csv
streamlit run app.py            # dashboard
```

Score specific names only: `python run_monitor.py QAN.AX SGR.AX TAH.AX`

## Running on GitHub

1. Create a new repo on GitHub and upload this folder (including the
   hidden `.github` folder).
2. Actions tab: enable workflows, open "Daily credit monitor", click
   "Run workflow" to test. The results summary appears on the run page and
   `data/latest.csv` and `data/history.csv` are committed back.
3. It then runs automatically at 07:00 UTC on weekdays.
4. Dashboard: sign in at share.streamlit.io with GitHub, choose the repo,
   main file `app.py`. It reads the committed `data/latest.csv`, so the
   page stays up to date after every Action run.

Note: GitHub pauses scheduled workflows after 60 days with no repo
activity. The daily commit counts as activity, so it keeps itself alive.

## Data source

v6 runs on Yahoo Finance only. The FMP cross-check code is kept in
`monitor/crosscheck.py` but switched off (`USE_SECOND_SOURCE = False` in
`config.py`). To switch it back on, set it to True and add an `FMP_API_KEY`
repository secret plus an `env:` line in the workflow's "Run monitor" step.

## How it works (v6 scoring)

| Signal | Method | RED | AMBER |
|---|---|---|---|
| Distance to default (lead signal) | KMV iteration on 1 year of daily market cap; default point = ST debt + 0.5 x LT debt; risk-neutral drift | < 2.0 | < 3.5 |
| Altman Z'' (sector-aware) | Uses EBIT normalised for one-offs; switched OFF for supermarkets, airlines, travel, infrastructure, utilities and property | < 1.1 | < 2.6 |
| Interest cover | Normalised EBITDA / interest. Cyclicals (Basic Materials, Energy) use the 3-year average; everyone else the latest year. Thresholds by sector (below) | sector | sector |
| Earnings trend | Latest normalised EBITDA versus its 3-year average | | < 60% |
| Market stress | Drawdown from 52-week high | | >= 40% |

Interest cover thresholds (red / amber): default 2.0x / 4.0x; REITs and
property 1.75x / 2.75x; infrastructure, toll roads, pipelines and
utilities 1.5x / 2.5x.

**Status rule**

- **RED**: distance to default below 2.0, or two independent signals red
- **AMBER**: distance to default amber, or one accounting signal red, or two
  accounting signals amber, or a drawdown together with any accounting warning
- **WATCH**: a single accounting signal amber, or a drawdown on its own
- **GREEN**: everything else

**How EBITDA is built.** EBITDA is operating income plus depreciation and
amortisation, for each of the last 3 to 4 years. Yahoo's "EBIT" line is not
used as the base, because it is pre-tax profit plus interest and so includes
non-operating items such as impairments booked below the operating line.
Treasury Wine FY26: Yahoo EBIT -673m, operating profit +57.5m. When the two
differ by more than 10%, `data_issues` shows the gap.

Unusual items are added back only when they sit inside operating income
(EBIT and operating income agree). When they sit below the line, operating
income already excludes them, so adding them back would double count.

**Net cash.** When cash exceeds total debt, interest cover is switched OFF:
tiny interest costs make the ratio swing wildly and it says nothing about
default risk (e.g. IGO, whose earnings come mainly from an equity-accounted
joint venture below operating income).

**Through the cycle.** Mining and energy earnings swing with commodity
prices, so their interest cover is judged on the 3-year average. Averages
hide deterioration, so every company also gets an earnings-trend signal:
latest EBITDA below 60% of its 3-year average adds an AMBER-level signal.
On its own it gives WATCH; it never creates a RED by itself.

**One-off adjustment.** Ratios use Yahoo's "Normalized EBITDA" and EBIT with
unusual items added back. If the ratios have been cleaned this way and are
still red, the weakness is underlying, so Z'' and interest cover count as two
independent signals. Only when one-offs are suspected but cannot be removed
(no unusual-items data) do the two count as one signal.

**Data quality.** The `data_issues` column lists missing fields, statements
older than about 15 months and currency conversion. Renamed tickers are
mapped in `TICKER_ALIASES` (e.g. SVW to SGH); delisted names are removed
from the watchlist (QUB, taken private in 2026).

**Version history**

- v1: any red ratio made a company RED; about a third of the list was RED.
- v2: distance to default leads, sector-aware Z'', EBITDA cover. RED fell to
  two names, but one-off impairments still created false REDs and AMBER was
  crowded with grey-zone scores.
- v3: normalised earnings, sector cover thresholds, shared-cause rule and a
  WATCH tier so AMBER is a short list worth reading.
- v4: fixed the shared-cause rule, which wrongly demoted Star to AMBER by
  merging signals that were already cleaned of one-offs. Added FMP as a
  second source to cross-check Yahoo and fill gaps.
- v5: EBITDA built from EBIT + D&A (fixes the Treasury Wine false RED),
  3-year average cover for cyclicals, and an earnings-trend signal.
- v6: Yahoo only. Operating income replaces Yahoo EBIT (the real cause of
  the Treasury Wine false RED), no double counting of below-the-line
  one-offs, interest cover off for net-cash companies, clearer wording for
  negative earnings.

Thresholds live in `monitor/config.py`. Treat them as a starting risk
appetite and recalibrate after backtesting.

## Project layout

```
monitor/config.py   watchlist, risk-free rate, thresholds
monitor/data.py     yfinance fetch, statement field mapping, FX conversion
monitor/models.py   Merton/KMV, Altman Z'', coverage, RAG logic
monitor/engine.py   scores one company and the whole watchlist
monitor/crosscheck.py  optional FMP second source (off by default)
run_monitor.py      CLI run, snapshot history, change detection
app.py              Streamlit dashboard with drill-down DD history
tests/              synthetic-data tests (KMV recovers known asset vol and DD)
```

## Known limitations (be upfront about these in interviews)

- **Financials excluded.** Banks and insurers fund themselves with deposits
  and policy liabilities, so Merton "debt" is not meaningful for them.
- **Annual statements.** yfinance gives the latest annual balance sheet, so
  the default point can be up to 12 months stale. Half-year results are
  the obvious upgrade.
- **Risk-neutral PD.** PDs use the risk-free drift. They rank risk well but
  are not calibrated real-world default rates. Map DD to PD empirically
  later (the Moody's EDF idea).
- **Field mapping.** Yahoo statement labels vary by company. Check any
  surprising output against the annual report.
- **Currency.** Companies reporting in USD (e.g. BHP, RIO) are converted
  to the price currency using spot FX.
- **REITs.** DD works, but Altman Z'' is less meaningful for property
  trusts. Gearing and ICR versus covenants matter more.

## Backtest

Does distance to default warn before real credit events? `run_backtest.py`
replays it week by week using only information available at each date
(balance sheets count from 60 days after their date).

Cases live in `backtest_cases.py`:

| Case | Type | Event |
|---|---|---|
| Virgin Australia | failure | Voluntary administration, 21 Apr 2020 |
| Mosaic Brands | failure | Voluntary administration, 28 Oct 2024 (tested with financial debt only and with all creditor claims) |
| Star Entertainment | near miss | Trading halt then suspension, 28 Feb 2025 |
| Mineral Resources | control | Stressed, no default |
| Woolworths | control | Healthy |

Metrics: first sustained AMBER (DD below 3.5) and RED (below 2.0), lead time
to the event, lowest DD, and for controls the share of weeks spent in
AMBER/RED (false alarms).

Run it: Actions tab > **Backtest** > Run workflow. Results (charts, weekly
CSVs, `summary.md`) are committed to `backtest/results/` and shown on the
run page. Locally: `python run_backtest.py`.

Delisted companies need a price file, see `backtest/prices/README.md`.
Cases without data are skipped and listed in the summary.

Known limits: only the market signal is backtested (Yahoo keeps 4 years of
statements, too short for the accounting signals); the risk-free rate is
one figure per case; debt between balance dates is held flat.

## Roadmap

2. **Announcement NLP.** Pull ASX announcements daily and flag trading
   halts, going-concern language, auditor changes, covenant waivers and
   deeply discounted raisings.
3. **Macro layer.** RBA arrears and credit growth, APRA ADI property
   exposure statistics, ASIC insolvency statistics by industry.
4. **Stress testing.** Shock equity values and rates under RBA-style
   scenarios and show how many names move to RED.
5. **Storage and scheduling.** Move history into DuckDB or Postgres and run
   daily after market close; deploy the dashboard to Render.

Research tool only. Not a credit rating or investment advice.
