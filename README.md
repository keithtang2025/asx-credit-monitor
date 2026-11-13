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

## How it works

| Signal | Method | RED | AMBER |
|---|---|---|---|
| Distance to default | KMV iteration on 1 year of daily market cap; default point = ST debt + 0.5 x LT debt; risk-neutral drift | < 2.0 | < 3.5 |
| Altman Z'' | 6.56 WC/TA + 3.26 RE/TA + 6.72 EBIT/TA + 1.05 BE/TL | < 1.1 | < 2.6 |
| Interest coverage | EBIT / interest expense | < 1.5x | < 3.0x |
| Market stress | Drawdown from 52-week high | | >= 40% |

Thresholds live in `monitor/config.py`. Treat them as a starting risk
appetite and recalibrate after backtesting.

## Project layout

```
monitor/config.py   watchlist, risk-free rate, thresholds
monitor/data.py     yfinance fetch, statement field mapping, FX conversion
monitor/models.py   Merton/KMV, Altman Z'', coverage, RAG logic
monitor/engine.py   scores one company and the whole watchlist
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

## Roadmap

1. **Backtest.** Run rolling DD on past ASX failures and near-failures
   (e.g. Virgin Australia 2020, Mosaic Brands 2024) and measure lead time.
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
