"""Backtest cases: did distance to default warn before real credit events?

Three groups:
  failure   - the company entered administration (true positives wanted)
  near_miss - severe distress but no default (an early warning is still right)
  control   - survived without a credit event (false alarms are measured)

Data rules
- Prices: Yahoo first. Delisted companies are usually missing from Yahoo,
  so put a CSV at backtest/prices/<CODE>.csv with columns Date,Close
  (optionally Shares or MarketCap). EODHD's one-month plan includes
  delisted data; your broker's charting export also works.
- Shares: Yahoo share-count history when available, else `shares` below.
- Debt: Yahoo annual balance sheets when they cover the window, else the
  `debt` schedule below. Each balance-sheet figure is only "known" after a
  reporting lag (default 60 days) to avoid look-ahead bias.
- Figures marked VERIFY are best available estimates; check them against
  the annual reports and edit freely.
"""

CASES = [
    {
        "name": "Virgin Australia",
        "ticker": "VAH.AX",
        "group": "failure",
        "event_date": "2020-04-21",
        "event": "Voluntary administration (Deloitte appointed)",
        "lookback_years": 3,
        "risk_free": 0.015,
        # Shares on issue, fully diluted, ~8,442m (VERIFY vs annual report)
        "shares": 8_442e6,
        # Debt schedule: (balance date, short-term debt, long-term debt), AUD.
        # Total borrowings ~A$5bn at 31 Dec 2019 per media reports at the
        # time of administration. ST/LT split is an ASSUMPTION (VERIFY).
        "debt": [
            ("2019-12-31", 1.0e9, 4.0e9),
        ],
        "notes": "Free float under 10% (mostly airline shareholders), so the "
                 "share price may react slowly. Shares suspended 16 Apr 2020.",
    },
    {
        "name": "Mosaic Brands",
        "ticker": "MOZ.AX",
        "group": "failure",
        "event_date": "2024-10-28",
        "event": "Voluntary administration and receivers appointed",
        "lookback_years": 4,
        "risk_free": 0.04,
        "shares": None,            # FILL IN from the annual report, or supply MarketCap in the CSV
        # Two definitions of "debt" to show why the default point matters:
        # financial debt only (senior lender ~36m + convertible notes ~18m)
        # versus all creditor claims tallied after failure (~318m).
        "debt": [("2024-06-30", 36e6, 18e6)],
        "debt_alt": [("2024-06-30", 200e6, 236e6)],   # ~318m default point (ST + 0.5 LT)
        "alt_label": "all creditor claims",
        "notes": "Administrators later said it was likely insolvent from "
                 "31 Dec 2020, almost four years before administration. "
                 "Lease and supplier obligations dwarfed bank debt.",
    },
    {
        "name": "Star Entertainment",
        "ticker": "SGR.AX",
        "group": "near_miss",
        "event_date": "2025-02-28",
        "event": "Trading halt, then suspension (going-concern uncertainty)",
        "lookback_years": 3,
        "risk_free": 0.04,
        "shares": None,            # Yahoo share history (equity raisings matter here)
        "debt": None,              # Yahoo annual balance sheets
        "notes": "Rescued rather than liquidated, so it counts as distress "
                 "the model should have flagged, not a default.",
    },
    {
        "name": "Mineral Resources",
        "ticker": "MIN.AX",
        "group": "control",
        "event_date": "2025-06-30",
        "event": "End of window (lithium downturn, high leverage, no default)",
        "lookback_years": 3,
        "risk_free": 0.04,
        "shares": None,
        "debt": None,
        "notes": "Stressed but survived: measures how loudly the model "
                 "warns on a company that did not default.",
    },
    {
        "name": "Woolworths",
        "ticker": "WOW.AX",
        "group": "control",
        "event_date": "2025-06-30",
        "event": "End of window (healthy control)",
        "lookback_years": 3,
        "risk_free": 0.04,
        "shares": None,
        "debt": None,
        "notes": "Should stay GREEN throughout.",
    },
]

# Backtest settings
STEP_DAYS = 5              # recompute weekly
WINDOW_DAYS = 252          # trailing volatility window
REPORTING_LAG_DAYS = 60    # balance sheet becomes public ~2 months after year end
CONFIRM_OBS = 2            # a breach must last 2 consecutive weeks to count
