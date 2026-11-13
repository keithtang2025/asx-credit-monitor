# Distance-to-default backtest

Thresholds: RED below 2.0, AMBER below 3.5. A breach counts only if it lasts 2 consecutive weekly observations. Balance sheets are used 60 days after their date to avoid look-ahead.

## Failures and near misses (did it warn, and how early?)

| Case | Event | First AMBER | Lead | First RED | Lead | Lowest DD |
|---|---|---|---|---|---|---|
| Star Entertainment | 28 Feb 2025 | 16 Feb 2023 | 24 months | 30 Aug 2023 | 18 months | 0.89 |

## Controls (how often did it cry wolf?)

| Case | Weeks AMBER or worse | Weeks RED | Lowest DD | Lowest on |
|---|---|---|---|---|
| Mineral Resources | 23% | 9% | 1.40 | 22 Apr 2025 |
| Woolworths | 0% | 0% | 8.52 | 23 Sep 2022 |

## Look-ahead caveat

- Star Entertainment: 16% of observations used a balance sheet that was not yet public (no earlier figures supplied). Add earlier debt figures to remove this.
- Mineral Resources: 6% of observations used a balance sheet that was not yet public (no earlier figures supplied). Add earlier debt figures to remove this.
- Woolworths: 39% of observations used a balance sheet that was not yet public (no earlier figures supplied). Add earlier debt figures to remove this.

## Skipped

- Virgin Australia: No price data for VAH.AX. Yahoo has no history for this window (common for delisted companies). Add backtest/prices/VAH.csv with columns Date,Close[,Shares|MarketCap].
- Mosaic Brands: No price data for MOZ.AX. Yahoo has no history for this window (common for delisted companies). Add backtest/prices/MOZ.csv with columns Date,Close[,Shares|MarketCap].

Charts and weekly data: one PNG and CSV per case in this folder.