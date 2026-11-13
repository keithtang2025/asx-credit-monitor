# Price files for delisted companies

Yahoo usually has no history for delisted companies (Virgin Australia,
Mosaic Brands). Put a CSV here named after the ASX code, e.g. `VAH.csv`:

```
Date,Close
2016-01-04,0.465
2016-01-05,0.460
...
```

Optional extra columns:
- `Shares`: shares on issue on that date (handles capital raisings)
- `MarketCap`: market capitalisation in AUD (then Close and Shares are ignored)

Dates as YYYY-MM-DD, one row per trading day. Cover from about 18 months
before the backtest window starts, so the first volatility estimate has a
full year of data.

Where to get it: EODHD's one-month EOD plan includes delisted tickers;
some broker platforms (e.g. CommSec, Bell Direct) let you export price
history; or your old Bloomberg/FactSet access if you still have it.
