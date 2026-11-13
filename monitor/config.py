"""Settings for the ASX credit risk monitor. Edit freely."""

# Non-financial ASX names. Banks and insurers are excluded on purpose:
# deposits and policy liabilities break the Merton "debt" assumption.
WATCHLIST = [
    # Resources and energy
    "BHP.AX", "RIO.AX", "FMG.AX", "WDS.AX", "STO.AX", "S32.AX", "MIN.AX",
    "PLS.AX", "IGO.AX", "LYC.AX", "NIC.AX", "WHC.AX", "NST.AX", "EVN.AX",
    # Consumer and gaming
    "WES.AX", "WOW.AX", "COL.AX", "JBH.AX", "HVN.AX", "TWE.AX", "A2M.AX",
    "SGR.AX", "TAH.AX", "QAN.AX", "FLT.AX", "WEB.AX",
    # Industrials, infrastructure, utilities
    "TCL.AX", "APA.AX", "AZJ.AX", "BXB.AX", "AMC.AX", "ORA.AX",
    "JHX.AX", "BSL.AX", "SGH.AX", "LLC.AX", "AGL.AX", "ORG.AX", "TLS.AX",
    # Healthcare and tech
    "CSL.AX", "RMD.AX", "SHL.AX", "RHC.AX", "NXT.AX", "XRO.AX", "WTC.AX",
    # Property (A-REITs)
    "GMG.AX", "SCG.AX", "SGP.AX", "MGR.AX", "VCX.AX", "DXS.AX", "CHC.AX",
]

# Annual risk-free rate used in Merton (set close to the current RBA cash
# rate or the 1-year government bond yield). Update before each run.
RISK_FREE_RATE = 0.04

# Merton horizon in years
HORIZON = 1.0

# Price history used for equity volatility (trading days)
VOL_WINDOW = 252

# Alert thresholds: (red, amber). A metric below red is RED, below amber is AMBER.
THRESHOLDS = {
    "distance_to_default": (2.0, 3.5),
    "altman_z2": (1.1, 2.6),           # Altman Z'' distress / grey zone cut-offs
    "ebitda_coverage": (2.0, 4.0),     # normalised EBITDA / interest (default)
}

# Interest cover thresholds by business model (red, amber). Matched against
# Yahoo industry/sector text; first match wins, otherwise the default above.
# Property trusts and regulated/concession assets carry more debt by design;
# their bank covenants typically sit around 1.5x to 2.0x.
COVERAGE_BY_SECTOR = [
    (["reit", "real estate"], (1.75, 2.75)),
    (["infrastructure", "toll", "midstream", "utilities"], (1.5, 2.5)),
]

# Status rule (v3):
#   RED   = DD below its red line, OR at least RED_SIGNALS_NEEDED independent reds
#   AMBER = DD amber, OR one accounting signal red, OR two accounting signals
#           amber, OR drawdown together with any accounting warning
#   WATCH = a single accounting signal amber, OR drawdown on its own
#   GREEN = otherwise
RED_SIGNALS_NEEDED = 2

# Unusual items (impairments, restructuring) larger than this share of
# normalised EBITDA mark the year as "one-off affected". In that case Z'' and
# interest cover count as ONE signal, because they share the same cause.
ONE_OFF_SHARE = 0.25

# --- v5: earnings through the cycle ---
# Sectors where interest cover is judged on the 3-year average EBITDA, so a
# single commodity downturn year does not trigger alerts by itself.
CYCLICAL_SECTORS = {"Basic Materials", "Energy"}
CYCLICAL_TICKERS = set()          # add codes here to force 3-year treatment
MIN_YEARS_FOR_AVERAGE = 2         # need at least this many years for an average

# "Earnings falling" flag: latest normalised EBITDA below this share of the
# 3-year average (applies to every company, cyclical or not).
EARNINGS_FALL_RATIO = 0.60

# Flag Yahoo's EBITDA field when it differs from EBIT + D&A by more than this
EBITDA_FIELD_TOLERANCE = 0.10

# Old or renamed ASX codes. Delisted names should be removed from WATCHLIST.
TICKER_ALIASES = {
    "SVW.AX": "SGH.AX",   # Seven Group Holdings renamed SGH Ltd, Nov 2024
}

# Altman Z'' is switched off for business models it misreads:
# customer-funded working capital, concession/regulated assets, property trusts.
# Matched case-insensitively against the Yahoo "industry" and "sector" text.
Z_EXCLUDE_KEYWORDS = [
    "grocery", "discount stores", "airline", "travel", "infrastructure",
    "toll", "midstream", "utilities", "reit", "real estate",
]
# Ticker-level overrides (in case Yahoo's industry label misses one)
Z_EXCLUDE_TICKERS = {"WOW.AX", "COL.AX", "QAN.AX", "FLT.AX", "WEB.AX", "TCL.AX",
                     "APA.AX", "AZJ.AX", "AGL.AX", "ORG.AX", "SGP.AX"}
# Same idea for interest-cover thresholds when Yahoo's label misses
COVERAGE_TICKER_OVERRIDES = {
    "TCL.AX": (1.5, 2.5), "APA.AX": (1.5, 2.5), "AZJ.AX": (1.5, 2.5),
}

# Balance sheets older than this are flagged as stale (days)
STALE_STATEMENT_DAYS = 456   # about 15 months

# Drawdown from 52-week high that triggers an AMBER market-stress flag
DRAWDOWN_ALERT = 0.40
