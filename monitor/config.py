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
    "TCL.AX", "APA.AX", "AZJ.AX", "QUB.AX", "BXB.AX", "AMC.AX", "ORA.AX",
    "JHX.AX", "BSL.AX", "SVW.AX", "LLC.AX", "AGL.AX", "ORG.AX", "TLS.AX",
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
    "ebitda_coverage": (2.0, 4.0),     # EBITDA / interest expense (primary)
}

# Status rule (v2):
#   RED   = DD below its red line, OR at least RED_SIGNALS_NEEDED signals RED
#   AMBER = any single signal RED or AMBER, or the drawdown flag
#   GREEN = otherwise
RED_SIGNALS_NEEDED = 2

# Altman Z'' is switched off for business models it misreads:
# customer-funded working capital, concession/regulated assets, property trusts.
# Matched case-insensitively against the Yahoo "industry" and "sector" text.
Z_EXCLUDE_KEYWORDS = [
    "grocery", "discount stores", "airline", "travel", "infrastructure",
    "toll", "midstream", "utilities", "reit", "real estate",
]
# Ticker-level overrides (in case Yahoo's industry label misses one)
Z_EXCLUDE_TICKERS = {"WOW.AX", "COL.AX", "QAN.AX", "FLT.AX", "WEB.AX", "TCL.AX",
                     "APA.AX", "AZJ.AX", "QUB.AX", "AGL.AX", "ORG.AX", "SGP.AX"}

# Balance sheets older than this are flagged as stale (days)
STALE_STATEMENT_DAYS = 456   # about 15 months

# Drawdown from 52-week high that triggers an AMBER market-stress flag
DRAWDOWN_ALERT = 0.40
