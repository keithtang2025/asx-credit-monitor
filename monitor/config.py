"""Settings for the ASX credit risk monitor. Edit freely."""

# Universe: S&P/ASX 300 codes (FNArena constituent list, Nov 2026; check
# after each quarterly rebalance in March, June, September and December).
ASX300 = """
360 4DX A2M A4N AAI ABB ABG ACL AEF AEL AGL AIA ALD ALK ALL ALQ ALX AMC AMP ANN
ANZ AOV APA APE APZ ARB ARF ARU ASB ASX AUB AXQ AYA AZJ BC8 BEN BGA BGL BHP BJY
BMN BNZ BOE BOQ BPT BRE BRG BSL BVS BWP BXB CAR CAT CBA CBO CCL CCP CDA CGF CHC
CHN CIA CIP CKF CLW CMM CMW CNI CNU COH COL CPU CQE CQR CSC CSL CU6 CUV CWY CXO
CYL DBI DDR DGT DMP DNL DOW DPM DRO DRR DTL DVP DXI DXS DYL EBO EDV EIQ ELD ELS
ELV EMR EOS EQR EQT EVN EVT FBU FFM FLT FMG FPH FPR GDG GGP GLF GMD GMG GNC GNG
GNP GOZ GPT GQG GWA GYG HDN HLI HMC HSN HUB HVN IAG IDX IEL IFT IGO ILU IMD IMR
INA ING IPH IPX IRE JBH JDO JHX KAR KCN KLS L1G LIC LIN LLC LNW LOV LTR LYC MAF
MAH MAQ MGH MGR MI6 MIN MLX MMS MND MP1 MPL MQG MSB MTS MYS NAB NAN NCK NEC NEM
NEU NGI NHC NHF NIC NST NUF NWH NWL NWS NXG NXT OBM OML ORA ORG ORI PDI PDN PLS
PME PMT PMV PNI PNR PNV PPT PRN PRU PWH PXA PYC QAN QBE QRI RDX REA REG REH RFF
RGN RHC RIC RIO RMD RMS RPL RRL RSG RWC S32 SBM SCG SDF SDR SEK SFR SGH SGLLV SGM
SGP SHL SHV SIG SIQ SKG SKS SLC SLX SMI SMR SNL SOL SPK SRG SRL SSM STO SUL SUN
SX2 SXE TAH TCG TCL TEA TLC TLS TLX TNE TPG TPW TUA TWE TYR UNI VAU VCX VEA VGN
VNT VSL VUL WA1 WAF WBC WBT WDS WEB WES WGX WHC WOR WOW WPR WTC XRO XYZ YAL ZIP
""".split()

# Financial institutions excluded on purpose: deposits, policy liabilities,
# client money and lending books break the Merton "debt" assumption and
# would create false alarms. Banks, insurers, lenders/BNPL, payments,
# exchanges, fund and wealth managers, credit funds.
EXCLUDED_FINANCIALS = set("""
ANZ BEN BOQ CBA MYS NAB WBC JDO CCL
AUB GDG HLI IAG MPL NHF QBE SDF SUN
CCP FPR TYR XYZ ZIP QRI
AEF AMP ASX BJY CGF CNI EQT GQG HMC HUB L1G MAF MQG NGI NWL PNI PPT RPL
""".split())

# Belt and braces: anything Yahoo classifies under these sectors is skipped
# at run time too, in case the list above misses one.
EXCLUDED_SECTORS = {"Financial Services"}

# Names kept even though they are no longer in the ASX 300
EXTRA_TICKERS = ["SGR"]   # Star Entertainment: the backtest's near-miss case

WATCHLIST = [f"{c}.AX" for c in ASX300 if c not in EXCLUDED_FINANCIALS] + \
            [f"{c}.AX" for c in EXTRA_TICKERS if c not in ASX300]

# Pause between companies (seconds) so Yahoo does not rate-limit the run
FETCH_PAUSE = 0.4

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

# --- v8: the right credit metric for each business model ---
# Property and long-life infrastructure are judged on GEARING (debt / total
# assets), as lenders do: their earnings run through revaluations,
# development profits and equity-accounted stakes that EBITDA misses.
# Thresholds are (red_above, amber_above).
GEARING_BY_SECTOR = [
    (["reit", "real estate"], (0.45, 0.35)),
]
GEARING_TICKER_OVERRIDES = {
    # toll roads, ports, airports: structurally higher gearing
    "TCL.AX": (0.65, 0.55), "ALX.AX": (0.65, 0.55),
    "DBI.AX": (0.65, 0.55), "AIA.AX": (0.65, 0.55),
    "IFT.AX": (0.65, 0.55),   # Infratil: infrastructure investor, equity-accounted stakes
}

# Loss-making companies (EBITDA <= 0) are judged on CASH RUNWAY:
# cash / (annual cash burn + interest), in months. (red_below, amber_below)
RUNWAY_MONTHS = (12, 24)

# Altman Z'' red on its own only gives WATCH; it needs a second signal to
# raise an AMBER. It is switched off entirely for net-cash companies.
Z_ALONE_STATUS = "WATCH"

# v9: when a hard accounting red appears but the market prices little risk
# (distance to default at or above this), add a "check data" note.
MARKET_DISAGREES_DD = 4.0

# --- v6: data sources ---
# Second source (FMP) for cross-checking. Off: the monitor runs on Yahoo only.
# Set True and add an FMP_API_KEY secret to switch it back on.
USE_SECOND_SOURCE = False

# Operating income vs Yahoo "EBIT": a gap larger than this (and > 5m) means
# EBIT contains non-operating items such as impairments below the line.
EBIT_GAP_TOLERANCE = 0.10

# --- v5: earnings through the cycle ---
# Sectors where interest cover is judged on the 3-year average EBITDA, so a
# single commodity downturn year does not trigger alerts by itself.
CYCLICAL_SECTORS = {"Basic Materials", "Energy"}
# v9: agribusiness swings with harvests and seasons (Cobram's olive crop
# alternates big and small years), so it is judged through the cycle too.
CYCLICAL_INDUSTRY_KEYWORDS = ["farm products", "agricultural"]
CYCLICAL_TICKERS = {"GNC.AX", "CBO.AX", "NUF.AX", "ELD.AX", "SHV.AX"}
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
                     "APA.AX", "AZJ.AX", "AGL.AX", "ORG.AX", "SGP.AX", "VGN.AX",
                     "MTS.AX", "EDV.AX", "ALX.AX", "DBI.AX", "AIA.AX", "SIG.AX"}
# Same idea for interest-cover thresholds when Yahoo's label misses
COVERAGE_TICKER_OVERRIDES = {
    "TCL.AX": (1.5, 2.5), "APA.AX": (1.5, 2.5), "AZJ.AX": (1.5, 2.5),
    "ALX.AX": (1.5, 2.5), "DBI.AX": (1.5, 2.5), "AIA.AX": (1.5, 2.5),
}

# Balance sheets older than this are flagged as stale (days)
STALE_STATEMENT_DAYS = 456   # about 15 months

# Drawdown from 52-week high that triggers an AMBER market-stress flag
DRAWDOWN_ALERT = 0.40
