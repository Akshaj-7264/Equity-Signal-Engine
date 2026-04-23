"""
config.py — Single source of truth for the Quant Research Platform.
"""
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR    = BASE_DIR / "logs"
MODEL_DIR  = BASE_DIR / "models"

for _d in (DATA_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DB_PATH       = DATA_DIR / "quant.db"
MODEL_PATH    = MODEL_DIR / "xgb_model.pkl"
SCALER_PATH   = MODEL_DIR / "scaler.pkl"
SIGNALS_CSV   = OUTPUT_DIR / "signals.csv"

# ── Universe — 50 liquid US equities across sectors ───────────
TICKERS = [
    # Tech
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","AMD","INTC","ORCL","CRM",
    # Finance
    "JPM","GS","BAC","MS","WFC","BLK","AXP","C","USB","TFC",
    # Healthcare
    "JNJ","UNH","PFE","MRK","ABBV","LLY","TMO","ABT","CVS","MDT",
    # Energy
    "XOM","CVX","COP","SLB","EOG",
    # Consumer
    "TSLA","NKE","MCD","SBUX","HD","LOW","TGT","COST","WMT","PG",
    # Media/Telecom
    "NFLX","DIS","CMCSA","T","VZ",
]
TICKER_ALLOWLIST = set(TICKERS)

# ── Date range — 10 years of data ─────────────────────────────
START_DATE = "2014-01-01"
END_DATE   = "2024-12-31"

# ── Feature engineering params ─────────────────────────────────
LOOK_BACK_DAYS    = 252        # 1 trading year for rolling features
SHORT_MA          = 20
LONG_MA           = 50
RSI_PERIOD        = 14
ATR_PERIOD        = 14
BOLLINGER_PERIOD  = 20
BOLLINGER_STD     = 2.0

# ── Model params ───────────────────────────────────────────────
TARGET_HORIZON_DAYS = 5        # predict 5-day forward return
TRAIN_RATIO         = 0.75
VAL_RATIO           = 0.10
TEST_RATIO          = 0.15
SIGNIFICANCE_LEVEL  = 0.05

# ── Signal thresholds ──────────────────────────────────────────
STRONG_BUY_THRESHOLD  =  3.0   # predicted 5d return > 3%
BUY_THRESHOLD         =  1.5
SELL_THRESHOLD        = -1.5
STRONG_SELL_THRESHOLD = -3.0

# ── Fetch settings ─────────────────────────────────────────────
FETCH_TIMEOUT  = 30
FETCH_RETRIES  = 3

LOG_FILE  = LOG_DIR / "platform.log"
LOG_LEVEL = "INFO"
