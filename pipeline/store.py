"""
pipeline/store.py — All DB operations. Parameterised queries only.
"""
import sqlite3, sys
from contextlib import contextmanager
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from logger import get_logger
log = get_logger(__name__)


@contextmanager
def get_conn(db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA cache_size=-65536;")   # 64MB cache
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    with get_conn(db_path) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL, high REAL, low REAL,
            close REAL NOT NULL CHECK(close > 0),
            volume INTEGER,
            UNIQUE(ticker, date)
        );
        CREATE INDEX IF NOT EXISTS idx_prices_td ON prices(ticker, date);

        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            feature_json TEXT NOT NULL,
            target_5d_return REAL,
            UNIQUE(ticker, date)
        );
        CREATE INDEX IF NOT EXISTS idx_feat_td ON features(ticker, date);

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            predicted_return REAL,
            confidence REAL,
            signal TEXT,
            model_version TEXT,
            UNIQUE(ticker, date)
        );
        CREATE INDEX IF NOT EXISTS idx_pred_td ON predictions(ticker, date);

        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            n_train INTEGER, n_val INTEGER, n_test INTEGER,
            train_r2 REAL, val_r2 REAL, test_r2 REAL,
            test_rmse REAL, test_mae REAL,
            n_features INTEGER,
            top_features TEXT,
            model_version TEXT
        );
        """)
    log.info("DB initialised at %s", db_path)


def upsert_prices(df: pd.DataFrame, db_path=DB_PATH):
    if df.empty: return
    records = [{
        "ticker": str(r["Ticker"]), "date": str(r["Date"])[:10],
        "open": float(r.get("Open") or 0), "high": float(r.get("High") or 0),
        "low":  float(r.get("Low") or 0),  "close": float(r["Close"]),
        "volume": int(r.get("Volume") or 0),
    } for _, r in df.iterrows()]
    sql = """INSERT OR IGNORE INTO prices (ticker,date,open,high,low,close,volume)
             VALUES (:ticker,:date,:open,:high,:low,:close,:volume)"""
    with get_conn(db_path) as c:
        c.executemany(sql, records)
    log.info("Upserted %d price rows", len(records))


def upsert_features(df: pd.DataFrame, db_path=DB_PATH):
    """Store feature rows as JSON blobs — flexible schema."""
    import json
    SKIP = {"Date","Ticker","target_5d_return","Open","High","Low","Close","Volume"}
    records = []
    for _, r in df.iterrows():
        feat_dict = {k: (float(v) if pd.notna(v) else None)
                     for k, v in r.items() if k not in SKIP}
        records.append({
            "ticker": str(r["Ticker"]),
            "date":   str(r["Date"])[:10],
            "feature_json": json.dumps(feat_dict),
            "target": float(r["target_5d_return"]) if pd.notna(r.get("target_5d_return")) else None,
        })
    sql = """INSERT OR IGNORE INTO features (ticker,date,feature_json,target_5d_return)
             VALUES (:ticker,:date,:feature_json,:target)"""
    with get_conn(db_path) as c:
        c.executemany(sql, records)
    log.info("Upserted %d feature rows", len(records))


def upsert_predictions(df: pd.DataFrame, model_version: str, db_path=DB_PATH):
    records = [{
        "ticker":  str(r["Ticker"]),
        "date":    str(r["Date"])[:10],
        "pred":    float(r["predicted_return"]),
        "conf":    float(r.get("confidence") or 0),
        "signal":  str(r["signal"]),
        "version": model_version,
    } for _, r in df.iterrows()]
    sql = """INSERT OR REPLACE INTO predictions
             (ticker,date,predicted_return,confidence,signal,model_version)
             VALUES (:ticker,:date,:pred,:conf,:signal,:version)"""
    with get_conn(db_path) as c:
        c.executemany(sql, records)
    log.info("Upserted %d predictions", len(records))


def save_model_run(stats: dict, db_path=DB_PATH):
    import json
    sql = """INSERT INTO model_runs
             (n_train,n_val,n_test,train_r2,val_r2,test_r2,
              test_rmse,test_mae,n_features,top_features,model_version)
             VALUES (:n_train,:n_val,:n_test,:train_r2,:val_r2,:test_r2,
                     :test_rmse,:test_mae,:n_features,:top_features,:model_version)"""
    with get_conn(db_path) as c:
        c.execute(sql, {**stats, "top_features": json.dumps(stats.get("top_features",[]))})
    log.info("Model run saved")


def load_predictions(db_path=DB_PATH) -> pd.DataFrame:
    with get_conn(db_path) as c:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY date DESC, ticker", c
        )

def load_latest_predictions(db_path=DB_PATH) -> pd.DataFrame:
    with get_conn(db_path) as c:
        return pd.read_sql_query("""
            SELECT p.* FROM predictions p
            INNER JOIN (SELECT ticker, MAX(date) as md FROM predictions GROUP BY ticker) latest
            ON p.ticker=latest.ticker AND p.date=latest.md
            ORDER BY predicted_return DESC
        """, c)

def load_prices_df(ticker: str = None, db_path=DB_PATH) -> pd.DataFrame:
    with get_conn(db_path) as c:
        if ticker:
            return pd.read_sql_query(
                "SELECT * FROM prices WHERE ticker=? ORDER BY date", c, params=(ticker,)
            )
        return pd.read_sql_query("SELECT * FROM prices ORDER BY ticker, date", c)

def load_model_runs(db_path=DB_PATH) -> pd.DataFrame:
    with get_conn(db_path) as c:
        return pd.read_sql_query("SELECT * FROM model_runs ORDER BY run_ts DESC", c)
