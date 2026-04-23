"""
pipeline/ingest.py
Fetches 10 years of daily OHLCV data for 50 tickers + macro indicators.
Expects ~130,000 price rows total.
"""
import re, sys, time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TICKER_ALLOWLIST, START_DATE, END_DATE,
                    FETCH_TIMEOUT, FETCH_RETRIES)
from logger import get_logger

log = get_logger(__name__)
_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

# Macro tickers fetched alongside equities
MACRO_TICKERS = {
    "^VIX":  "vix",           # Market fear gauge
    "^GSPC": "sp500",         # S&P 500 index level
    "^TNX":  "treasury_10y",  # 10-year yield
    "GLD":   "gold",          # Gold ETF
    "DX-Y.NYB": "dxy",        # Dollar index
}


def _validate(ticker: str) -> str:
    t = ticker.strip().upper()
    if not _TICKER_RE.match(t):
        raise ValueError(f"Malformed ticker: {ticker!r}")
    if t not in TICKER_ALLOWLIST:
        raise ValueError(f"Ticker not in allowlist: {t!r}")
    return t


def fetch_ticker(ticker: str, start=START_DATE, end=END_DATE,
                 validate=True) -> pd.DataFrame:
    """Fetch OHLCV for one ticker with retries."""
    if validate:
        ticker = _validate(ticker)
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            df = yf.Ticker(ticker).history(
                start=start, end=end, timeout=FETCH_TIMEOUT, auto_adjust=True
            )
            if df.empty:
                log.warning("Empty data for %s", ticker)
                return pd.DataFrame()
            df = df.reset_index()
            df["Ticker"] = ticker
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
            cols = [c for c in ["Date","Open","High","Low","Close","Volume","Ticker"] if c in df.columns]
            log.info("Fetched %d rows for %s", len(df), ticker)
            return df[cols].copy()
        except Exception as e:
            log.warning("Attempt %d failed for %s: %s", attempt, ticker, e)
            if attempt < FETCH_RETRIES:
                time.sleep(2 ** attempt)
    return pd.DataFrame()


def fetch_macro(start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """
    Fetch macro indicators: VIX, S&P500, 10Y yield, Gold, DXY.
    Returns a wide DataFrame indexed by Date.
    """
    frames = []
    for sym, col in MACRO_TICKERS.items():
        try:
            df = yf.Ticker(sym).history(start=start, end=end,
                                         timeout=FETCH_TIMEOUT, auto_adjust=True)
            if df.empty:
                continue
            df = df.reset_index()[["Date", "Close"]].copy()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
            df = df.rename(columns={"Close": col})
            frames.append(df.set_index("Date"))
            log.info("Fetched macro: %s (%s)", col, sym)
            time.sleep(0.3)
        except Exception as e:
            log.warning("Macro fetch failed for %s: %s", sym, e)

    if not frames:
        log.warning("No macro data fetched — will proceed without it")
        return pd.DataFrame()

    macro = frames[0]
    for f in frames[1:]:
        macro = macro.join(f, how="outer")
    macro = macro.sort_index().ffill().reset_index()
    log.info("Macro dataset: %d rows, %d indicators", len(macro), len(macro.columns)-1)
    return macro


def fetch_all(tickers: Optional[list] = None,
              start=START_DATE, end=END_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch all equity prices + macro data.
    Returns (prices_df, macro_df).
    """
    tickers = tickers or list(TICKER_ALLOWLIST)
    frames = []
    for t in tickers:
        df = fetch_ticker(t, start, end)
        if not df.empty:
            frames.append(df)
        time.sleep(0.4)

    if not frames:
        log.error("No price data fetched — using GBM simulation")
        prices = _simulate_prices(tickers, start, end)
    else:
        prices = pd.concat(frames, ignore_index=True)
        log.info("Total price rows: %d across %d tickers", len(prices), len(frames))

    macro = fetch_macro(start, end)
    return prices, macro


# ── GBM simulation fallback (sandbox / offline) ───────────────
_SEED_PRICE = {
    "AAPL":130,"MSFT":240,"GOOGL":90,"AMZN":85,"META":125,
    "NVDA":145,"AMD":65,"INTC":28,"ORCL":80,"CRM":150,
    "JPM":135,"GS":330,"BAC":32,"MS":85,"WFC":42,
    "BLK":700,"AXP":155,"C":45,"USB":42,"TFC":35,
    "JNJ":160,"UNH":490,"PFE":40,"MRK":80,"ABBV":145,
    "LLY":320,"TMO":530,"ABT":110,"CVS":90,"MDT":82,
    "XOM":80,"CVX":155,"COP":100,"SLB":45,"EOG":120,
    "TSLA":110,"NKE":105,"MCD":265,"SBUX":95,"HD":295,
    "LOW":195,"TGT":145,"COST":495,"WMT":155,"PG":145,
    "NFLX":300,"DIS":95,"CMCSA":38,"T":17,"VZ":36,
}

def _simulate_prices(tickers, start, end) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end)
    frames = []
    for t in tickers:
        seed = _SEED_PRICE.get(t, 100.0)
        rng  = np.random.default_rng(abs(hash(t)) % (2**31))
        rets = rng.normal(0.0003, 0.018, len(dates))
        px   = seed * np.cumprod(1 + rets)
        frames.append(pd.DataFrame({
            "Date": dates, "Ticker": t,
            "Open":  px * rng.uniform(0.998, 1.002, len(dates)),
            "High":  px * rng.uniform(1.000, 1.015, len(dates)),
            "Low":   px * rng.uniform(0.985, 1.000, len(dates)),
            "Close": px,
            "Volume": rng.integers(5_000_000, 80_000_000, len(dates)),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    log.info("Simulated %d price rows for %d tickers", len(df), len(tickers))
    return df
