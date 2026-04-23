"""
pipeline/features.py
--------------------
Engineers 40+ features used by Wall Street quant teams:
  - Price momentum (multiple horizons)
  - Technical indicators (RSI, MACD, Bollinger, ATR)
  - Volume signals
  - Volatility regime
  - Macro environment (VIX, yield, DXY, Gold)
  - Calendar effects
  - Mean reversion signals
  - Sector relative strength
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (SHORT_MA, LONG_MA, RSI_PERIOD, ATR_PERIOD,
                    BOLLINGER_PERIOD, BOLLINGER_STD, TARGET_HORIZON_DAYS,
                    LOOK_BACK_DAYS)
from logger import get_logger

log = get_logger(__name__)


# ── Core technical calculations ─────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high, low, close, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def _bollinger(close: pd.Series, period=20, n_std=2.0):
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    return ma + n_std * std, ma - n_std * std, ma


def _williams_r(high, low, close, period=14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _mfi(high, low, close, volume, period=14) -> pd.Series:
    tp  = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(), 0).rolling(period).sum()
    neg = rmf.where(tp < tp.shift(), 0).rolling(period).sum()
    mfr = pos / neg.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


# ── Per-ticker feature builder ──────────────────────────────────

def build_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features for a single ticker's price DataFrame.
    Input must have: Date, Open, High, Low, Close, Volume
    """
    df = df.sort_values("Date").copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # ── Returns (momentum) ──────────────────────────────────────
    df["ret_1d"]  = c.pct_change(1)
    df["ret_5d"]  = c.pct_change(5)
    df["ret_10d"] = c.pct_change(10)
    df["ret_21d"] = c.pct_change(21)
    df["ret_63d"] = c.pct_change(63)
    df["ret_126d"]= c.pct_change(126)
    df["ret_252d"]= c.pct_change(252)

    # ── Moving averages ─────────────────────────────────────────
    df["ma_20"]  = c.rolling(SHORT_MA).mean()
    df["ma_50"]  = c.rolling(LONG_MA).mean()
    df["ma_200"] = c.rolling(200).mean()

    # Price relative to MAs
    df["price_to_ma20"]  = c / df["ma_20"] - 1
    df["price_to_ma50"]  = c / df["ma_50"] - 1
    df["price_to_ma200"] = c / df["ma_200"] - 1

    # MA crossover signal
    df["ma_cross_20_50"]  = (df["ma_20"] > df["ma_50"]).astype(int)
    df["ma_cross_50_200"] = (df["ma_50"] > df["ma_200"]).astype(int)

    # ── RSI ─────────────────────────────────────────────────────
    df["rsi_14"]   = _rsi(c, 14)
    df["rsi_28"]   = _rsi(c, 28)
    df["rsi_overbought"]  = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"]    = (df["rsi_14"] < 30).astype(int)

    # ── MACD ────────────────────────────────────────────────────
    df["macd"], df["macd_signal"] = _macd(c)
    df["macd_hist"]  = df["macd"] - df["macd_signal"]
    df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)

    # ── Bollinger Bands ─────────────────────────────────────────
    bb_up, bb_lo, bb_ma = _bollinger(c, BOLLINGER_PERIOD, BOLLINGER_STD)
    df["bb_upper"]  = bb_up
    df["bb_lower"]  = bb_lo
    df["bb_width"]  = (bb_up - bb_lo) / bb_ma   # band width = volatility proxy
    df["bb_pct"]    = (c - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)  # 0=bottom,1=top

    # ── ATR — volatility ────────────────────────────────────────
    df["atr"]        = _atr(h, l, c, ATR_PERIOD)
    df["atr_pct"]    = df["atr"] / c               # normalised ATR
    df["vol_regime"] = (df["atr_pct"] > df["atr_pct"].rolling(63).mean()).astype(int)

    # ── Williams %R ─────────────────────────────────────────────
    df["williams_r"] = _williams_r(h, l, c, 14)

    # ── Volume signals ───────────────────────────────────────────
    df["vol_ma20"]       = v.rolling(20).mean()
    df["volume_ratio"]   = v / df["vol_ma20"].replace(0, np.nan)  # surge = high ratio
    df["obv"]            = _obv(c, v)
    df["obv_ma20"]       = df["obv"].rolling(20).mean()
    df["obv_trend"]      = (df["obv"] > df["obv_ma20"]).astype(int)
    df["mfi"]            = _mfi(h, l, c, v, 14)

    # ── Realised volatility ──────────────────────────────────────
    log_ret = np.log(c / c.shift(1))
    df["realised_vol_21d"]  = log_ret.rolling(21).std()  * np.sqrt(252)
    df["realised_vol_63d"]  = log_ret.rolling(63).std()  * np.sqrt(252)
    df["vol_of_vol"]        = df["realised_vol_21d"].rolling(21).std()

    # ── Mean reversion ───────────────────────────────────────────
    df["z_score_20d"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    df["z_score_63d"] = (c - c.rolling(63).mean()) / c.rolling(63).std()

    # ── Calendar effects ─────────────────────────────────────────
    df["day_of_week"]   = df["Date"].dt.dayofweek
    df["month"]         = df["Date"].dt.month
    df["quarter"]       = df["Date"].dt.quarter
    df["is_month_end"]  = df["Date"].dt.is_month_end.astype(int)
    df["is_quarter_end"]= df["Date"].dt.is_quarter_end.astype(int)

    # ── Gap (overnight return) ───────────────────────────────────
    df["overnight_gap"] = (df["Open"] / c.shift(1)) - 1

    # ── Intraday range ───────────────────────────────────────────
    df["intraday_range"] = (h - l) / c

    # ── TARGET: forward 5-day return ────────────────────────────
    df["target_5d_return"] = c.shift(-TARGET_HORIZON_DAYS) / c - 1

    return df


def attach_macro(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro indicators onto the per-ticker price DataFrame."""
    if macro.empty:
        log.warning("No macro data — skipping macro join")
        for col in ["vix","sp500","treasury_10y","gold","dxy"]:
            prices[col] = np.nan
        return prices

    macro["Date"] = pd.to_datetime(macro["Date"]).dt.normalize()
    macro = macro.sort_values("Date")

    prices = prices.sort_values("Date")
    prices = prices.merge(macro, on="Date", how="left")

    # Forward-fill macro (macro data has gaps on weekends)
    macro_cols = [c for c in macro.columns if c != "Date"]
    prices[macro_cols] = prices[macro_cols].ffill()

    # Derived macro features
    if "vix" in prices.columns:
        prices["vix_regime"]  = (prices["vix"] > 20).astype(int)  # high fear
        prices["vix_ma10"]    = prices["vix"].rolling(10).mean()
        prices["vix_spike"]   = (prices["vix"] > prices["vix_ma10"] * 1.2).astype(int)

    if "treasury_10y" in prices.columns:
        prices["yield_change_5d"] = prices["treasury_10y"].diff(5)
        prices["yield_regime"]    = (prices["treasury_10y"] > 3.0).astype(int)

    if "dxy" in prices.columns:
        prices["dxy_momentum"] = prices["dxy"].pct_change(10)

    if "sp500" in prices.columns:
        prices["mkt_ret_5d"]  = prices["sp500"].pct_change(5)
        prices["mkt_ret_21d"] = prices["sp500"].pct_change(21)

    log.info("Macro features attached: %d columns added", len(macro_cols))
    return prices


def compute_sector_relative_strength(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each stock's 21-day return relative to the median of all stocks
    on the same day — a cross-sectional momentum signal.
    """
    ret21 = prices.pivot_table(
        index="Date", columns="Ticker", values="ret_21d"
    )
    median_ret = ret21.median(axis=1)
    rel_strength = ret21.subtract(median_ret, axis=0)

    rel_long = rel_strength.stack().reset_index()
    rel_long.columns = ["Date", "Ticker", "rel_strength_21d"]

    prices = prices.merge(rel_long, on=["Date", "Ticker"], how="left")
    log.info("Relative strength computed")
    return prices


def build_full_feature_set(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Master feature pipeline:
    1. Per-ticker technical features
    2. Macro join
    3. Sector relative strength
    4. Drop NaN rows needed for model training
    """
    log.info("Building features for %d tickers...", prices["Ticker"].nunique())

    ticker_frames = []
    for ticker, grp in prices.groupby("Ticker"):
        feats = build_ticker_features(grp)
        ticker_frames.append(feats)

    all_feats = pd.concat(ticker_frames, ignore_index=True)
    log.info("Technical features built: %d rows", len(all_feats))

    all_feats = attach_macro(all_feats, macro)
    all_feats = compute_sector_relative_strength(all_feats)

    # Drop rows where target is NaN (last 5 days per ticker)
    before = len(all_feats)
    all_feats = all_feats.dropna(subset=["target_5d_return"])
    log.info("Dropped %d rows with missing target (last %d days per ticker)",
             before - len(all_feats), TARGET_HORIZON_DAYS)

    log.info("Final feature set: %d rows, %d columns", len(all_feats), all_feats.shape[1])
    return all_feats
