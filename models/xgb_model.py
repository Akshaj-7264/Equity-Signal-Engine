"""
models/xgb_model.py
-------------------
XGBoost regressor predicting 5-day forward return.

Real-world features used:
  - Multi-horizon price momentum
  - RSI, MACD, Bollinger, ATR, Williams %R
  - Volume signals (OBV, MFI, volume ratio)
  - Realised volatility + vol-of-vol
  - Mean reversion z-scores
  - Macro: VIX regime, yield curve, DXY, market return
  - Calendar effects
  - Relative strength vs. universe

Train/Val/Test split is TIME-BASED (no lookahead leakage).
"""
import sys, pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (MODEL_PATH, SCALER_PATH, TRAIN_RATIO, VAL_RATIO,
                    SIGNIFICANCE_LEVEL, STRONG_BUY_THRESHOLD,
                    BUY_THRESHOLD, SELL_THRESHOLD, STRONG_SELL_THRESHOLD)
from logger import get_logger

log = get_logger(__name__)

MODEL_VERSION = f"xgb_v1_{datetime.now().strftime('%Y%m%d')}"

# Features used for training (subset of engineered columns)
FEATURE_COLS = [
    # Momentum
    "ret_1d","ret_5d","ret_10d","ret_21d","ret_63d","ret_126d","ret_252d",
    # MA signals
    "price_to_ma20","price_to_ma50","price_to_ma200",
    "ma_cross_20_50","ma_cross_50_200",
    # Oscillators
    "rsi_14","rsi_28","rsi_overbought","rsi_oversold",
    "macd_hist","macd_cross",
    # Bollinger
    "bb_width","bb_pct",
    # Volatility
    "atr_pct","vol_regime",
    "realised_vol_21d","realised_vol_63d","vol_of_vol",
    # Mean reversion
    "z_score_20d","z_score_63d",
    # Volume
    "volume_ratio","obv_trend","mfi","williams_r",
    # Macro
    "vix","vix_regime","vix_spike",
    "treasury_10y","yield_change_5d","yield_regime",
    "dxy_momentum","mkt_ret_5d","mkt_ret_21d",
    # Calendar
    "day_of_week","month","quarter",
    "is_month_end","is_quarter_end",
    # Cross-sectional
    "rel_strength_21d",
    # Intraday
    "overnight_gap","intraday_range",
]


def _time_split(df: pd.DataFrame):
    """
    Strict chronological train/val/test split — NO random shuffling.
    This is mandatory in finance to prevent lookahead leakage.
    """
    df = df.sort_values("Date")
    dates = df["Date"].unique()
    n = len(dates)
    t_end = int(n * TRAIN_RATIO)
    v_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_dates = dates[:t_end]
    val_dates   = dates[t_end:v_end]
    test_dates  = dates[v_end:]

    train = df[df["Date"].isin(train_dates)]
    val   = df[df["Date"].isin(val_dates)]
    test  = df[df["Date"].isin(test_dates)]

    log.info("Split — Train: %d | Val: %d | Test: %d",
             len(train), len(val), len(test))
    return train, val, test


def _prepare(df: pd.DataFrame, scaler=None, fit_scaler=False):
    """Extract, clean, and scale feature matrix. Always 48 columns."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = set(FEATURE_COLS) - set(available)
    if missing:
        log.warning("Missing features (will fill 0): %s", missing)

    # Build X with FEATURE_COLS order — add zero columns for missing
    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        if col in df.columns:
            X[col] = df[col].values
        else:
            X[col] = 0.0

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    y = df["target_5d_return"].values

    if fit_scaler:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X) if scaler else X.values

    return X_scaled, y, scaler, FEATURE_COLS


def train(df: pd.DataFrame) -> dict:
    """
    Full training pipeline.
    Returns stats dict and saves model + scaler to disk.
    """
    log.info("Starting XGBoost training on %d rows...", len(df))

    # Drop rows with NaN target
    df = df.dropna(subset=["target_5d_return"]).copy()

    train_df, val_df, test_df = _time_split(df)

    X_train, y_train, scaler, feat_cols = _prepare(train_df, fit_scaler=True)
    X_val,   y_val,   _,      _        = _prepare(val_df,   scaler=scaler)
    X_test,  y_test,  _,      _        = _prepare(test_df,  scaler=scaler)

    model = XGBRegressor(
        n_estimators      = 800,
        max_depth         = 5,
        learning_rate     = 0.03,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 10,     # prevents overfitting on small leaf nodes
        reg_alpha         = 0.1,    # L1 regularisation
        reg_lambda        = 1.0,    # L2 regularisation
        early_stopping_rounds = 30,
        eval_metric       = "rmse",
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Metrics
    def metrics(X, y, label):
        pred = model.predict(X)
        r2   = r2_score(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        mae  = mean_absolute_error(y, pred)
        log.info("%s — R2: %.4f | RMSE: %.4f | MAE: %.4f", label, r2, rmse, mae)
        return r2, rmse, mae, pred

    train_r2, _, _, _              = metrics(X_train, y_train, "TRAIN")
    val_r2,   _, _, _              = metrics(X_val,   y_val,   "VAL  ")
    test_r2, test_rmse, test_mae, test_preds = metrics(X_test, y_test, "TEST ")

    # Feature importance — X was built with FEATURE_COLS order (available + zeros for missing)
    # Use the full FEATURE_COLS list since _prepare adds zero columns for missing ones
    n_imp = len(model.feature_importances_)
    imp = pd.DataFrame({
        "feature":    FEATURE_COLS[:n_imp],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    top = imp.head(15)
    log.info("Top 10 features:\n%s", top.head(10).to_string(index=False))

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump((scaler, feat_cols), f)
    log.info("Model saved to %s", MODEL_PATH)

    stats = {
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "train_r2": round(train_r2, 4), "val_r2": round(val_r2, 4),
        "test_r2":  round(test_r2,  4), "test_rmse": round(test_rmse, 4),
        "test_mae": round(test_mae, 4),
        "n_features": len(feat_cols),
        "top_features": top["feature"].tolist(),
        "model_version": MODEL_VERSION,
    }
    return stats, model, scaler, feat_cols, test_df, test_preds


def load_model():
    """Load saved model and scaler from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run train first.")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler, feat_cols = pickle.load(f)
    log.info("Model loaded from %s", MODEL_PATH)
    return model, scaler, feat_cols


def predict(df: pd.DataFrame, model=None, scaler=None, feat_cols=None) -> pd.DataFrame:
    """Generate predictions. Adds predicted_return, confidence, signal."""
    if model is None:
        model, scaler, feat_cols = load_model()

    # Build X with exact FEATURE_COLS order as DataFrame (scaler needs named columns)
    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        if col in df.columns:
            X[col] = df[col].values
        else:
            X[col] = 0.0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    # Confidence: use tree variance as proxy (std across trees)
    # XGBoost doesn't natively output uncertainty, so we use |pred| as proxy
    # In production, you'd use conformal prediction or quantile regression
    confidence = np.abs(preds) / (np.abs(preds).max() + 1e-9)

    result = df[["Date","Ticker"]].copy()
    result["predicted_return"] = (preds * 100).round(4)   # as %
    result["confidence"]       = confidence.round(4)
    result["signal"]           = pd.cut(
        result["predicted_return"],
        bins=[-np.inf, STRONG_SELL_THRESHOLD, SELL_THRESHOLD,
               BUY_THRESHOLD, STRONG_BUY_THRESHOLD, np.inf],
        labels=["Strong Sell","Sell","Hold","Buy","Strong Buy"]
    )
    return result


def get_feature_importance(model=None) -> pd.DataFrame:
    if model is None:
        model, scaler, feat_cols = load_model()
        _, _, feat_cols = pickle.load(open(SCALER_PATH,"rb"))
    return pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
