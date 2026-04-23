"""
run_pipeline.py
Orchestrates: ingest → features → store → train → predict → export
Run: python run_pipeline.py [--skip-fetch] [--skip-train]
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import TICKERS, DB_PATH, SIGNALS_CSV, MODEL_PATH
from logger import get_logger
from pipeline.ingest   import fetch_all
from pipeline.features import build_full_feature_set
from pipeline.store    import (init_db, upsert_prices, upsert_features,
                                upsert_predictions, save_model_run,
                                load_predictions, load_prices_df)
from models.xgb_model  import train, predict, MODEL_VERSION

log = get_logger("run_pipeline")


def run(skip_fetch=False, skip_train=False, tickers=None):
    tickers = tickers or TICKERS
    log.info("=== Quant Research Platform — Pipeline Start ===")
    log.info("Universe: %d tickers | DB: %s", len(tickers), DB_PATH)

    # 1. DB
    log.info("--- Stage 1: Database init ---")
    init_db()

    # 2. Ingest
    log.info("--- Stage 2: Data ingestion ---")
    if skip_fetch and load_prices_df().shape[0] > 1000:
        log.info("Using cached prices from DB")
        prices_raw = load_prices_df()
        prices_raw = prices_raw.rename(columns={
            "ticker":"Ticker","date":"Date","open":"Open",
            "high":"High","low":"Low","close":"Close","volume":"Volume"
        })
        prices_raw["Date"] = pd.to_datetime(prices_raw["Date"])
        macro = pd.DataFrame()
    else:
        prices_raw, macro = fetch_all(tickers)
        upsert_prices(prices_raw)

    log.info("Price rows available: %d", len(prices_raw))

    # 3. Features
    log.info("--- Stage 3: Feature engineering ---")
    features = build_full_feature_set(prices_raw, macro)
    log.info("Feature matrix: %d rows x %d cols", *features.shape)

    # Store sample of features (latest per ticker)
    latest = features.groupby("Ticker").tail(1)
    upsert_features(latest)

    # 4. Train
    if not skip_train:
        log.info("--- Stage 4: XGBoost training ---")
        stats, model, scaler, feat_cols, test_df, test_preds = train(features)
        save_model_run(stats)

        print("\n" + "="*60)
        print("  MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"  Train R2  : {stats['train_r2']:+.4f}")
        print(f"  Val   R2  : {stats['val_r2']:+.4f}")
        print(f"  Test  R2  : {stats['test_r2']:+.4f}")
        print(f"  Test  RMSE: {stats['test_rmse']:.4f}")
        print(f"  Test  MAE : {stats['test_mae']:.4f}")
        print(f"  Features  : {stats['n_features']}")
        print(f"  Top feats : {stats['top_features'][:5]}")
        print("="*60 + "\n")
    else:
        log.info("Skipping training (--skip-train)")
        model, scaler, feat_cols = None, None, None
        stats = {}

    # 5. Predict (latest signal per ticker)
    log.info("--- Stage 5: Generating signals ---")
    latest_features = features.groupby("Ticker").tail(1).copy()
    signals = predict(latest_features, model, scaler, feat_cols)
    upsert_predictions(signals, MODEL_VERSION)

    # 6. Export
    log.info("--- Stage 6: CSV export ---")
    signals.to_csv(SIGNALS_CSV, index=False)
    log.info("Signals exported to %s", SIGNALS_CSV)

    print("\nCurrent Signals:")
    print(signals[["Ticker","Date","predicted_return","signal","confidence"]].to_string(index=False))

    log.info("=== Pipeline complete ===")
    return features, signals, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()
    run(skip_fetch=args.skip_fetch, skip_train=args.skip_train, tickers=args.tickers)
