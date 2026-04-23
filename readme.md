# Multi-Factor Equity Signal Engine

An end-to-end quantitative equity research platform that ingests 10 years of market data for 50 US stocks, engineers 48 real-world financial features, trains an XGBoost model to predict 5-day forward returns, and delivers signals through an AI-powered Streamlit dashboard with LLM-generated analyst notes.

---

## What it does

Every serious quant research team runs some version of this workflow: ingest price and macro data, engineer signals, train a model, generate Buy/Hold/Sell calls, and deliver them to analysts in plain English. This project implements that full stack in Python.

The LLM layer (Claude Sonnet) converts raw model outputs — a predicted return percentage and confidence score — into proper analyst notes with risk assessments, key driver explanations, and position sizing guidance. You can also chat with it: *"Why is NVDA a hold?"*, *"What are the biggest macro risks this week?"*, *"Which sector looks strongest?"*

---

## Architecture

```
yfinance API + Macro (VIX, Yields, DXY, Gold)
           │
           ▼
  pipeline/ingest.py        ← Fetch & validate 143,500+ price rows
           │
           ▼
  pipeline/features.py      ← Engineer 48 financial features
           │
           ▼
  pipeline/store.py         ← Persist to SQLite (parameterised queries)
           │
           ▼
  models/xgb_model.py       ← XGBoost: train → validate → test → predict
           │
           ▼
  models/llm_analyst.py     ← Claude Sonnet: signals → analyst language
           │
           ▼
  dashboard/app.py          ← Streamlit: 4-page research dashboard
           │
           ▼
  outputs/signals.csv       ← Analyst-ready export
```

---

## Feature Set (48 features)

| Category | Features |
|---|---|
| **Price Momentum** | 1d, 5d, 10d, 21d, 63d, 126d, 252d returns |
| **Moving Averages** | MA20, MA50, MA200, price-to-MA ratios, golden/death cross signals |
| **Oscillators** | RSI (14, 28), overbought/oversold flags |
| **MACD** | MACD histogram, signal line crossover |
| **Bollinger Bands** | Band width (volatility proxy), %B position |
| **Volatility** | ATR%, realised vol (21d, 63d), vol-of-vol, vol regime |
| **Volume** | OBV trend, MFI, volume ratio vs 20d average |
| **Mean Reversion** | Z-scores vs 20d and 63d mean |
| **Macro** | VIX level + regime, 10Y yield + change, DXY momentum, S&P 500 return |
| **Relative Strength** | Stock return vs universe median (cross-sectional momentum) |
| **Calendar** | Day of week, month, quarter end, month end |
| **Intraday** | Overnight gap, intraday range |

---

## Model

**Algorithm:** XGBoost Regressor
**Target:** 5-day forward return
**Split:** Strict chronological train (75%) / val (10%) / test (15%) — no random shuffling to prevent lookahead leakage

**Regularisation:**
- L1 (reg_alpha) + L2 (reg_lambda) to prevent overfitting
- Minimum child weight = 10 (stable leaf nodes)
- Early stopping on validation RMSE

**Signal thresholds:**

| Signal | Predicted 5d Return |
|---|---|
| Strong Buy | > +3% |
| Buy | +1.5% to +3% |
| Hold | -1.5% to +1.5% |
| Sell | -3% to -1.5% |
| Strong Sell | < -3% |

> **Note on R²:** Equity return prediction is extremely hard. Academic literature considers R² of 0.03–0.08 on out-of-sample data to be a strong result. The model's value is in ranking stocks relative to each other, not in point prediction accuracy.

---

## Dashboard — 4 Pages

**Portfolio Overview**
- Signal heatmap across all 50 tickers (colour-coded bar chart)
- Signal distribution pie chart
- Full signal table with predicted return, confidence, and signal
- One-click AI portfolio morning note

**Stock Deep Dive**
- Interactive candlestick chart with MA20/50/200 overlays
- RSI panel with overbought/oversold zones
- Volume panel with directional colouring
- Adjustable lookback (60–1000 days)
- Per-stock AI analyst note with optional custom question

**AI Analyst Chat**
- Multi-turn conversation powered by Claude Sonnet
- Automatically injects live model data and stock technicals as context
- Pre-built suggested questions for quick exploration

**Model Performance**
- Train/Val/Test R², RMSE, MAE
- Feature importance bar chart (top 20)
- Historical model run comparison
- Prediction return distribution by signal

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-username/equity-signal-engine
cd equity-signal-engine

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline (fetches data, trains model, generates signals)
python run_pipeline.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

**For AI analyst notes** (optional), paste your Anthropic API key in the dashboard sidebar, or set it as an environment variable:

```bash
# Windows
set ANTHROPIC_API_KEY=your_key_here

# Mac/Linux
export ANTHROPIC_API_KEY=your_key_here
```

Without the key, the platform uses rule-based analyst notes — all other features work fully.

**Re-run pipeline without re-fetching data:**
```bash
python run_pipeline.py --skip-fetch
```

**Run for a subset of tickers:**
```bash
python run_pipeline.py --tickers AAPL MSFT NVDA JPM
```

---

## Project Structure

```
equity-signal-engine/
├── config.py                  # All constants — tickers, dates, thresholds
├── logger.py                  # Centralised UTF-8 safe logging
├── run_pipeline.py            # Master orchestrator (7 stages)
├── requirements.txt
├── pipeline/
│   ├── ingest.py              # yfinance fetch + macro data + GBM fallback
│   ├── features.py            # 48-feature engineering pipeline
│   └── store.py               # SQLite CRUD (parameterised queries only)
├── models/
│   ├── xgb_model.py           # XGBoost train/predict/feature importance
│   └── llm_analyst.py         # Claude Sonnet analyst + chat + fallback
├── dashboard/
│   └── app.py                 # 4-page Streamlit dashboard
├── data/
│   └── quant.db               # SQLite database (auto-created)
├── outputs/
│   └── signals.csv            # Latest signal export
└── logs/
    └── platform.log           # Full pipeline log
```

---

## Universe — 50 US Equities

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, INTC, ORCL, CRM |
| Finance | JPM, GS, BAC, MS, WFC, BLK, AXP, C, USB, TFC |
| Healthcare | JNJ, UNH, PFE, MRK, ABBV, LLY, TMO, ABT, CVS, MDT |
| Energy | XOM, CVX, COP, SLB, EOG |
| Consumer | TSLA, NKE, MCD, SBUX, HD, LOW, TGT, COST, WMT, PG |
| Media/Telecom | NFLX, DIS, CMCSA, T, VZ |

---

## Security

- Ticker inputs validated against explicit allowlist + `[A-Z]{1,5}` regex
- All SQL uses parameterised placeholders — zero string interpolation
- DB path resolved from `config.py` only — never from external input
- No `eval()`, `exec()`, or `subprocess` calls anywhere

---

## Tech Stack

`Python 3.12` · `pandas` · `numpy` · `scipy` · `scikit-learn` · `xgboost` · `yfinance` · `streamlit` · `plotly` · `anthropic` · `sqlite3`

---

## Limitations & Honest Notes

- **R² near zero on simulated data** is expected — GBM-generated returns are by design unpredictable. Real Yahoo Finance data produces meaningful signals.
- **Past performance does not guarantee future returns.** This is a research tool, not a trading system.
- **Macro features require network access** to Yahoo Finance. In offline/sandboxed environments, GBM simulation runs automatically.
- **40 observations per ticker** in the earnings variant is a known limitation — per-ticker R² should be interpreted with caution at small sample sizes.
- This project is for educational and portfolio demonstration purposes only.