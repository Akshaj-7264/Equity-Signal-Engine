"""
models/llm_analyst.py
---------------------
LLM-powered analyst that converts raw model outputs into
natural language research notes, risk assessments, and
interactive Q&A about any stock or the portfolio.

Uses the Anthropic API (claude-sonnet-4-20250514).
Falls back to rule-based templates if API key is unavailable.
"""
import os, sys, json
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior quantitative research analyst at a top-tier investment bank.
You have access to a machine learning model (XGBoost) that predicts 5-day forward returns
for 50 US equities using 40+ features including technical indicators, macro factors, and
cross-sectional momentum.

When given model outputs, you must:
1. Explain the prediction in plain English — what the model sees and why
2. Identify the KEY drivers behind the signal (which features are pushing it)
3. Give a RISK ASSESSMENT — what could make the model wrong
4. State a CONFIDENCE level and why
5. Suggest a POSITION SIZE guideline (be conservative)
6. Use professional financial language but be clear enough for a junior analyst

Always be honest about model limitations. A high predicted return does NOT guarantee profit.
Never give specific price targets — only directional signals and % return estimates.
Format responses with clear sections using markdown headers."""


def _get_client():
    """Get Anthropic client. Returns None if key not set."""
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.warning("ANTHROPIC_API_KEY not set — using rule-based fallback")
            return None
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        log.warning("anthropic package not installed")
        return None


def _build_stock_context(
    ticker: str,
    prediction_row: pd.Series,
    feature_row: Optional[pd.Series] = None,
    top_features: Optional[list] = None,
) -> str:
    """Build structured context string for the LLM."""
    pred   = prediction_row.get("predicted_return", 0)
    signal = prediction_row.get("signal", "Hold")
    conf   = prediction_row.get("confidence", 0)
    date   = prediction_row.get("date", "N/A")

    ctx = f"""
=== MODEL OUTPUT FOR {ticker} ===
Date: {date}
Signal: {signal}
Predicted 5-day return: {pred:+.2f}%
Model confidence score: {conf:.2%}

=== KEY TECHNICAL STATE ===
"""
    if feature_row is not None:
        tech_fields = {
            "RSI (14)":           "rsi_14",
            "MACD histogram":     "macd_hist",
            "Bollinger %B":       "bb_pct",
            "Z-score (20d)":      "z_score_20d",
            "Volume ratio":       "volume_ratio",
            "Realised vol (21d)": "realised_vol_21d",
            "Price vs MA200":     "price_to_ma200",
            "Rel strength 21d":   "rel_strength_21d",
            "ATR %":              "atr_pct",
            "1-day return":       "ret_1d",
            "5-day return":       "ret_5d",
            "21-day return":      "ret_21d",
        }
        for label, col in tech_fields.items():
            val = feature_row.get(col)
            if val is not None and pd.notna(val):
                ctx += f"  {label}: {float(val):.4f}\n"

        macro_fields = {
            "VIX level":       "vix",
            "10Y yield":       "treasury_10y",
            "VIX regime":      "vix_regime",
            "Market ret 5d":   "mkt_ret_5d",
        }
        ctx += "\n=== MACRO ENVIRONMENT ===\n"
        for label, col in macro_fields.items():
            val = feature_row.get(col)
            if val is not None and pd.notna(val):
                ctx += f"  {label}: {float(val):.4f}\n"

    if top_features:
        ctx += f"\n=== TOP MODEL FEATURES (global importance) ===\n"
        for i, f in enumerate(top_features[:8], 1):
            ctx += f"  {i}. {f}\n"

    return ctx


def analyse_stock(
    ticker: str,
    prediction_row: pd.Series,
    feature_row: Optional[pd.Series] = None,
    top_features: Optional[list] = None,
    question: Optional[str] = None,
) -> str:
    """
    Generate a full analyst note for a single stock.
    If `question` is provided, answer it specifically.
    """
    client = _get_client()
    context = _build_stock_context(ticker, prediction_row, feature_row, top_features)

    if question:
        user_msg = f"{context}\n\nAnalyst question: {question}"
    else:
        user_msg = f"{context}\n\nPlease write a full analyst note for {ticker} based on this model output."

    if client is None:
        return _rule_based_note(ticker, prediction_row, question)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text
    except Exception as e:
        log.error("LLM call failed: %s — using fallback", e)
        return _rule_based_note(ticker, prediction_row, question)


def analyse_portfolio(
    predictions: pd.DataFrame,
    question: Optional[str] = None,
) -> str:
    """
    Generate a portfolio-level summary note.
    predictions: DataFrame with ticker, predicted_return, signal columns.
    """
    client = _get_client()

    buys  = predictions[predictions["signal"].isin(["Buy","Strong Buy"])]
    sells = predictions[predictions["signal"].isin(["Sell","Strong Sell"])]
    holds = predictions[predictions["signal"] == "Hold"]

    top5_buy  = buys.nlargest(5,  "predicted_return")[["ticker","predicted_return","signal","confidence"]]
    top5_sell = sells.nsmallest(5,"predicted_return")[["ticker","predicted_return","signal","confidence"]]

    context = f"""
=== PORTFOLIO MODEL SUMMARY ===
Total tickers covered: {len(predictions)}
Buy signals: {len(buys)} | Hold: {len(holds)} | Sell: {len(sells)}
Average predicted return: {predictions['predicted_return'].mean():+.2f}%
Median predicted return:  {predictions['predicted_return'].median():+.2f}%
Signal distribution: {predictions['signal'].value_counts().to_dict()}

TOP 5 BUY SIGNALS:
{top5_buy.to_string(index=False)}

TOP 5 SELL SIGNALS:
{top5_sell.to_string(index=False)}
"""
    user_msg = context
    if question:
        user_msg += f"\n\nAnalyst question: {question}"
    else:
        user_msg += "\n\nWrite a portfolio morning note based on these model signals."

    if client is None:
        return _rule_based_portfolio(predictions, question)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text
    except Exception as e:
        log.error("LLM portfolio call failed: %s", e)
        return _rule_based_portfolio(predictions, question)


def chat(
    history: list[dict],
    new_message: str,
    predictions: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    top_features: Optional[list] = None,
) -> str:
    """
    Multi-turn chat about the portfolio/stocks.
    history: list of {"role":"user"/"assistant","content":"..."}
    """
    client = _get_client()

    # Build context summary to inject at start
    buys  = predictions[predictions["signal"].isin(["Buy","Strong Buy"])]
    sells = predictions[predictions["signal"].isin(["Sell","Strong Sell"])]

    context_inject = f"""[LIVE MODEL DATA]
Tickers: {', '.join(predictions['ticker'].tolist())}
Buy signals: {', '.join(buys['ticker'].tolist())}
Sell signals: {', '.join(sells['ticker'].tolist())}
Avg predicted return: {predictions['predicted_return'].mean():+.2f}%
Top features (global): {', '.join((top_features or [])[:5])}
"""
    # Enrich if asking about specific ticker
    ticker_mentioned = None
    for t in predictions["ticker"].tolist():
        if t.upper() in new_message.upper():
            ticker_mentioned = t
            break

    if ticker_mentioned and features is not None:
        row = predictions[predictions["ticker"] == ticker_mentioned].iloc[0]
        feat_row = features[features["Ticker"] == ticker_mentioned]
        if not feat_row.empty:
            feat_row = feat_row.iloc[-1]
            context_inject += _build_stock_context(
                ticker_mentioned, row, feat_row, top_features
            )

    messages = [{"role": "user", "content": context_inject + "\n\nUser: " + history[0]["content"]}] if history else []
    for i, h in enumerate(history[1:] if history else []):
        messages.append(h)
    messages.append({"role": "user", "content": new_message})

    if client is None:
        return _rule_based_chat(new_message, predictions)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages if messages else [{"role":"user","content": context_inject + "\n" + new_message}],
        )
        return response.content[0].text
    except Exception as e:
        log.error("LLM chat failed: %s", e)
        return _rule_based_chat(new_message, predictions)


# ── Rule-based fallbacks (no API key needed) ───────────────────

def _rule_based_note(ticker, pred_row, question=None) -> str:
    pred   = pred_row.get("predicted_return", 0)
    signal = pred_row.get("signal", "Hold")
    conf   = pred_row.get("confidence", 0)

    direction = "bullish" if pred > 0 else "bearish"
    strength  = "strongly " if abs(pred) > 3 else ""

    note = f"""## {ticker} — Model Signal: {signal}

**Predicted 5-day return:** {pred:+.2f}%
**Model confidence:** {conf:.0%}

### Summary
The model is {strength}{direction} on {ticker} over the next 5 trading days,
forecasting a {pred:+.2f}% move. This is classified as **{signal}**.

### Key Drivers
The signal is driven by a combination of price momentum, technical positioning,
volume patterns, and macro environment. A {signal} signal at this confidence
level suggests the model has identified a {"favorable" if pred > 0 else "unfavorable"}
risk/reward setup.

### Risk Assessment
- Model predictions are probabilistic, not guaranteed
- Macro shocks (Fed announcements, geopolitical events) can override technical signals
- Confidence score of {conf:.0%} means {"moderate" if conf < 0.6 else "high"} model conviction
- Position sizing should reflect this uncertainty

### Recommendation
{"Consider a small long position with tight stop-loss" if pred > 2 else
 "Consider a small short position" if pred < -2 else
 "No clear edge — stay flat or reduce position"}

*Note: Set ANTHROPIC_API_KEY for full LLM-powered analysis.*
"""
    if question:
        note += f"\n\n**Re: '{question}'** — Full LLM Q&A requires ANTHROPIC_API_KEY."
    return note


def _rule_based_portfolio(predictions, question=None) -> str:
    buys  = predictions[predictions["signal"].isin(["Buy","Strong Buy"])]
    sells = predictions[predictions["signal"].isin(["Sell","Strong Sell"])]
    avg   = predictions["predicted_return"].mean()

    bias = "bullish" if avg > 0 else "bearish"
    return f"""## Portfolio Morning Note

**Model bias:** {bias.upper()} (avg predicted return: {avg:+.2f}%)
**Buy signals ({len(buys)}):** {', '.join(buys['ticker'].tolist())}
**Sell signals ({len(sells)}):** {', '.join(sells['ticker'].tolist())}

The model sees a broadly {bias} setup for the next 5 days. Focus on
high-confidence signals and maintain appropriate position sizing.

*Set ANTHROPIC_API_KEY for full AI-powered analysis.*
"""


def _rule_based_chat(message, predictions) -> str:
    msg = message.lower()
    if "buy" in msg or "long" in msg:
        buys = predictions[predictions["signal"].isin(["Buy","Strong Buy"])]
        return f"Top buy signals: {', '.join(buys.nlargest(5,'predicted_return')['ticker'].tolist())}.\n*Set ANTHROPIC_API_KEY for detailed analysis.*"
    if "sell" in msg or "short" in msg:
        sells = predictions[predictions["signal"].isin(["Sell","Strong Sell"])]
        return f"Top sell signals: {', '.join(sells.nsmallest(5,'predicted_return')['ticker'].tolist())}.\n*Set ANTHROPIC_API_KEY for detailed analysis.*"
    return f"The model covers {len(predictions)} tickers. Avg predicted return: {predictions['predicted_return'].mean():+.2f}%.\n*Set ANTHROPIC_API_KEY for full Q&A.*"
