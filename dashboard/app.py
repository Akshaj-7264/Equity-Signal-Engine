"""
dashboard/app.py
Full professional research dashboard with:
  - Portfolio overview with signal heatmap
  - Per-stock deep dive (price chart + indicators)
  - Model performance metrics
  - Feature importance chart
  - LLM analyst chat (multi-turn)

Run: streamlit run dashboard/app.py
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from config import DB_PATH, SIGNALS_CSV, MODEL_PATH
from pipeline.store import (load_latest_predictions, load_predictions,
                              load_prices_df, load_model_runs)
from models.llm_analyst import analyse_stock, analyse_portfolio, chat

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 600; }
.signal-buy  { color: #00c853; font-weight: 700; }
.signal-sell { color: #f44336; font-weight: 700; }
.signal-hold { color: #9e9e9e; }
.stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; }
div[data-testid="metric-container"] {
    background: #1e1e2e; border-radius: 10px; padding: 12px;
    border: 1px solid #2d2d3e;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    try:
        preds   = load_latest_predictions()
        history = load_predictions()
        runs    = load_model_runs()
        return preds, history, runs
    except Exception as e:
        st.error(f"DB error: {e}. Run run_pipeline.py first.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=600)
def load_price_data(ticker):
    return load_prices_df(ticker)


def signal_color(sig):
    colors = {
        "Strong Buy":  "#00c853",
        "Buy":         "#69f0ae",
        "Hold":        "#9e9e9e",
        "Sell":        "#ff5252",
        "Strong Sell": "#b71c1c",
    }
    return colors.get(str(sig), "#9e9e9e")


def signal_emoji(sig):
    return {
        "Strong Buy": "🟢🟢", "Buy": "🟢",
        "Hold": "⚪", "Sell": "🔴", "Strong Sell": "🔴🔴"
    }.get(str(sig), "⚪")


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Quant Platform")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Portfolio Overview",
        "🔍 Stock Deep Dive",
        "🤖 AI Analyst Chat",
        "⚙️ Model Performance",
    ])
    st.markdown("---")
    api_key = st.text_input("Anthropic API Key", type="password",
                             help="Set for full LLM analysis. Optional.")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    st.caption("Pipeline: `python run_pipeline.py`")
    st.caption("Dashboard: `streamlit run dashboard/app.py`")


preds, history, runs = load_data()

if preds.empty:
    st.warning("No data found. Please run `python run_pipeline.py` first.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# PAGE 1: PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════
if "Portfolio" in page:
    st.title("📈 Portfolio Signal Dashboard")
    st.caption(f"Model date: {preds['date'].max()} | {len(preds)} tickers covered")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    buys   = preds[preds["signal"].isin(["Buy","Strong Buy"])]
    sells  = preds[preds["signal"].isin(["Sell","Strong Sell"])]
    holds  = preds[preds["signal"] == "Hold"]
    avg_r  = preds["predicted_return"].mean()
    bull_bias = len(buys) / len(preds) * 100

    col1.metric("🟢 Buy Signals",     len(buys))
    col2.metric("🔴 Sell Signals",    len(sells))
    col3.metric("⚪ Hold",            len(holds))
    col4.metric("Avg Predicted Ret",  f"{avg_r:+.2f}%")
    col5.metric("Bull Bias",          f"{bull_bias:.0f}%")

    st.markdown("---")

    # Signal heatmap
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Signal Heatmap — All Tickers")
        sig_order = ["Strong Buy","Buy","Hold","Sell","Strong Sell"]
        color_map  = {"Strong Buy":2,"Buy":1,"Hold":0,"Sell":-1,"Strong Sell":-2}
        heat_data = preds.copy()
        heat_data["score"] = heat_data["signal"].map(color_map)
        heat_data = heat_data.sort_values("predicted_return", ascending=False)

        fig_heat = px.bar(
            heat_data, x="ticker", y="predicted_return",
            color="signal",
            color_discrete_map={
                "Strong Buy":"#00c853","Buy":"#69f0ae","Hold":"#9e9e9e",
                "Sell":"#ff5252","Strong Sell":"#b71c1c"
            },
            labels={"predicted_return":"5-Day Predicted Return (%)","ticker":"Ticker"},
            height=380,
        )
        fig_heat.update_layout(
            plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
            font_color="white", legend_title="Signal",
            xaxis=dict(tickangle=45),
        )
        fig_heat.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_right:
        st.subheader("Signal Distribution")
        sig_counts = preds["signal"].value_counts().reindex(sig_order, fill_value=0)
        fig_pie = px.pie(
            values=sig_counts.values,
            names=sig_counts.index,
            color=sig_counts.index,
            color_discrete_map={
                "Strong Buy":"#00c853","Buy":"#69f0ae","Hold":"#9e9e9e",
                "Sell":"#ff5252","Strong Sell":"#b71c1c"
            },
            hole=0.45, height=380,
        )
        fig_pie.update_layout(
            plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
            font_color="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Signal table
    st.subheader("Full Signal Table")
    display = preds[["ticker","date","predicted_return","confidence","signal"]].copy()
    display["predicted_return"] = display["predicted_return"].apply(lambda x: f"{x:+.2f}%")
    display["confidence"] = display["confidence"].apply(lambda x: f"{x:.0%}")
    display["signal_icon"] = display["signal"].apply(signal_emoji)
    display = display.rename(columns={
        "ticker":"Ticker","date":"Date",
        "predicted_return":"Pred Return","confidence":"Confidence",
        "signal":"Signal","signal_icon":""
    })
    st.dataframe(display.sort_values("Pred Return", ascending=False),
                 use_container_width=True, height=400)

    # Portfolio note
    st.markdown("---")
    st.subheader("🤖 AI Portfolio Morning Note")
    if st.button("Generate Portfolio Note", type="primary"):
        with st.spinner("Generating analyst note..."):
            note = analyse_portfolio(preds)
        st.markdown(note)


# ══════════════════════════════════════════════════════════════════
# PAGE 2: STOCK DEEP DIVE
# ══════════════════════════════════════════════════════════════════
elif "Deep Dive" in page:
    st.title("🔍 Stock Deep Dive")

    ticker = st.selectbox("Select Ticker",
                           sorted(preds["ticker"].tolist()),
                           index=0)

    pred_row = preds[preds["ticker"] == ticker].iloc[0] if not preds.empty else pd.Series()
    prices   = load_price_data(ticker)

    if prices.empty:
        st.warning(f"No price data for {ticker}. Run the pipeline first.")
    else:
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values("date")

        # Signal badge
        sig = pred_row.get("signal", "N/A")
        ret = pred_row.get("predicted_return", 0)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Signal", f"{signal_emoji(sig)} {sig}")
        col2.metric("Predicted 5d Return", f"{float(ret):+.2f}%")
        col3.metric("Confidence", f"{float(pred_row.get('confidence',0)):.0%}")
        col4.metric("Latest Close", f"${prices['close'].iloc[-1]:.2f}")

        # Price chart with volume
        st.subheader(f"{ticker} — Price Chart + Volume")
        lookback = st.slider("Lookback (days)", 60, 1000, 252)
        prices_plot = prices.tail(lookback)

        # Compute MAs for chart
        prices_plot = prices_plot.copy()
        prices_plot["ma20"]  = prices_plot["close"].rolling(20).mean()
        prices_plot["ma50"]  = prices_plot["close"].rolling(50).mean()
        prices_plot["ma200"] = prices_plot["close"].rolling(200).mean()

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             row_heights=[0.55, 0.25, 0.20],
                             vertical_spacing=0.03)

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=prices_plot["date"],
            open=prices_plot["open"], high=prices_plot["high"],
            low=prices_plot["low"],  close=prices_plot["close"],
            name="Price", increasing_line_color="#00c853",
            decreasing_line_color="#f44336",
        ), row=1, col=1)

        for ma, color, name in [("ma20","#ffd600","MA20"),
                                  ("ma50","#ff9800","MA50"),
                                  ("ma200","#e91e63","MA200")]:
            fig.add_trace(go.Scatter(
                x=prices_plot["date"], y=prices_plot[ma],
                name=name, line=dict(color=color, width=1.2),
            ), row=1, col=1)

        # RSI
        close = prices_plot["close"]
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        fig.add_trace(go.Scatter(
            x=prices_plot["date"], y=rsi, name="RSI",
            line=dict(color="#00bcd4", width=1.5),
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#f44336",
                      opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00c853",
                      opacity=0.5, row=2, col=1)

        # Volume
        colors_vol = ["#00c853" if r["close"] >= r["open"] else "#f44336"
                       for _, r in prices_plot.iterrows()]
        fig.add_trace(go.Bar(
            x=prices_plot["date"], y=prices_plot["volume"],
            name="Volume", marker_color=colors_vol, opacity=0.7,
        ), row=3, col=1)

        fig.update_layout(
            height=650, plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
            font_color="white", xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.02),
        )
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI",  row=2, col=1, range=[0,100])
        fig.update_yaxes(title_text="Vol",  row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # AI note for this stock
        st.markdown("---")
        st.subheader(f"🤖 AI Analyst Note — {ticker}")
        question = st.text_input("Ask a specific question (optional)",
                                  placeholder="e.g. Why is this a Buy? What are the risks?")
        if st.button("Generate Analysis", type="primary"):
            with st.spinner("Analysing..."):
                note = analyse_stock(
                    ticker, pred_row,
                    question=question if question else None
                )
            st.markdown(note)


# ══════════════════════════════════════════════════════════════════
# PAGE 3: AI ANALYST CHAT
# ══════════════════════════════════════════════════════════════════
elif "Chat" in page:
    st.title("🤖 AI Analyst Chat")
    st.caption("Ask anything about the portfolio, signals, or specific stocks.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask about any stock or the portfolio..."):
        st.session_state.chat_messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat(
                    history     = st.session_state.chat_history,
                    new_message = prompt,
                    predictions = preds,
                    top_features= None,
                )
            st.markdown(response)
            st.session_state.chat_messages.append({"role":"assistant","content":response})
            st.session_state.chat_history.append({"role":"user","content":prompt})
            st.session_state.chat_history.append({"role":"assistant","content":response})

    # Suggested questions
    if not st.session_state.chat_messages:
        st.markdown("**Suggested questions:**")
        suggestions = [
            "What are the top 5 buy signals right now?",
            "Which stocks should I avoid this week?",
            "Explain the NVDA signal",
            "What is the overall market bias?",
            "What are the biggest risks to these signals?",
        ]
        cols = st.columns(len(suggestions))
        for col, q in zip(cols, suggestions):
            if col.button(q, use_container_width=True):
                st.session_state.chat_messages.append({"role":"user","content":q})
                st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.chat_messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.title("⚙️ Model Performance")

    if runs.empty:
        st.info("No model runs found. Run `python run_pipeline.py` first.")
    else:
        latest = runs.iloc[0]

        # KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Train R²",  f"{latest.get('train_r2', 0):.4f}")
        col2.metric("Val R²",    f"{latest.get('val_r2',   0):.4f}")
        col3.metric("Test R²",   f"{latest.get('test_r2',  0):.4f}")
        col4.metric("Test RMSE", f"{latest.get('test_rmse',0):.4f}")
        col5.metric("Test MAE",  f"{latest.get('test_mae', 0):.4f}")

        st.markdown("---")

        # Top features
        st.subheader("Feature Importance")
        try:
            top_feats = json.loads(latest.get("top_features","[]"))
            if top_feats:
                try:
                    from models.xgb_model import get_feature_importance
                    fi = get_feature_importance()
                    fig_fi = px.bar(
                        fi.head(20), x="importance", y="feature",
                        orientation="h", height=550,
                        color="importance",
                        color_continuous_scale="Teal",
                        labels={"importance":"Importance Score","feature":"Feature"},
                    )
                    fig_fi.update_layout(
                        plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                        font_color="white", yaxis=dict(autorange="reversed"),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
                except Exception:
                    st.write("Top features:", top_feats)
        except Exception:
            pass

        # Historical model runs
        st.subheader("Model Run History")
        if len(runs) > 1:
            fig_runs = px.line(
                runs.sort_values("run_ts"),
                x="run_ts", y=["train_r2","val_r2","test_r2"],
                labels={"value":"R²","run_ts":"Run timestamp","variable":"Split"},
                height=300,
            )
            fig_runs.update_layout(
                plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                font_color="white",
            )
            st.plotly_chart(fig_runs, use_container_width=True)

        st.dataframe(runs[["run_ts","train_r2","val_r2","test_r2",
                             "test_rmse","test_mae","n_features","model_version"]],
                     use_container_width=True)

        # Prediction distribution
        st.subheader("Prediction Distribution")
        if not preds.empty:
            fig_dist = px.histogram(
                preds, x="predicted_return", nbins=30,
                color="signal",
                color_discrete_map={
                    "Strong Buy":"#00c853","Buy":"#69f0ae","Hold":"#9e9e9e",
                    "Sell":"#ff5252","Strong Sell":"#b71c1c"
                },
                labels={"predicted_return":"Predicted 5d Return (%)"},
                height=350,
            )
            fig_dist.update_layout(
                plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                font_color="white", bargap=0.05,
            )
            fig_dist.add_vline(x=0, line_dash="dash",
                               line_color="white", opacity=0.5)
            st.plotly_chart(fig_dist, use_container_width=True)
