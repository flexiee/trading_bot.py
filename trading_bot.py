# app.py
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

st.set_page_config(page_title="Pro Trading Bot (Live)", layout="wide")
st.markdown("<h1 style='text-align:center;color:white;'>📈 Pro Trading Bot (Live)</h1>", unsafe_allow_html=True)

# -----------------------
# Sidebar: settings
# -----------------------
st.sidebar.header("Account & Settings")
account_balance = st.sidebar.number_input("Account Balance ($)", value=1000.0, step=10.0)
risk_percent = st.sidebar.slider("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
timeframe = st.sidebar.selectbox("Chart timeframe", ["1m", "5m", "15m", "1h"], index=0)
refresh_seconds = st.sidebar.number_input("Auto-refresh (seconds, 0=off)", value=0, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("Made for manual execution. No auto-trading included.")

# -----------------------
# Market mapping
# -----------------------
markets = {
    "EUR/USD": "EURUSD=X",
    "GBP/JPY": "GBPJPY=X",
    "USD/JPY": "JPY=X",        # note: JPY ticker in yfinance
    "AUD/USD": "AUDUSD=X",
    "XAU/USD (Gold)": "XAUUSD=X",
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD"
}

tv_symbol_map = {
    "EUR/USD": "OANDA:EURUSD",
    "GBP/JPY": "OANDA:GBPJPY",
    "USD/JPY": "OANDA:USDJPY",
    "AUD/USD": "OANDA:AUDUSD",
    "XAU/USD (Gold)": "OANDA:XAUUSD",
    "BTC/USD": "BINANCE:BTCUSDT",
    "ETH/USD": "BINANCE:ETHUSDT"
}

selected_market = st.selectbox("Select Market", list(markets.keys()))

# -----------------------
# TradingView embed
# -----------------------
st.subheader(f"📡 Live TradingView Chart — {selected_market}")
tv_symbol = tv_symbol_map.get(selected_market, "OANDA:EURUSD")
tradingview_html = f"""
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
    "width": "100%",
    "height": 520,
    "symbol": "{tv_symbol}",
    "interval": "{timeframe.replace('m','') if 'm' in timeframe else timeframe}",
    "timezone": "Etc/UTC",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#222",
    "enable_publishing": false,
    "allow_symbol_change": true,
    "container_id": "tradingview_chart"
  }});
  </script>
</div>
"""
components.html(tradingview_html, height=540)

# -----------------------
# Data fetching & indicators
# -----------------------
@st.cache_data(ttl=30)
def fetch_ohlc(symbol, interval):
    # Map Streamlit timeframe to yfinance period/interval
    yf_interval = interval
    yf_period = "7d"
    # For 1m we need 7d or 1d depending on availability; use 7d to be safe
    if interval == "1m":
        yf_period = "7d"
    elif interval == "5m":
        yf_period = "30d"
    elif interval == "15m":
        yf_period = "60d"
    elif interval == "1h":
        yf_period = "120d"

    try:
        df = yf.download(tickers=symbol, interval=interval, period=yf_period, progress=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        return df
    except Exception as e:
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

symbol = markets[selected_market]
with st.spinner("Fetching latest market data..."):
    data = fetch_ohlc(symbol, timeframe)
if data is None:
    st.error("Failed to fetch data. yfinance may not provide this interval for this symbol.")
    st.stop()

# compute indicators
data["EMA20"] = calculate_ema(data["Close"], 20)
data["RSI14"] = calculate_rsi(data["Close"], 14)

latest = data.iloc[-1]
price = float(latest["Close"])
ema20 = float(latest["EMA20"])
rsi14 = float(latest["RSI14"])

# -----------------------
# Signal logic (combination)
# -----------------------
# We'll combine RSI + EMA crossover + momentum to create confidence score
def compute_signal(price, ema, rsi):
    reasons = []
    score = 0

    # Trend check
    if price > ema:
        trend = "uptrend"
        score += 30
    else:
        trend = "downtrend"
        score += 30

    # RSI check
    if rsi < 30:
        score += 40
        reasons.append("RSI oversold (buy bias)")
    elif rsi > 70:
        score += 40
        reasons.append("RSI overbought (sell bias)")
    else:
        # proximity to middle adds little
        if 40 <= rsi <= 60:
            score += 10

    # EMA/price confirmation
    if (price > ema and rsi < 60) or (price < ema and rsi > 40):
        score += 20
        reasons.append("EMA confirmation")

    # Cap confidence 0-100
    confidence = min(110, int(score))
    # Determine signal
    signal = "WAIT"
    if price > ema and rsi < 50 and confidence >= 60:
        signal = "BUY"
    elif price < ema and rsi > 50 and confidence >= 60:
        signal = "SELL"
    # More aggressive rules for strong rsi extremes
    if rsi < 25 and price > ema:
        signal = "BUY"
        confidence = max(confidence, 90)
        reasons.append("Strong RSI buy")
    if rsi > 75 and price < ema:
        signal = "SELL"
        confidence = max(confidence, 90)
        reasons.append("Strong RSI sell")

    return signal, confidence, reasons, trend

signal, confidence, reasons, trend = compute_signal(price, ema20, rsi14)

# -----------------------
# Risk management & lot size
# -----------------------
def estimate_pip_value(market_name):
    # approximate pip unit in price terms
    if "JPY" in market_name:
        return 0.01
    if "XAU" in market_name:
        return 0.1  # gold volatility unit approx (not exact)
    if "BTC" in market_name or "ETH" in market_name:
        return 1.0  # work in USD units
    return 0.0001

pip_unit = estimate_pip_value(selected_market)
# choose a practical SL distance based on volatility (use recent std)
recent_std = np.std(data["Close"].pct_change().tail(20)) * price
# fall back if tiny
sl_distance = max(abs(recent_std) * 1.5, pip_unit * 10)

if signal == "BUY":
    sl = price - sl_distance
    tp = price + sl_distance * 3
elif signal == "SELL":
    sl = price + sl_distance
    tp = price - sl_distance * 3
else:
    sl = None
    tp = None

risk_amount = (risk_percent / 100.0) * account_balance
lot_size = 0
if sl is not None:
    stop_loss_pips = abs(price - sl) / pip_unit
    # pip monetary value approximation: vary by symbol; for forex approx $10 per pip per lot (very approximate)
    # We'll assume pip_value_usd_per_lot; using 100000 units as standard lot doesn't apply for all symbols;
    # Provide a conservative small-lot recommendation in units (not exact broker lots)
    pip_usd_val = 10  # approximate USD per pip per 1 standard lot (for major forex)
    # number of standard lots sized to risk_amount
    lot_size = (risk_amount) / (stop_loss_pips * pip_usd_val) if stop_loss_pips > 0 else 0
    lot_size = round(lot_size, 4)

# -----------------------
# UI: Show signal + metrics
# -----------------------
st.subheader("🔍 Signal Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Price", f"{price:.6f}")
col2.metric("EMA20", f"{ema20:.6f}")
col3.metric("RSI(14)", f"{rsi14:.2f}")

st.markdown(f"**Trend:** {trend}   •   **Signal:** {signal}   •   **Confidence:** {confidence}%")
if reasons:
    st.markdown("**Reasons:** " + ", ".join(reasons))

if sl is not None and tp is not None:
    st.info(f"Entry: {price:.6f}  •  SL: {sl:.6f}  •  TP: {tp:.6f}  •  Risk ${risk_amount:.2f}  •  Lot Size ≈ {lot_size}")
else:
    st.info("No SL/TP (no actionable signal)")

# Buttons to save the signal to history
if st.button("📥 Save Signal to History"):
    hist_file = "signal_history.csv"
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Market": selected_market,
        "Signal": signal,
        "Entry": price,
        "SL": sl if sl is not None else "",
        "TP": tp if tp is not None else "",
        "Risk($)": round(risk_amount,2),
        "Lot": lot_size,
        "Confidence": confidence,
        "Result": "Pending"
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(hist_file):
        df_row.to_csv(hist_file, mode="a", header=False, index=False)
    else:
        df_row.to_csv(hist_file, index=False)
    st.success("Saved to signal_history.csv")

# -----------------------
# Signal history viewer & mark result
# -----------------------
st.subheader("📜 Signal History (Last 50)")
hist_file = "signal_history.csv"
if os.path.exists(hist_file):
    hist_df = pd.read_csv(hist_file)
    st.dataframe(hist_df.tail(50).reset_index(drop=True))
    # Option to mark a row as success/fail
    idx = st.number_input("Enter history row number (0..n-1) to mark result", min_value=0, value=0, step=1)
    action = st.selectbox("Mark selected row as", ["No change", "Success", "Failure"])
    if st.button("Update History Row Result"):
        if os.path.exists(hist_file):
            dfh = pd.read_csv(hist_file)
            if 0 <= idx < len(dfh):
                dfh.at[idx, "Result"] = "Success" if action=="Success" else ("Failure" if action=="Failure" else dfh.at[idx,"Result"])
                dfh.to_csv(hist_file, index=False)
                st.success("Updated history file.")
            else:
                st.error("Index out of range.")
else:
    st.info("No signal history yet. Save signals to populate history.")

# -----------------------
# Footer & auto-refresh
# -----------------------
st.markdown("---")
st.caption("Disclaimer: For educational/analysis use only. Always test & confirm signals on your broker platform before placing real trades.")

# auto refresh helper (simple)
if refresh_seconds > 0:
    st.experimental_rerun()
