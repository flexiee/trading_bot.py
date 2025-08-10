# app.py
# Pro Trading Bot — Update: improved signal engine, risk manager, lot sizing & history
# NOTE: This update keeps the exact same UI/layout as the last working version (TradingView embed, sidebar, etc.)
# Copy-paste this file replacing your existing app.py. Install requirements (requirements.txt) and run:
# pip install -r requirements.txt
# streamlit run app.py

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# ----------------------- App config -----------------------
st.set_page_config(page_title="Pro Trading Bot (Live)", layout="wide")
st.markdown("<h1 style='text-align:center;color:white;'>📈 Pro Trading Bot (Live)</h1>", unsafe_allow_html=True)

# ----------------------- Sidebar: settings (unchanged UI) -----------------------
st.sidebar.header("Account & Settings")
account_balance = st.sidebar.number_input("Account Balance ($)", value=1000.0, step=10.0, format="%.2f")
risk_percent = st.sidebar.slider("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
timeframe = st.sidebar.selectbox("Chart timeframe", ["1m", "5m", "15m", "1h"], index=0)
refresh_seconds = st.sidebar.number_input("Auto-refresh (seconds, 0=off)", value=0, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("Made for manual execution. No auto-trading included.")

# ----------------------- Market mapping (unchanged UI) -----------------------
markets = {
    "EUR/USD": "EURUSD=X",
    "GBP/JPY": "GBPJPY=X",
    "USD/JPY": "JPY=X",
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

# ----------------------- TradingView embed (kept as requested, larger) -----------------------
st.subheader(f"📡 Live TradingView Chart — {selected_market}")
tv_symbol = tv_symbol_map.get(selected_market, "OANDA:EURUSD")
tv_interval_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60"}
tv_interval = tv_interval_map.get(timeframe, "1")

tradingview_html = f"""
<div class="tradingview-widget-container" style="height:100%;">
  <div id="tradingview_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
    "width": "100%",
    "height": 800,
    "symbol": "{tv_symbol}",
    "interval": "{tv_interval}",
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
components.html(tradingview_html, height=820)

# ----------------------- Data fetching & indicator utilities -----------------------
@st.cache_data(ttl=30)
def fetch_ohlc(symbol, interval):
    # map to yfinance interval and period
    yf_interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m"}
    yf_interval = yf_interval_map.get(interval, "1m")
    period_map = {"1m": "7d", "5m": "30d", "15m": "60d", "1h": "120d"}
    yf_period = period_map.get(interval, "7d")
    try:
        df = yf.download(tickers=symbol, interval=yf_interval, period=yf_period, progress=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        return df
    except Exception:
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

# ----------------------- Fetch data -----------------------
symbol = markets[selected_market]
with st.spinner("Fetching latest market data..."):
    data = fetch_ohlc(symbol, timeframe)
if data is None or data.empty:
    st.error("Failed to fetch data. yfinance may not provide this interval or data is unavailable.")
    st.stop()

# compute indicators
data["EMA20"] = calculate_ema(data["Close"], 20)
data["RSI14"] = calculate_rsi(data["Close"], 14)
data["MACD"], data["MACD_SIGNAL"], data["MACD_HIST"] = calculate_macd(data["Close"])

latest = data.iloc[-1]
price = float(latest["Close"])
ema20 = float(latest["EMA20"])
rsi14 = float(latest["RSI14"])
macd = float(latest["MACD"])
macd_signal = float(latest["MACD_SIGNAL"])
macd_hist = float(latest["MACD_HIST"])

# ----------------------- Improved signal engine (multi-strategy merge) -----------------------
def compute_signal_multi(price, ema, rsi, macd_val, macd_sig, df):
    reasons = []
    score = 0

    # Trend via EMA
    if price > ema:
        trend = "uptrend"
        score += 25
    else:
        trend = "downtrend"
        score += 25

    # RSI zones
    if rsi < 30:
        score += 30
        reasons.append("RSI oversold")
    elif rsi > 70:
        score += 30
        reasons.append("RSI overbought")
    else:
        if 40 <= rsi <= 60:
            score += 10

    # MACD histogram momentum
    if macd_hist > 0:
        score += 15
        reasons.append("MACD bullish")
    else:
        score += 5
        reasons.append("MACD bearish")

    # Candle momentum check (last 3 closes)
    closes = df["Close"].tail(5).values
    if len(closes) >= 3 and closes[-1] > closes[-2] > closes[-3]:
        score += 10
        reasons.append("3-bar upward momentum")
    elif len(closes) >= 3 and closes[-1] < closes[-2] < closes[-3]:
        score += 10
        reasons.append("3-bar downward momentum")

    # Multi-timeframe confirmation: check higher timeframe EMA direction (simple)
    # We'll try to fetch a small bit of higher timeframe (if user used 1m or 5m, use 15m/1h for confirmation)
    ht_confirm = False
    try:
        if timeframe == "1m":
            ht = "5m"
        elif timeframe == "5m":
            ht = "15m"
        else:
            ht = "1h"
        ht_data = fetch_ohlc(symbol, ht)
        if ht_data is not None and not ht_data.empty:
            ht_ema = calculate_ema(ht_data["Close"], 20).iloc[-1]
            # confirm trend
            if price > ema and ht_data["Close"].iloc[-1] > ht_ema:
                score += 10
                reasons.append("Higher-TF confirms uptrend")
                ht_confirm = True
            elif price < ema and ht_data["Close"].iloc[-1] < ht_ema:
                score += 10
                reasons.append("Higher-TF confirms downtrend")
                ht_confirm = True
    except Exception:
        pass

    confidence = min(120, int(score))  # allow >100 to express very strong convergence
    signal = "WAIT"
    # Decision rules:
    if price > ema and rsi < 55 and macd_hist > 0 and confidence >= 60:
        signal = "BUY"
    elif price < ema and rsi > 45 and macd_hist < 0 and confidence >= 60:
        signal = "SELL"

    # strong override
    if rsi < 20 and price > ema:
        signal = "BUY"
        confidence = max(confidence, 110)
        reasons.append("Extreme RSI buy override")
    if rsi > 80 and price < ema:
        signal = "SELL"
        confidence = max(confidence, 110)
        reasons.append("Extreme RSI sell override")

    return signal, confidence, reasons, trend, ht_confirm

signal, confidence, reasons, trend, ht_confirm = compute_signal_multi(price, ema20, rsi14, macd, macd_signal, data)

# ----------------------- Risk management & lot size improvements -----------------------
def estimate_pip_value(market_name):
    if "JPY" in market_name:
        return 0.01
    if "XAU" in market_name:
        return 0.1
    if "BTC" in market_name or "ETH" in market_name:
        return 1.0
    return 0.0001

pip_unit = estimate_pip_value(selected_market)

recent_pct = data["Close"].pct_change().dropna().tail(20)
if recent_pct.empty:
    backup = data["Close"].pct_change().dropna()
    if not backup.empty:
        recent_std_fraction = float(np.nanstd(backup))
    else:
        recent_std_fraction = 0.0001
else:
    recent_std_fraction = float(np.nanstd(recent_pct))

recent_std_fraction = abs(recent_std_fraction) if not np.isnan(recent_std_fraction) else 0.0001
recent_std_price = recent_std_fraction * price

sl_distance = max(recent_std_price * 1.5, pip_unit * 10)

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
lot_size = 0.0
if sl is not None:
    stop_loss_pips = abs(price - sl) / pip_unit if pip_unit != 0 else 0
    pip_usd_val = 10  # conservative approx USD per pip per standard lot
    lot_size = (risk_amount) / (stop_loss_pips * pip_usd_val) if stop_loss_pips > 0 else 0
    lot_size = round(lot_size, 6)

# compute R:R (ratio)
rr_ratio = None
if sl is not None and tp is not None:
    risk_distance = abs(price - sl)
    reward_distance = abs(tp - price)
    rr_ratio = round(reward_distance / risk_distance, 2) if risk_distance > 0 else None

# ----------------------- Notification: high R:R with high confidence -----------------------
# Show a prominent warning/notification if conditions met (1:6 or 1:9 and confidence >= 110)
if rr_ratio is not None and confidence >= 110 and (rr_ratio >= 6 or rr_ratio >= 9):
    # show a big warning box on top of UI (keeps interface unchanged otherwise)
    st.warning(f"🚨 High R:R Opportunity detected: R:R = {rr_ratio} with Confidence = {confidence}% — check trade carefully!")

# ----------------------- UI Display (keeps interface look) -----------------------
st.subheader("🔍 Signal Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Price", f"{price:.6f}")
col2.metric("EMA20", f"{ema20:.6f}")
col3.metric("RSI(14)", f"{rsi14:.2f}")

st.markdown(f"**Trend:** {trend}   •   **Signal:** {signal}   •   **Confidence:** {confidence}%")
if reasons:
    st.markdown("**Reasons:** " + ", ".join(reasons))
if ht_confirm:
    st.markdown("*Higher timeframe confirmed trend.*")

if sl is not None and tp is not None:
    st.info(f"Entry: {price:.6f}  •  SL: {sl:.6f}  •  TP: {tp:.6f}  •  R:R: {rr_ratio}  •  Risk ${risk_amount:.2f}  •  Lot Size ≈ {lot_size}")
else:
    st.info("No SL/TP (no actionable signal)")

# ----------------------- Save signal to CSV (history) -----------------------
hist_file = "signal_history.csv"
if st.button("📥 Save Signal to History"):
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Market": selected_market,
        "Signal": signal,
        "Entry": price,
        "SL": sl if sl is not None else "",
        "TP": tp if tp is not None else "",
        "R:R": rr_ratio if rr_ratio is not None else "",
        "Risk($)": round(risk_amount, 2),
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

# ----------------------- Signal history viewer & mark result (unchanged UI) -----------------------
st.subheader("📜 Signal History (Last 50)")
if os.path.exists(hist_file):
    hist_df = pd.read_csv(hist_file)
    st.dataframe(hist_df.tail(50).reset_index(drop=True))
    idx = st.number_input("Enter history row number (0..n-1) to mark result", min_value=0, value=0, step=1)
    action = st.selectbox("Mark selected row as", ["No change", "Success", "Failure"])
    if st.button("Update History Row Result"):
        dfh = pd.read_csv(hist_file)
        if 0 <= idx < len(dfh):
            dfh.at[idx, "Result"] = "Success" if action == "Success" else ("Failure" if action == "Failure" else dfh.at[idx, "Result"])
            dfh.to_csv(hist_file, index=False)
            st.success("Updated history file.")
        else:
            st.error("Index out of range.")
else:
    st.info("No signal history yet. Save signals to populate history.")

# ----------------------- Footer & auto-refresh (keeps behaviour) -----------------------
st.markdown("---")
st.caption("Disclaimer: For educational/analysis use only. Always test & confirm signals on your broker platform before placing real trades.")

if refresh_seconds > 0:
    st.experimental_rerun()
