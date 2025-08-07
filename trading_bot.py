import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Trading Bot", layout="wide")
st.title("🤖 Trading Bot – RSI & EMA Strategy with Risk Management")

# Select Market
markets = ["EUR/USD", "GBP/JPY", "USD/JPY", "BTC/USD", "XAU/USD"]
selected_market = st.selectbox("📌 Select Market", markets)

# Account Settings
st.sidebar.header("💼 Account & Risk Settings")
account_balance = st.sidebar.number_input("Account Balance ($)", value=1000.0, step=10.0)
risk_percent = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)

# Simulate OHLC Data
@st.cache_data(ttl=60)
def generate_data():
    now = datetime.now()
    times = [now - timedelta(minutes=i) for i in range(60)][::-1]
    prices = [100 + np.random.normal(0, 0.5) for _ in range(60)]
    df = pd.DataFrame({"datetime": times, "close": prices})
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) + np.random.uniform(0.1, 0.4, size=60)
    df["low"] = df[["open", "close"]].min(axis=1) - np.random.uniform(0.1, 0.4, size=60)
    return df

df = generate_data()

# Indicators
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, period=14):
    return series.ewm(span=period, adjust=False).mean()

df["RSI"] = calculate_rsi(df["close"])
df["EMA"] = calculate_ema(df["close"])

# Signal Logic
latest = df.iloc[-1]
entry_price = latest["close"]
rr_ratio = 2  # Risk:Reward 1:2
signal = ""
sl = tp = lot_size = confidence = None

if latest["RSI"] < 30 and latest["close"] > latest["EMA"]:
    signal = "📈 BUY"
    sl = entry_price - 0.5
    tp = entry_price + (entry_price - sl) * rr_ratio
    confidence = 80
elif latest["RSI"] > 70 and latest["close"] < latest["EMA"]:
    signal = "📉 SELL"
    sl = entry_price + 0.5
    tp = entry_price - (sl - entry_price) * rr_ratio
    confidence = 85
else:
    signal = "⚠️ WAIT"
    confidence = 50

# Risk Management
risk_amount = (risk_percent / 100) * account_balance
pip_value = 10 if "USD" in selected_market else 1  # Rough estimate for forex vs crypto
if sl:
    stop_loss_pips = abs(entry_price - sl)
    lot_size = risk_amount / (stop_loss_pips * pip_value)

# Display Signal & Metrics
st.subheader(f"📊 {selected_market} – Signal Overview")
col1, col2, col3 = st.columns(3)
col1.metric("RSI", f"{latest['RSI']:.2f}")
col2.metric("EMA", f"{latest['EMA']:.4f}")
col3.metric("Price", f"{entry_price:.4f}")
st.success(f"🔔 Signal: {signal}  |  Confidence: {confidence}%")

if sl and tp:
    st.info(f"""
    💰 Entry: {entry_price:.4f}  
    📉 SL: {sl:.4f}  
    🎯 TP: {tp:.4f}  
    ⚖️ Risk:Reward = 1:{rr_ratio}  
    💵 Risk Amount: ${risk_amount:.2f}  
    📦 Lot Size: {lot_size:.2f}
    """)

# Candlestick Chart
st.subheader("📈 Live Candlestick Chart")
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["datetime"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="Candles"
))
fig.add_trace(go.Scatter(
    x=df["datetime"],
    y=df["EMA"],
    mode="lines",
    name="EMA",
    line=dict(color="blue", width=1)
))
if signal in ["📈 BUY", "📉 SELL"]:
    fig.add_hline(y=entry_price, line=dict(color="orange", dash="dash"), annotation_text="Entry")
    fig.add_hline(y=tp, line=dict(color="green", dash="dot"), annotation_text="TP")
    fig.add_hline(y=sl, line=dict(color="red", dash="dot"), annotation_text="SL")

fig.update_layout(height=500, margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

with st.expander("🔍 Show Latest Data"):
    st.dataframe(df[["datetime", "open", "high", "low", "close", "RSI", "EMA"]].tail(10), use_container_width=True)

st.caption("✅ Risk-managed trading signal. Next: Telegram alerts, signal log, multi-timeframe confirmation.")
