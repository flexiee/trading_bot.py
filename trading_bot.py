import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Trading Bot with RSI & EMA", layout="wide")
st.title("📉 Trading Bot – RSI & EMA Strategy (Simulated Data)")

# Market selection
markets = ["EUR/USD", "GBP/JPY", "USD/JPY", "BTC/USD", "XAU/USD"]
selected_market = st.selectbox("Select Market", markets)

# Generate Simulated OHLC Data
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

# RSI calculation
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# EMA calculation
def calculate_ema(series, period=14):
    return series.ewm(span=period, adjust=False).mean()

df["RSI"] = calculate_rsi(df["close"])
df["EMA"] = calculate_ema(df["close"])

# Generate signal
latest = df.iloc[-1]
signal = ""
if latest["RSI"] < 30 and latest["close"] > latest["EMA"]:
    signal = "📈 BUY"
elif latest["RSI"] > 70 and latest["close"] < latest["EMA"]:
    signal = "📉 SELL"
else:
    signal = "⚠️ WAIT"

# Display metrics
st.subheader(f"📊 {selected_market} – Signal & Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("RSI", f"{latest['RSI']:.2f}")
col2.metric("EMA", f"{latest['EMA']:.4f}")
col3.metric("Price", f"{latest['close']:.4f}")

st.success(f"🔔 Signal: {signal}")

# Show table
with st.expander("Show Candlestick Data"):
    st.dataframe(df[["datetime", "open", "high", "low", "close", "RSI", "EMA"]].tail(10), use_container_width=True)

st.caption("✅ RSI & EMA strategy applied. Next: Add real chart view + risk management.")
