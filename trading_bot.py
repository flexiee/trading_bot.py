import streamlit as st
import pandas as pd
import numpy as np
import datetime
from tvDatafeed import TvDatafeed, Interval
import os

# ---------------------------
# INIT TRADINGVIEW CONNECTION
# ---------------------------
tv = TvDatafeed()

# ---------------------------
# MARKET SYMBOLS
# ---------------------------
MARKET_SYMBOLS = {
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "BTC/USD": ("BINANCE", "BTCUSDT"),
    "XAU/USD": ("OANDA", "XAUUSD")
}

# ---------------------------
# TIMEFRAME MAPPING
# ---------------------------
TIMEFRAME_MAP = {
    "1 Minute": Interval.in_1_minute,
    "5 Minutes": Interval.in_5_minute,
    "15 Minutes": Interval.in_15_minute,
    "1 Hour": Interval.in_1_hour,
    "4 Hours": Interval.in_4_hour,
    "1 Day": Interval.in_daily
}

# ---------------------------
# STRATEGY FUNCTION
# ---------------------------
def generate_signal(df, risk_percent, balance):
    """Generate trading signal using multi-strategy confirmation."""
    try:
        df["EMA50"] = df["close"].ewm(span=50).mean()
        df["EMA200"] = df["close"].ewm(span=200).mean()

        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["Signal_Line"] = df["MACD"].ewm(span=9).mean()

        latest = df.iloc[-1]
        lot_size = round(((balance * (risk_percent / 100)) / (0.001 * latest.close)), 2)
        sl = round(latest.close - (latest.close * 0.002), 5)
        tp = round(latest.close + (latest.close * 0.004), 5)

        signal = "WAIT"
        confidence = 0

        if latest.EMA50 > latest.EMA200 and latest.RSI > 55 and latest.MACD > latest.Signal_Line:
            signal = "BUY"
            confidence = 90
        elif latest.EMA50 < latest.EMA200 and latest.RSI < 45 and latest.MACD < latest.Signal_Line:
            signal = "SELL"
            confidence = 90

        return {
            "Signal": signal,
            "Price": round(latest.close, 5),
            "SL": sl,
            "TP": tp,
            "Lot Size": lot_size,
            "Confidence": confidence
        }

    except Exception as e:
        return {"Signal": "ERROR", "Error": str(e)}

# ---------------------------
# SIGNAL HISTORY LOGGER
# ---------------------------
def log_signal(market, timeframe, signal_data):
    file_path = "signal_history.csv"
    entry = {
        "Time": datetime.datetime.now(),
        "Market": market,
        "Timeframe": timeframe,
        **signal_data
    }
    df_entry = pd.DataFrame([entry])
    if os.path.exists(file_path):
        df_entry.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df_entry.to_csv(file_path, index=False)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Pro Trading Bot", layout="wide")

st.title("📈 Pro Trading Bot")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    market = st.selectbox("Select Market", list(MARKET_SYMBOLS.keys()))
with col2:
    timeframe_label = st.selectbox("Select Timeframe", list(TIMEFRAME_MAP.keys()))
with col3:
    balance = st.number_input("Account Balance ($)", value=1000.0)

risk_percent = 1.0  # Default risk = 1%

if st.button("🚀 Generate Signal"):
    try:
        symbol, exchange = MARKET_SYMBOLS[market]
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=TIMEFRAME_MAP[timeframe_label], n_bars=500)

        if df is None or df.empty:
            st.error("No data fetched. Try another market/timeframe.")
        else:
            signal_data = generate_signal(df, risk_percent, balance)
            log_signal(market, timeframe_label, signal_data)
            st.subheader(f"Signal: {signal_data['Signal']}")
            st.write(f"Price: {signal_data['Price']}")
            st.write(f"SL: {signal_data['SL']}")
            st.write(f"TP: {signal_data['TP']}")
            st.write(f"Lot Size: {signal_data['Lot Size']}")
            st.write(f"Confidence: {signal_data['Confidence']}%")
    except Exception as e:
        st.error(f"Error: {e}")
