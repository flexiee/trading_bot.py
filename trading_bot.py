# trading_bot.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import pytz
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval
import os

# ==================== CONFIG =========================

MARKET_SYMBOLS = {
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "XAU/USD": ("OANDA", "XAUUSD"),
    "BTC/USD": ("BINANCE", "BTCUSDT"),
    "ETH/USD": ("BINANCE", "ETHUSDT"),
    "BNB/USD": ("BINANCE", "BNBUSDT"),
    "BANKNIFTY": ("NSE", "BANKNIFTY"),
    "NIFTY 50": ("NSE", "NIFTY"),
    "SENSEX": ("BSE", "SENSEX"),
}

CATEGORY_MAP = {
    "Forex": ["EUR/USD", "GBP/JPY", "USD/JPY", "AUD/USD", "XAU/USD"],
    "Crypto": ["BTC/USD", "ETH/USD", "BNB/USD"],
    "Indices": ["BANKNIFTY", "NIFTY 50", "SENSEX"],
}

SIGNAL_HISTORY_FILE = "signal_history.csv"

# ==================== FUNCTIONS =========================

@st.cache_resource
def get_tv_connection():
    return TvDatafeed()

def get_data(symbol, exchange, interval=Interval.in_5_minute, n_bars=200):
    tv = get_tv_connection()
    data = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
    return data.reset_index()

def calculate_indicators(df):
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['RSI'] = compute_rsi(df['close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_trend(df):
    if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]:
        return 'uptrend'
    elif df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1]:
        return 'downtrend'
    else:
        return 'sideways'

def detect_momentum(df):
    rsi = df['RSI'].iloc[-1]
    if rsi > 70:
        return 'overbought'
    elif rsi < 30:
        return 'oversold'
    else:
        return 'neutral'
def detect_candlestick_pattern(df):
    last_candle = df.iloc[-1]
    body = abs(last_candle['close'] - last_candle['open'])
    range_ = last_candle['high'] - last_candle['low']

    if body < (0.3 * range_):
        return "Doji"
    elif last_candle['close'] > last_candle['open'] and body > (0.7 * range_):
        return "Bullish Engulfing"
    elif last_candle['open'] > last_candle['close'] and body > (0.7 * range_):
        return "Bearish Engulfing"
    else:
        return "None"

def calculate_volatility(df):
    return np.std(df['close'].pct_change().dropna()) * 100000

def signal_strength_logic(trend, momentum, pattern):
    strength = 0
    if trend in ['uptrend', 'downtrend']:
        strength += 40
    if momentum in ['overbought', 'oversold']:
        strength += 30
    if pattern in ['Bullish Engulfing', 'Bearish Engulfing']:
        strength += 30
    return min(strength, 100)

def generate_signal(df, market, account_balance, risk_percent):
    df = calculate_indicators(df)
    trend = detect_trend(df)
    momentum = detect_momentum(df)
    pattern = detect_candlestick_pattern(df)
    volatility = calculate_volatility(df)

    # Base signal
    signal = "WAIT"
    confidence = signal_strength_logic(trend, momentum, pattern)

    # Entry logic
    price = df['close'].iloc[-1]
    sl = None
    tp = None
    reasons = []

    if confidence >= 70:
        if trend == "uptrend" and momentum != "overbought":
            signal = "BUY"
            sl = price - (volatility * 0.5)
            tp = price + (volatility * 1.5)
            reasons.append("EMA crossover & bullish confirmation")
        elif trend == "downtrend" and momentum != "oversold":
            signal = "SELL"
            sl = price + (volatility * 0.5)
            tp = price - (volatility * 1.5)
            reasons.append("EMA crossover & bearish confirmation")
        else:
            reasons.append("Conditions mixed, no clear entry")
    else:
        reasons.append("Low confidence, no trade")

    # Risk Management
    risk_amount = account_balance * (risk_percent / 100)
    if sl and signal != "WAIT":
        pip_risk = abs(price - sl)
        lot_size = risk_amount / pip_risk if pip_risk != 0 else 0
        rr_ratio = abs(tp - price) / pip_risk if pip_risk != 0 else 0
    else:
        lot_size = 0
        rr_ratio = None

    return {
        "signal": signal,
        "confidence": confidence,
        "entry": price,
        "sl": sl,
        "tp": tp,
        "risk": round(risk_amount, 2),
        "lot_size": round(lot_size, 2),
        "rr_ratio": round(rr_ratio, 2) if rr_ratio else "None",
        "reasons": reasons,
        "trend": trend,
        "momentum": momentum,
        "pattern": pattern,
        "volatility": int(volatility)
    }
def log_signal_to_history(signal_data, market):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {
        "Time": now,
        "Market": market,
        "Signal": signal_data["signal"],
        "Confidence": signal_data["confidence"],
        "Entry": signal_data["entry"],
        "SL": signal_data["sl"],
        "TP": signal_data["tp"],
        "Lot Size": signal_data["lot_size"],
        "R:R": signal_data["rr_ratio"],
        "Reasons": "; ".join(signal_data["reasons"]),
        "Result": "Pending"
    }
    history_df = pd.read_csv("signal_history.csv") if os.path.exists("signal_history.csv") else pd.DataFrame(columns=new_row.keys())
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    history_df.to_csv("signal_history.csv", index=False)

def update_signal_result(index, result):
    if os.path.exists("signal_history.csv"):
        df = pd.read_csv("signal_history.csv")
        if 0 <= index < len(df):
            df.at[index, "Result"] = result
            df.to_csv("signal_history.csv", index=False)

def calculate_success_rate():
    if os.path.exists("signal_history.csv"):
        df = pd.read_csv("signal_history.csv")
        total = len(df)
        wins = len(df[df["Result"] == "Success"])
        rate = (wins / total) * 100 if total > 0 else 0
        return f"{wins}/{total} ({rate:.1f}%)"
    return "0/0 (0%)"

def main():
    st.title("📈 Pro Real-Time Trading Bot v2 (Manual Execution)")

    st.sidebar.image("https://i.ibb.co/CKMzKpV/chart.png", use_column_width=True)
    account_balance = st.sidebar.number_input("Account Balance", min_value=10.0, value=1000.0, step=10.0)
    risk_percent = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, step=0.5)

    selected_market = st.selectbox("Select Market", list(MARKETS.keys()))
    exchange, symbol = MARKETS[selected_market]

    interval = Interval.in_1_minute
    bars = 500

    with st.spinner(f"Fetching data for {selected_market}..."):
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=bars)

    if df is not None and not df.empty:
        df.reset_index(inplace=True)
        df.rename(columns={"datetime": "Date", "close": "close", "open": "open", "high": "high", "low": "low", "volume": "volume"}, inplace=True)

        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        fig.update_layout(title=f"{selected_market} - Live Market Chart", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        signal_data = generate_signal(df, selected_market, account_balance, risk_percent)
        if signal_data:
            st.subheader("📊 Signal Output")
            st.markdown(f"""
            **Signal:** {signal_data["signal"]}  
            **Confidence:** {signal_data["confidence"]}  
            **Market Movement:** {signal_data["movement"]}  
            **Trend Direction:** {signal_data["trend"]}  
            **Reasons:** {"; ".join(signal_data["reasons"])}  
            """)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry", f"{signal_data['entry']:.4f}")
            with col2:
                st.metric("Stop Loss", f"{signal_data['sl']:.4f}")
            with col3:
                st.metric("Take Profit", f"{signal_data['tp']:.4f}")

            col4, col5 = st.columns(2)
            with col4:
                st.metric("Lot Size", f"{signal_data['lot_size']}")
            with col5:
                st.metric("Risk:Reward", f"{signal_data['rr_ratio']}")

            if signal_data["rr_ratio"] >= 6 and signal_data["confidence"] == "110%":
                st.success("🚨 High R:R Opportunity! Possible 1:6+ or 1:9+ Setup")

            log_signal_to_history(signal_data, selected_market)
        else:
            st.warning("No strong signal detected. Try again later.")

    else:
        st.error("Could not fetch data. Please check symbol or try again later.")

    with st.expander("📜 Signal History"):
        if os.path.exists("signal_history.csv"):
            hist_df = pd.read_csv("signal_history.csv")
            st.dataframe(hist_df.tail(20), use_container_width=True)
        else:
            st.info("No signal history found.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"✅ Win Rate: {calculate_success_rate()}")

if __name__ == "__main__":
    main()
