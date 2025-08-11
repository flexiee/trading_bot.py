import streamlit as st
import pandas as pd
import base64
import requests
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval

# =======================
# CONFIGURATION
# =======================
st.set_page_config(page_title="Pro Trading Bot", layout="wide")

# Replace with your base64 background image
BULL_BEAR_BASE64 = "PUT-YOUR-BASE64-IMAGE-HERE"

# Telegram alert setup
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# =======================
# STYLE - Background
# =======================
page_bg_img = f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{BULL_BEAR_BASE64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# =======================
# TELEGRAM ALERT FUNCTION
# =======================
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.get(url, params=params)
    except:
        pass  # Avoid stopping the bot if Telegram fails

# =======================
# ENGULFING DETECTION
# =======================
def is_bullish_engulfing(df):
    for i in range(1, len(df)):
        prev_o, prev_c = df['open'].iloc[i-1], df['close'].iloc[i-1]
        o, c = df['open'].iloc[i], df['close'].iloc[i]
        if (prev_c < prev_o) and (c > o) and (c > prev_o) and (o < prev_c):
            return True
    return False

def is_bearish_engulfing(df):
    for i in range(1, len(df)):
        prev_o, prev_c = df['open'].iloc[i-1], df['close'].iloc[i-1]
        o, c = df['open'].iloc[i], df['close'].iloc[i]
        if (prev_c > prev_o) and (c < o) and (c < prev_o) and (o > prev_c):
            return True
    return False

# =======================
# STRATEGY FUNCTION
# =======================
def generate_signal(df):
    df['EMA_9'] = df['close'].ewm(span=9).mean()
    df['EMA_21'] = df['close'].ewm(span=21).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    last_row = df.iloc[-1]

    if last_row['EMA_9'] > last_row['EMA_21'] and last_row['RSI'] > 50 and is_bullish_engulfing(df):
        return "BUY"
    elif last_row['EMA_9'] < last_row['EMA_21'] and last_row['RSI'] < 50 and is_bearish_engulfing(df):
        return "SELL"
    return "NO TRADE"

# =======================
# RISK/REWARD CALCULATION
# =======================
def calculate_risk_reward(balance, risk_percent, entry, sl, tp):
    risk_amount = balance * (risk_percent / 100)
    pip_value = abs(entry - sl)
    lot_size = risk_amount / pip_value if pip_value != 0 else 0
    rr_ratio = abs((tp - entry) / (entry - sl)) if (entry - sl) != 0 else 0
    return lot_size, risk_amount, rr_ratio

# =======================
# TRADINGVIEW CHART
# =======================
def get_tradingview_widget(symbol, exchange):
    return f"""
    <iframe src="https://s.tradingview.com/widgetembed/?symbol={exchange}:{symbol}&interval=1&theme=dark"
            width="100%" height="500" frameborder="0" allowfullscreen></iframe>
    """

# =======================
# MAIN UI
# =======================
st.title("📈 Pro Trading Bot with Multi-Confirmation Strategy")

# TV Datafeed Login (Public mode)
tv = TvDatafeed()

MARKETS = {
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "XAU/USD": ("OANDA", "XAUUSD"),
    "BTC/USD": ("BINANCE", "BTCUSDT"),
}

market_choice = st.selectbox("Select Market", list(MARKETS.keys()))
balance = st.number_input("Account Balance ($)", min_value=50.0, value=1000.0)
risk_percent = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)

# Fetch data
exchange, symbol = MARKETS[market_choice]
df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=200)

if df is not None and not df.empty:
    df = df.reset_index()
    signal = generate_signal(df)

    st.markdown(get_tradingview_widget(symbol, exchange), unsafe_allow_html=True)

    if signal != "NO TRADE":
        entry_price = df['close'].iloc[-1]
        sl_price = entry_price * (0.998 if signal == "BUY" else 1.002)
        tp_price = entry_price * (1.006 if signal == "BUY" else 0.994)

        lot_size, risk_amt, rr_ratio = calculate_risk_reward(balance, risk_percent, entry_price, sl_price, tp_price)

        st.subheader(f"📢 Signal: **{signal}**")
        st.write(f"Entry: {entry_price:.5f}")
        st.write(f"Stop Loss: {sl_price:.5f}")
        st.write(f"Take Profit: {tp_price:.5f}")
        st.write(f"Lot Size: {lot_size:.2f}")
        st.write(f"Risk Amount: ${risk_amt:.2f}")
        st.write(f"R:R Ratio: {rr_ratio:.2f}")

        if rr_ratio in [6, 9]:  # High R:R notification
            st.warning(f"🚀 High R:R Detected ({rr_ratio} : 1) with 110% Confidence!")
            send_telegram_message(f"🚀 {market_choice} - {signal} - R:R {rr_ratio}")

    else:
        st.info("No valid trade signal right now.")
else:
    st.error("Failed to fetch market data.")
