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
# Function to calculate signals with multiple strategy confirmation
def generate_signal(data):
    if data is None or data.empty:
        return "No data", 0, None

    # --- Strategy 1: RSI + EMA Crossover ---
    data['EMA_fast'] = ta.trend.ema_indicator(data['close'], window=9)
    data['EMA_slow'] = ta.trend.ema_indicator(data['close'], window=21)
    data['RSI'] = ta.momentum.rsi(data['close'], window=14)

    ema_buy = data['EMA_fast'].iloc[-1] > data['EMA_slow'].iloc[-1]
    ema_sell = data['EMA_fast'].iloc[-1] < data['EMA_slow'].iloc[-1]
    rsi_buy = data['RSI'].iloc[-1] < 30
    rsi_sell = data['RSI'].iloc[-1] > 70

    # --- Strategy 2: MACD Confirmation ---
    macd = ta.trend.macd_diff(data['close'])
    macd_buy = macd.iloc[-1] > 0
    macd_sell = macd.iloc[-1] < 0

    # --- Strategy 3: Bollinger Band Bounce ---
    bb_high = ta.volatility.bollinger_hband(data['close'], window=20)
    bb_low = ta.volatility.bollinger_lband(data['close'], window=20)
    bb_buy = data['close'].iloc[-1] <= bb_low.iloc[-1]
    bb_sell = data['close'].iloc[-1] >= bb_high.iloc[-1]

    # --- Combine Strategies ---
    buy_confidence = sum([ema_buy, rsi_buy, macd_buy, bb_buy])
    sell_confidence = sum([ema_sell, rsi_sell, macd_sell, bb_sell])

    if buy_confidence >= 3:
        return "BUY", buy_confidence * 25, data
    elif sell_confidence >= 3:
        return "SELL", sell_confidence * 25, data
    else:
        return "NO TRADE", max(buy_confidence, sell_confidence) * 25, data


# Signal button click event
if st.button("Generate Signal"):
    df = get_market_data(selected_symbol, selected_interval)
    signal, confidence, df = generate_signal(df)

    if signal != "No data":
        st.subheader(f"📊 Signal: {signal}")
        st.write(f"✅ Confidence Level: {confidence}%")
        st.write("📈 Chart below:")

        # Plot chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        fig.update_layout(title=f"{selected_symbol} - {selected_interval}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available for this symbol and interval.")

# ===================== END OF PART 2 =====================
# ===================== PART 3 =====================

# Pip value mapping (approximate per market type)
PIP_VALUES = {
    "forex": 10,        # $10 per pip for 1 lot
    "crypto": 1,        # $1 per pip for 1 lot
    "indices": 0.5,     # $0.5 per pip for 1 lot
    "commodities": 1.5  # $1.5 per pip for 1 lot
}

# Function to detect market type
def detect_market_type(symbol):
    if any(curr in symbol.upper() for curr in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]):
        return "forex"
    elif any(coin in symbol.upper() for coin in ["BTC", "ETH", "XRP", "DOGE", "SOL"]):
        return "crypto"
    elif any(idx in symbol.upper() for idx in ["NIFTY", "BANKNIFTY", "SENSEX", "DOW", "SPX", "NASDAQ"]):
        return "indices"
    else:
        return "commodities"

# Risk management calculation
def calculate_risk(account_balance, risk_percent, stop_loss_pips, symbol):
    market_type = detect_market_type(symbol)
    pip_value = PIP_VALUES[market_type]

    risk_amount = (account_balance * risk_percent) / 100
    lot_size = risk_amount / (stop_loss_pips * pip_value)

    # Take Profit = Risk:Reward ratio (e.g., 1:2)
    tp_pips = stop_loss_pips * 2
    return {
        "risk_amount": round(risk_amount, 2),
        "lot_size": round(lot_size, 2),
        "stop_loss_pips": stop_loss_pips,
        "take_profit_pips": tp_pips
    }

# User inputs for account balance and risk %
account_balance = st.number_input("💰 Account Balance ($)", value=1000.0, step=100.0)
risk_percent = st.slider("📉 Risk per Trade (%)", 0.5, 5.0, 1.0)

if st.button("Generate Signal with Risk Management"):
    df = get_market_data(selected_symbol, selected_interval)
    signal, confidence, df = generate_signal(df)

    if signal != "No data":
        stop_loss_pips = 20  # Can be dynamic based on volatility
        rm = calculate_risk(account_balance, risk_percent, stop_loss_pips, selected_symbol)

        st.subheader(f"📊 Signal: {signal}")
        st.write(f"✅ Confidence Level: {confidence}%")
        st.write(f"💵 Risk Amount: ${rm['risk_amount']}")
        st.write(f"📏 Lot Size: {rm['lot_size']} lots")
        st.write(f"⛔ Stop Loss: {rm['stop_loss_pips']} pips")
        st.write(f"🎯 Take Profit: {rm['take_profit_pips']} pips")
        st.write("📈 Chart below:")

        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        fig.update_layout(title=f"{selected_symbol} - {selected_interval}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available for this symbol and interval.")

# ===================== END OF PART 3 =====================
