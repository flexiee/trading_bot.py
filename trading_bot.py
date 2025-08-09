import streamlit as st
import pandas as pd
import requests
import datetime
import numpy as np

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(page_title="Pro Trading Bot", layout="wide")

# ------------------ HEADER ------------------ #
st.markdown("<h1 style='text-align:center; color:gold;'>📈 Pro Trading Bot</h1>", unsafe_allow_html=True)

# ------------------ MARKET LIST ------------------ #
markets = {
    "EUR/USD": "OANDA:EURUSD",
    "GBP/JPY": "OANDA:GBPJPY",
    "USD/JPY": "OANDA:USDJPY",
    "XAU/USD": "OANDA:XAUUSD",
    "BTC/USD": "BINANCE:BTCUSDT"
}

selected_market = st.selectbox("Select Market", list(markets.keys()))

# ------------------ TRADINGVIEW CHART ------------------ #
tradingview_symbol = markets[selected_market]

st.markdown(f"""
<iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{tradingview_symbol}&symbol={tradingview_symbol}&interval=1&hidesidetoolbar=1&symboledit=1&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc/UTC"
width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
""", unsafe_allow_html=True)

# ------------------ SIMPLE RSI + EMA STRATEGY ------------------ #
def get_ohlcv(symbol):
    """Fetch OHLCV data from TradingView's unofficial API via tvdatafeed-like source."""
    # This is just a mock - real API needs tvdatafeed or broker API
    np.random.seed(int(datetime.datetime.now().timestamp()))
    prices = np.random.uniform(1.05, 1.15, 50)  # Random price simulation
    df = pd.DataFrame({
        "close": prices
    })
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period=9):
    return series.ewm(span=period, adjust=False).mean()

# Get mock OHLCV data
df = get_ohlcv(selected_market)

# Calculate indicators
df['RSI'] = calculate_rsi(df['close'])
df['EMA'] = calculate_ema(df['close'])

# Generate basic signal
latest_rsi = df['RSI'].iloc[-1]
latest_price = df['close'].iloc[-1]
latest_ema = df['EMA'].iloc[-1]

if latest_rsi < 30 and latest_price > latest_ema:
    signal = "📈 BUY"
elif latest_rsi > 70 and latest_price < latest_ema:
    signal = "📉 SELL"
else:
    signal = "⏳ WAIT"

# ------------------ DISPLAY SIGNAL ------------------ #
st.subheader("Trading Signal")
st.markdown(f"<h2 style='color: cyan;'>{signal}</h2>", unsafe_allow_html=True)

st.write(f"**RSI:** {latest_rsi:.2f}")
st.write(f"**Price:** {latest_price:.5f}")
st.write(f"**EMA:** {latest_ema:.5f}")
