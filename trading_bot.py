import streamlit as st
import pandas as pd
import random
from datetime import datetime
import base64

# ======== APP CONFIG ========
st.set_page_config(page_title="Pro Trading Bot", layout="wide")

# ======== EMBED BACKGROUND IMAGE (BASE64) ========
BACKGROUND_IMAGE_BASE64 = """
iVBORw0KGgoAAAANSUhEUgAA...
"""  # Replace with your bull/bear image base64 string

page_bg_img = f"""
<style>
.stApp {{
background-image: url("data:image/png;base64,{BACKGROUND_IMAGE_BASE64}");
background-size: cover;
background-position: center;
background-attachment: fixed;
color: white;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ======== MARKET LIST ========
markets = ["EUR/USD", "GBP/JPY", "USD/JPY", "BTC/USD", "XAU/USD", "NIFTY 50", "BANKNIFTY", "SENSEX"]
favorites = st.session_state.get("favorites", [])

# ======== SIDEBAR ========
st.sidebar.header("Market Selection")
selected_market = st.sidebar.selectbox("Choose Market", markets)
if st.sidebar.button("⭐ Add to Favorites"):
    if selected_market not in favorites:
        favorites.append(selected_market)
        st.session_state["favorites"] = favorites

# ======== TRADINGVIEW CHART EMBED ========
def get_tradingview_embed(symbol):
    return f"""
    <iframe src="https://s.tradingview.com/widgetembed/?symbol={symbol}&interval=1&hidesidetoolbar=1&symboledit=1&saveimage=0&toolbarbg=fff&studies=[]&theme=dark&style=1&timezone=Etc%2FUTC"
    width="100%" height="500" frameborder="0" allowfullscreen></iframe>
    """

symbol_map = {
    "EUR/USD": "OANDA:EURUSD",
    "GBP/JPY": "OANDA:GBPJPY",
    "USD/JPY": "OANDA:USDJPY",
    "BTC/USD": "BINANCE:BTCUSDT",
    "XAU/USD": "OANDA:XAUUSD",
    "NIFTY 50": "INDEX:NIFTY",
    "BANKNIFTY": "INDEX:BANKNIFTY",
    "SENSEX": "BSE:SENSEX"
}

# ======== MOCK SIGNAL GENERATION ========
def generate_signal():
    signal = random.choice(["BUY", "SELL", "HOLD"])
    confidence = random.randint(70, 110)
    rr_ratio = random.choice([1.5, 2, 3, 6, 9])
    sl = round(random.uniform(0.001, 0.005), 5)
    tp = sl * rr_ratio
    return signal, confidence, rr_ratio, sl, tp

# ======== DISPLAY FAVORITES WATCHLIST ========
if favorites:
    st.subheader("⭐ Favorites Watchlist")
    for fav in favorites:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(f"**{fav}**")
            price = round(random.uniform(1.0, 2.0), 4)
            st.write(f"Price: {price}")
        with col2:
            st.markdown(get_tradingview_embed(symbol_map[fav]), unsafe_allow_html=True)
    st.divider()

# ======== MAIN CONTENT ========
st.title("📈 Pro Trading Bot")
st.markdown(get_tradingview_embed(symbol_map[selected_market]), unsafe_allow_html=True)

signal, confidence, rr_ratio, sl, tp = generate_signal()

st.subheader(f"Market: {selected_market}")
st.write(f"**Signal:** {signal}")
st.write(f"**Confidence:** {confidence}%")
st.write(f"**Risk/Reward:** 1:{rr_ratio}")
st.write(f"**Stop Loss:** {sl}")
st.write(f"**Take Profit:** {tp}")

# ======== ALERTS ========
if rr_ratio in [6, 9] and confidence >= 110:
    st.warning(f"🚨 High R:R Alert! 1:{rr_ratio} with {confidence}% confidence")

# ======== SIGNAL HISTORY ========
if "history" not in st.session_state:
    st.session_state["history"] = []

if st.button("Save Signal"):
    st.session_state["history"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": selected_market,
        "signal": signal,
        "confidence": confidence,
        "RR": rr_ratio,
        "SL": sl,
        "TP": tp
    })

if st.session_state["history"]:
    st.subheader("📜 Signal History")
    st.dataframe(pd.DataFrame(st.session_state["history"]))
