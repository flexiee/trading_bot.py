# Basic Trading Bot - Step 1
import streamlit as st
import pandas as pd
from datetime import datetime

# Title
st.set_page_config(page_title="Basic Trading Bot", layout="wide")
st.title("📈 Basic Trading Bot (Step 1)")

# Sample Market List
markets = ["EUR/USD", "GBP/JPY", "XAU/USD", "BTC/USD"]
selected_market = st.selectbox("Select Market", markets)

# Dummy Data (This will be replaced later)
current_price = 100.0  # Simulated live price
signal = "BUY" if current_price % 2 == 0 else "SELL"

# Display Output
st.metric(label=f"Live Price of {selected_market}", value=f"${current_price}")
st.success(f"Signal: {signal}")

st.caption(f"🕒 Last updated at {datetime.now().strftime('%H:%M:%S')}")

# Done ✅
