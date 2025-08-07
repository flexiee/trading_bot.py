# Step 2: Basic Trading Bot with Simulated Live Price Feed
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="Live Price Trading Bot", layout="wide")
st.title("📊 Trading Bot – Live Price (Simulated)")

# Sidebar - Market Selector
markets = ["EUR/USD", "GBP/JPY", "USD/JPY", "BTC/USD", "XAU/USD"]
selected_market = st.selectbox("Select Market", markets)

# Initialize session state
if "price" not in st.session_state:
    st.session_state.price = 100.0  # Starting price

# Live Price Simulator
price_container = st.empty()
signal_container = st.empty()

# Live update loop
run_live = st.checkbox("🔄 Auto Update Live Price", value=True)

if run_live:
    for i in range(200):  # Runs for ~200 updates
        price_change = np.random.uniform(-0.3, 0.3)  # Simulated fluctuation
        st.session_state.price += price_change
        st.session_state.price = round(st.session_state.price, 4)

        # Signal logic
        signal = "BUY" if price_change > 0 else "SELL"

        # Update display
        price_container.metric(label=f"📈 Live Price: {selected_market}", value=f"${st.session_state.price}")
        signal_container.success(f"Signal: {signal} | Time: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(1)
else:
    st.warning("Auto-update paused. Enable checkbox to simulate live data.")

st.caption("⚙️ Simulated live market data. Next: real indicators, chart, and automation.")
