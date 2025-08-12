import streamlit as st
import pandas as pd
from datetime import datetime
import random

# ===================== CONFIG ===================== #
st.set_page_config(page_title="Pro Trading Bot", layout="wide")

MARKET_SYMBOLS = {
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "BTC/USD": ("BINANCE", "BTCUSDT"),
    "XAU/USD": ("OANDA", "XAUUSD"),
}

# ===================== SIGNAL LOGGING ===================== #
def save_signal_history(market, signal, confidence, rr, sl, tp, lot):
    try:
        log_df = pd.read_csv("signal_history.csv")
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=["Date", "Market", "Signal", "Confidence", "R:R", "SL", "TP", "Lot"])

    log_df = pd.concat([log_df, pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Market": market,
        "Signal": signal,
        "Confidence": confidence,
        "R:R": rr,
        "SL": sl,
        "TP": tp,
        "Lot": lot
    }])], ignore_index=True)

    log_df.to_csv("signal_history.csv", index=False)

# ===================== RISK MANAGEMENT ===================== #
def calculate_lot_size(balance, risk_percent, sl_pips, market):
    if sl_pips <= 0:
        return 0
    risk_amount = balance * (risk_percent / 100)
    pip_value = 10 if "JPY" not in market else 100  # Example adjustment
    lot_size = risk_amount / (sl_pips * pip_value)
    return round(lot_size, 2)

# ===================== STRATEGY SIGNAL ===================== #
def generate_signal(market, timeframe, risk_percent):
    """
    Dummy signal generation (Replace with real market data & strategy).
    Uses random example values for now.
    """
    signal_options = ["BUY", "SELL", "NO TRADE"]
    signal = random.choice(signal_options)
    confidence = random.randint(70, 95) if signal != "NO TRADE" else random.randint(40, 60)
    rr_ratio = random.choice(["1:2", "1:3", "1:4", "1:5"])
    sl = round(random.uniform(10, 50), 2)
    tp = round(sl * int(rr_ratio.split(":")[1]), 2)
    return signal, confidence, rr_ratio, sl, tp

# ===================== UI LAYOUT ===================== #
def main():
    st.title("📊 Pro Real-Time Trading Bot")
    st.markdown("### Advanced Multi-Strategy Trading Bot with Risk Management")

    # Sidebar Settings
    st.sidebar.header("⚙️ Settings")
    selected_market = st.sidebar.selectbox("Select Market", list(MARKET_SYMBOLS.keys()))
    selected_interval = st.sidebar.selectbox("Select Timeframe", ["1", "5", "15", "30", "60", "240", "D"])
    account_balance = st.sidebar.number_input("Account Balance ($)", min_value=10.0, value=1000.0, step=50.0)
    risk_percent = st.sidebar.number_input("Risk % per Trade", min_value=0.1, value=1.0, step=0.1)

    # TradingView Chart (fills empty space)
    if selected_market:
        st.subheader(f"📈 {selected_market} Live Chart")
        market_symbol, market_exchange = MARKET_SYMBOLS[selected_market]
        tv_widget_code = f"""
        <iframe src="https://s.tradingview.com/widgetembed/?symbol={market_exchange}:{market_symbol}
        &interval={selected_interval}
        &theme=dark&style=1&locale=en&utm_source=localhost"
        width="100%" height="550" frameborder="0" allowfullscreen></iframe>
        """
        st.markdown(tv_widget_code, unsafe_allow_html=True)

    # Generate Signal Button
    if st.button("🔍 Generate Signal"):
        try:
            signal, confidence, rr_ratio, sl, tp = generate_signal(selected_market, selected_interval, risk_percent)
            st.success(f"✅ Signal: {signal}")
            st.write(f"📊 Confidence: {confidence}%")
            st.write(f"🎯 Risk/Reward: {rr_ratio}")
            st.write(f"🛑 Stop Loss: {sl}")
            st.write(f"💰 Take Profit: {tp}")

            # Lot size calculation
            lot_size = calculate_lot_size(account_balance, risk_percent, sl, selected_market)
            st.info(f"📦 Recommended Lot Size: {lot_size}")

            # Save to log
            save_signal_history(selected_market, signal, confidence, rr_ratio, sl, tp, lot_size)

        except Exception as e:
            st.error(f"❌ Error while generating signal: {e}")

    # Show signal history
    st.markdown("### 📜 Signal History")
    try:
        log_df = pd.read_csv("signal_history.csv")
        st.dataframe(log_df)
    except FileNotFoundError:
        st.warning("No signals logged yet.")

# ===================== RUN APP ===================== #
if __name__ == "__main__":
    main()
