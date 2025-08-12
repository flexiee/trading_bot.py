# trading_bot_streamlit.py
# Cloud-friendly single-file trading signal app (no Selenium). Copy-paste into your Streamlit app.
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
import io, base64

st.set_page_config(page_title="Pro Signal Bot (Cloud)", layout="wide", initial_sidebar_state="expanded")

# --- Markets & TradingView embeds ---
MARKET_YF = {
    "EUR/USD": "EURUSD=X", "GBP/JPY": "GBPJPY=X", "USD/JPY": "JPY=X", "AUD/USD": "AUDUSD=X",
    "XAU (Gold)": "GC=F", "Silver": "SI=F", "Oil WTI": "CL=F",
    "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD",
    "NIFTY 50": "^NSEI", "S&P 500": "^GSPC"
}
MARKET_TV = {
    "EUR/USD": "OANDA:EURUSD", "GBP/JPY":"OANDA:GBPJPY", "USD/JPY":"OANDA:USDJPY", "AUD/USD":"OANDA:AUDUSD",
    "XAU (Gold)":"OANDA:XAUUSD","Silver":"OANDA:XAGUSD","Oil WTI":"OANDA:WTICOUSD",
    "BTC/USD":"BINANCE:BTCUSDT","ETH/USD":"BINANCE:ETHUSDT","NIFTY 50":"NSE:NIFTY","S&P 500":"INDEX:SPX"
}

# --- Utilities (indicators) ---
def ema(series, n): return series.ewm(span=n, adjust=False).mean()
def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

# --- Fetch data (yfinance) ---
@st.cache_data(ttl=15)
def get_ohlc(yf_symbol, period="1d", interval="1m"):
    try:
        df = yf.download(tickers=yf_symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty: return None
        df = df.dropna().reset_index().rename(columns={"Datetime":"datetime"}) if "Datetime" in df.columns else df.reset_index()
        df.rename(columns={df.columns[0]:"datetime"}, inplace=True)
        return df
    except Exception:
        return None

# --- Signal logic & risk calc ---
def analyze(df):
    df = df.copy()
    df["EMA5"] = ema(df["Close"], 5)
    df["EMA20"] = ema(df["Close"], 20)
    df["RSI14"] = rsi(df["Close"], 14)
    df["returns"] = df["Close"].pct_change()
    vol = df["returns"].std() * 10000 if not df["returns"].isna().all() else 0
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    price = float(last["Close"])
    prev_price = float(prev["Close"])
    trend = "uptrend" if price > df["EMA20"].iloc[-1] else "downtrend"
    momentum = "strong" if abs(price - prev_price) > (df["Close"].std() * 0.5 if df["Close"].std()>0 else 0.0005) else "weak"
    return {"price":price,"prev_price":prev_price,"trend":trend,"momentum":momentum,"volatility":abs(vol),
            "EMA5":float(last["EMA5"]),"EMA20":float(last["EMA20"]), "RSI":float(last["RSI14"])}

def generate_signal(analysis, account_balance, risk_percent, market_name):
    entry = analysis["price"]
    signal = "WAIT"; reasons=[]
    # Basic preserved Rayner-style logic
    if analysis["trend"]=="uptrend" and analysis["momentum"]=="strong":
        sl = entry - (analysis["volatility"]*0.0001 if analysis["volatility"]>0 else 0.001)
        tp = entry + 3*(entry - sl)
        signal="BUY"; reasons.append("Breakout confirmation in uptrend")
    elif analysis["trend"]=="downtrend" and analysis["momentum"]=="strong":
        sl = entry + (analysis["volatility"]*0.0001 if analysis["volatility"]>0 else 0.001)
        tp = entry - 3*(sl - entry)
        signal="SELL"; reasons.append("Breakout confirmation in downtrend")
    else:
        sl, tp = None, None

    # Risk-sizing: approximate per market type
    risk_amount = account_balance * (risk_percent/100.0)
    lot_size = 0
    rr_ratio = None
    if sl and tp:
        risk_per_unit = abs(entry - sl)
        reward_per_unit = abs(tp - entry)
        rr_ratio = round((reward_per_unit / (risk_per_unit + 1e-9)), 2)
        # If Forex: pip-based lot calc assume 1 lot=100000 units; else crypto/indices treat unit-based
        if market_name in ["EUR/USD","GBP/JPY","USD/JPY","AUD/USD"]:
            # approximate pip unit = 0.0001 (JPY pairs 0.01)
            pip_unit = 0.01 if "JPY" in market_name else 0.0001
            units = risk_amount / (risk_per_unit) if risk_per_unit>0 else 0
            lot_size = round(units / 100000, 4)  # standard lots
        else:
            # for crypto/indices use units
            lot_size = round((risk_amount / (risk_per_unit+1e-9)), 6)

    # Confidence: base on volatility & RSI & momentum
    confidence = min(200, int(50 + analysis["volatility"]*0.02 + (20 if analysis["momentum"]=="strong" else 0) + (10 if analysis["RSI"]<30 or analysis["RSI"]>70 else 0)))
    return {"signal":signal,"entry":round(entry,6),"sl":round(sl,6) if sl else None,"tp":round(tp,6) if tp else None,
            "risk_amount":round(risk_amount,2),"lot_size":lot_size,"rr":rr_ratio,"confidence":confidence,"reasons":reasons}

# --- Persist history ---
HISTORY_FILE = "signal_history.csv"
def append_history(row:dict):
    cols = ["timestamp","market","timeframe","signal","entry","sl","tp","confidence","rr","lot","risk_amount","notes"]
    df = pd.DataFrame([row], columns=cols)
    if os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)

def read_history(n=20):
    if not os.path.exists(HISTORY_FILE): return pd.DataFrame(columns=["timestamp","market","timeframe","signal","entry","sl","tp","confidence","rr","lot","risk_amount","notes"])
    df = pd.read_csv(HISTORY_FILE)
    return df.tail(n).reset_index(drop=True)

# --- UI: Sidebar ---
st.sidebar.title("Account & Settings")
account_balance = st.sidebar.number_input("Account Balance ($)", min_value=1.0, value=1000.0, step=10.0, format="%.2f")
risk_percent = st.sidebar.slider("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
tf = st.sidebar.selectbox("Chart timeframe", options=["1m","5m","15m","1h","1d"], index=0)
auto_refresh = st.sidebar.number_input("Auto-refresh (sec, 0=off)", value=0, min_value=0, step=5)
st.sidebar.markdown("Made for manual execution. No auto-trading included.")

# favorites
if "favorites" not in st.session_state: st.session_state.favorites = ["EUR/USD","BTC/USD"]
# high movement (computed later)

# --- Main layout ---
col1, col2 = st.columns([1,1.4])
with col1:
    st.header("📈 Markets")
    market = st.selectbox("Select Market", list(MARKET_YF.keys()), index=0)
    # favorite toggle
    fav_col1, fav_col2 = st.columns([6,1])
    with fav_col1: st.write(f"Selected: **{market}**")
    with fav_col2:
        if st.button("★" if market in st.session_state.favorites else "☆"):
            if market in st.session_state.favorites: st.session_state.favorites.remove(market)
            else: st.session_state.favorites.append(market)

    # TradingView chart embed (visual only)
    tv_sym = MARKET_TV.get(market, "")
    if tv_sym:
        st.components.v1.iframe(f"https://s.tradingview.com/widgetembed/?symbol={tv_sym}&interval={tf}&theme=dark", height=480)
    else:
        st.info("TradingView symbol not available for this market.")

    st.markdown("---")
    if st.button("🔄 Refresh Chart / Data (manual)"):
        st.experimental_rerun()

with col2:
    st.header("🔎 Signal Summary")
    with st.spinner("Fetching data..."):
        yf_sym = MARKET_YF.get(market)
        interval_map = {"1m":"1m","5m":"5m","15m":"15m","1h":"60m","1d":"1d"}
        yf_interval = interval_map.get(tf,"1m")
        # yfinance uses period; for 1m need '1d' period typically
        period_map = {"1m":"1d","5m":"5d","15m":"30d","1h":"90d","1d":"365d"}
        period = period_map.get(tf,"1d")
        df = get_ohlc(yf_sym, period=period, interval=yf_interval)
    if df is None or df.empty:
        st.error("Unable to fetch live OHLC data for the selected market/timeframe. Try again or pick a different timeframe.")
    else:
        analysis = analyze(df)
        st.metric("Price", f"{analysis['price']:.6f}")
        st.write(f"Trend: **{analysis['trend']}**  |  Momentum: **{analysis['momentum']}**  |  Volatility: **{analysis['volatility']:.2f}**")
        st.write(f"EMA5: {analysis['EMA5']:.6f}  EMA20: {analysis['EMA20']:.6f}  RSI: {analysis['RSI']:.2f}")

        if st.button("🔮 Generate Signal"):
            sig = generate_signal(analysis, account_balance, risk_percent, market)
            st.success(f"Signal: {sig['signal']}  |  Confidence: {sig['confidence']}%")
            st.write(f"Entry: {sig['entry']}  |  SL: {sig['sl']}  |  TP: {sig['tp']}")
            st.write(f"Risk ${sig['risk_amount']}  |  Lot Size: {sig['lot_size']}  |  R:R: {sig['rr']}")
            if sig["reasons"]: st.write("Reasons:", ", ".join(sig["reasons"]))
            # Notification for large R:R with high confidence
            if sig["rr"] and sig["rr"] >= 6 and sig["confidence"] >= 110:
                st.balloons()
                st.warning(f"HIGH R:R {sig['rr']} with confidence {sig['confidence']}%")
            # append history
            row = {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "market": market, "timeframe": tf, "signal": sig["signal"],
                "entry": sig["entry"], "sl": sig["sl"], "tp": sig["tp"],
                "confidence": sig["confidence"], "rr": sig["rr"], "lot": sig["lot_size"],
                "risk_amount": sig["risk_amount"], "notes": "|".join(sig["reasons"])
            }
            append_history(row)

    st.markdown("---")
    st.subheader("📋 Recent Signals")
    history = read_history(10)
    st.dataframe(history if not history.empty else pd.DataFrame({"info":["No history yet"]}), use_container_width=True)

# --- Top movers (simple heuristic) ---
try:
    movers = {}
    for m, sym in MARKET_YF.items():
        d = get_ohlc(sym, period="1d", interval="5m")
        if d is None or d.empty: continue
        movers[m] = abs(d["Close"].pct_change().tail(6).sum())
    top = sorted(movers.items(), key=lambda x: x[1], reverse=True)[:5]
    st.sidebar.subheader("🔥 Top Movers")
    for name, score in top:
        st.sidebar.write(f"{name}: {score:.4f}")
except Exception:
    st.sidebar.info("Top movers unavailable")

# --- Download history button ---
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "rb") as f:
        b = f.read()
    st.sidebar.download_button("Download signal_history.csv", data=b, file_name="signal_history.csv", mime="text/csv")

# --- Auto-refresh (simple) ---
if auto_refresh and auto_refresh > 0:
    st.experimental_set_query_params(_refresh=datetime.utcnow().timestamp())
    st.experimental_rerun()
