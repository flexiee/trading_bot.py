# app.py
# Pro Trading Bot vNext — Full-width TradingView chart + Generate Signal button
# - Timeframe selector
# - Default risk = 1%
# - Ensemble strategy (EMA + RSI + MACD + Bollinger + ATR + candlesticks + confirmation)
# - Additional confirmation layer (MACD + engulfing) to increase accuracy
# - Signal history saved to signal_history.csv
#
# Requirements:
#   pip install streamlit pandas numpy yfinance
#
# Run:
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
import os
from datetime import datetime
import streamlit.components.v1 as components

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="Pro Signal Bot", layout="wide")
st.markdown("<style>body {background-color: #0b0e13; color: #dbe6ff;} .stApp {background-color: #0b0e13;}</style>", unsafe_allow_html=True)
st.title("📈 Pro Signal Bot — Master Strategy (Generate Signal)")

# ---------------------------
# Market and symbol mapping
# ---------------------------
# Left: friendly name, value: (yfinance ticker, TradingView symbol for widget)
MARKETS = {
    "EUR/USD": ("EURUSD=X", "OANDA:EURUSD"),
    "GBP/JPY": ("GBPJPY=X", "OANDA:GBPJPY"),
    "USD/JPY": ("JPY=X", "OANDA:USDJPY"),
    "AUD/USD": ("AUDUSD=X", "OANDA:AUDUSD"),
    "XAU/USD (Gold)": ("XAUUSD=X", "OANDA:XAUUSD"),
    "BTC/USD": ("BTC-USD", "BINANCE:BTCUSDT"),
    "ETH/USD": ("ETH-USD", "BINANCE:ETHUSDT"),
}

TIMEFRAME_MAP = {
    "1m": ("1m", "7d"),    # yfinance interval, period
    "5m": ("5m", "30d"),
    "15m": ("15m", "60d"),
    "1h": ("60m", "120d"),
    "4h": ("90m", "180d"),  # yfinance doesn't have 4h; 90m often used as fallback
    "1d": ("1d", "2y"),
}

# ---------------------------
# Sidebar: settings
# ---------------------------
with st.sidebar:
    st.header("Account & Settings")
    account_balance = st.number_input("Account Balance ($)", value=1000.0, step=10.0, format="%.2f")
    risk_percent = st.number_input("Risk per trade (%)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, format="%.2f")
    timeframe = st.selectbox("Chart timeframe", options=list(TIMEFRAME_MAP.keys()), index=0)
    auto_refresh = st.number_input("Auto-refresh every N seconds (0 = off)", value=0, step=1)
    enable_htf = st.checkbox("Enable Higher-Timeframe Confirmation", value=True)
    st.markdown("---")
    st.info("This bot generates signals for manual trading only. No auto-trading included.")

# ---------------------------
# Top controls
# ---------------------------
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    selected_market = st.selectbox("Select Market", list(MARKETS.keys()))
with col2:
    show_history_rows = st.number_input("History rows to show", min_value=5, max_value=200, value=20, step=5)
with col3:
    if st.button("🔄 Refresh Chart / Data (manual)"):
        st.experimental_rerun()

# ---------------------------
# Helpers: indicators & patterns
# ---------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)

def macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def bollinger(series, length=20, mult=2.0):
    ma = series.rolling(length).mean()
    std = series.rolling(length).std().fillna(0)
    upper = ma + std * mult
    lower = ma - std * mult
    width = (upper - lower) / ma.replace(0, np.nan)
    return upper, lower, width.fillna(0)

def atr(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean().fillna(method='bfill')

def is_bullish_engulfing(df):
    if len(df) < 2: return False
    prev_o, prev_c = df['Open'].iloc[-2], df['Close'].iloc[-2]
    o, c = df['Open'].iloc[-1], df['Close'].iloc[-1]
    return (prev_c < prev_o) and (c > o) and (c > prev_o) and (o < prev_c)

def is_bearish_engulfing(df):
    if len(df) < 2: return False
    prev_o, prev_c = df['Open'].iloc[-2], df['Close'].iloc[-2]
    o, c = df['Open'].iloc[-1], df['Close'].iloc[-1]
    return (prev_c > prev_o) and (c < o) and (c < prev_o) and (o > prev_c)

def is_hammer(df):
    if len(df) < 1: return False
    o, c, low, high = df['Open'].iloc[-1], df['Close'].iloc[-1], df['Low'].iloc[-1], df['High'].iloc[-1]
    body = abs(c-o)
    lower_wick = min(o,c) - low
    upper_wick = high - max(o,c)
    return (lower_wick > body * 2.5) and (upper_wick < body)

def is_shooting_star(df):
    if len(df) < 1: return False
    o, c, low, high = df['Open'].iloc[-1], df['Close'].iloc[-1], df['Low'].iloc[-1], df['High'].iloc[-1]
    body = abs(c-o)
    upper_wick = high - max(o,c)
    lower_wick = min(o,c) - low
    return (upper_wick > body * 2.5) and (lower_wick < body)

# ---------------------------
# Fetch OHLC (yfinance)
# ---------------------------
@st.cache_data(ttl=30)
def fetch_ohlc_yf(yf_ticker: str, timeframe_key: str):
    try:
        yf_interval, yf_period = TIMEFRAME_MAP.get(timeframe_key, ("1m", "7d"))
        df = yf.download(tickers=yf_ticker, interval=yf_interval, period=yf_period, progress=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        df.columns = [c.capitalize() for c in df.columns]  # ensure 'Open','High','Low','Close','Volume'
        return df
    except Exception:
        return None

# ---------------------------
# Core strategy (book-informed ensemble)
# ---------------------------
def compute_master_signal(df, market_name, account_balance, risk_pct, enable_htf_confirm=True, timeframe_sel="1m"):
    result = {
        "signal": "WAIT", "confidence": 0, "reasons": [], "entry": None, "sl": None, "tp": None,
        "lot": 0.0, "rr": None, "indicators": {}
    }
    if df is None or df.empty or len(df) < 20:
        return result

    close = df['Close']
    price = float(close.iloc[-1])
    result["entry"] = round(price, 6)

    # Indicators
    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    ema200 = ema(close, 200).iloc[-1]
    rsi14 = float(rsi(close, 14).iloc[-1])
    macd_line, macd_sig, macd_hist = macd(close)
    macd_hist_val = float(macd_hist.iloc[-1])
    bb_u, bb_l, bb_w = bollinger(close, 20, 2.0)
    bb_width = float(bb_w.iloc[-1]) if not math.isnan(bb_w.iloc[-1]) else 0.0
    atrv = float(atr(df, 14).iloc[-1])

    # Base scoring (weights derived from pro practices)
    score = 0
    reasons = []

    # Trend structure: EMA alignment
    if ema20 > ema50 > ema200:
        score += 25; reasons.append("Strong bullish EMA alignment")
        trend = "up"
    elif ema20 < ema50 < ema200:
        score += 25; reasons.append("Strong bearish EMA alignment")
        trend = "down"
    else:
        trend = "side"

    # Momentum: RSI
    if rsi14 < 30:
        score += 20; reasons.append("RSI oversold")
    elif rsi14 > 70:
        score += 20; reasons.append("RSI overbought")
    else:
        if 40 <= rsi14 <= 60:
            score += 5

    # MACD momentum
    if macd_hist_val > 0:
        score += 15; reasons.append("MACD positive")
    else:
        score += 5; reasons.append("MACD negative")

    # Volatility and squeeze detection (Bollinger width)
    if bb_width < 0.01:
        score += 5; reasons.append("Bollinger squeeze (low vol)")
    elif bb_width > 0.02:
        score += 8; reasons.append("Bollinger wide (high vol)")

    # Candlestick confirmations
    if is_bullish_engulfing(df):
        score += 12; reasons.append("Bullish engulfing")
    if is_bearish_engulfing(df):
        score += 12; reasons.append("Bearish engulfing")
    if is_hammer(df):
        score += 6; reasons.append("Hammer")
    if is_shooting_star(df):
        score += 6; reasons.append("Shooting star")

    # HTF confirmation (if enabled)
    htf_confirm = False
    if enable_htf_confirm:
        try:
            # choose higher TF: map 1m->5m, 5m->15m, 15m->1h, 1h->4h, else use 1d
            hmap = {"1m":"5m","5m":"15m","15m":"1h","1h":"4h","4h":"1d","1d":"1d"}
            htf = hmap.get(timeframe, "1h")
            yf_ticker = MARKETS[market_name][0]
            hdf = fetch_ohlc_yf(yf_ticker, htf)
            if hdf is not None and len(hdf) > 20:
                hclose = hdf['Close']
                h_ema20 = ema(hclose,20).iloc[-1]
                if (price > ema20 and hclose.iloc[-1] > h_ema20) or (price < ema20 and hclose.iloc[-1] < h_ema20):
                    score += 10; reasons.append("Higher timeframe confirms")
                    htf_confirm = True
        except Exception:
            htf_confirm = False

    # Final decision logic
    confidence = min(120, int(score))
    result["confidence"] = confidence
    result["reasons"] = reasons
    result["indicators"] = {"ema20": round(ema20,6), "ema50": round(ema50,6), "ema200": round(ema200,6),
                            "rsi14": round(rsi14,2), "macd_hist": round(macd_hist_val,6), "atr": round(atrv,6),
                            "bb_width": round(bb_width,6)}

    # Conditions for buy/sell
    buy_condition = (ema20 > ema50) and (macd_hist_val > 0) and (rsi14 < 65)
    sell_condition = (ema20 < ema50) and (macd_hist_val < 0) and (rsi14 > 35)

    # Extra confirmation layer: require engulfing OR MACD + ATR alignment
    extra_buy_ok = is_bullish_engulfing(df) or (macd_hist_val > 0 and bb_width > 0.005)
    extra_sell_ok = is_bearish_engulfing(df) or (macd_hist_val < 0 and bb_width > 0.005)

    threshold = 60  # minimum confidence to allow a trade
    strong_threshold = 95

    decision = "WAIT"
    if buy_condition and confidence >= threshold and extra_buy_ok:
        decision = "BUY"
    if sell_condition and confidence >= threshold and extra_sell_ok:
        decision = "SELL"

    # Strong override: extreme RSI + MACD alignment
    if rsi14 < 20 and macd_hist_val > 0 and ema20 > ema50:
        decision = "BUY"; confidence = max(confidence, 110); reasons.append("Extreme RSI buy override")
    if rsi14 > 80 and macd_hist_val < 0 and ema20 < ema50:
        decision = "SELL"; confidence = max(confidence, 110); reasons.append("Extreme RSI sell override")

    result["signal"] = decision
    result["confidence"] = confidence

    # ATR-based SL/TP
    atr_factor = 1.5
    sl_distance = atrv * atr_factor if atrv > 0 else max(price * 0.0005, 0.0001)
    if decision == "BUY":
        sl = price - sl_distance
        tp = price + sl_distance * 3
    elif decision == "SELL":
        sl = price + sl_distance
        tp = price - sl_distance * 3
    else:
        sl = None; tp = None

    # Lot sizing (very approximate — adapt per broker). Use conservative pip USD mapping.
    def pip_unit(market_label):
        if "JPY" in market_label: return 0.01
        if "XAU" in market_label: return 0.1
        if "BTC" in market_label or "ETH" in market_label: return 1.0
        return 0.0001

    pip = pip_unit(market_name)
    if sl is not None and sl != price:
        stop_pips = abs(price - sl) / pip if pip != 0 else 0
        pip_value = 10.0  # rough $ per pip per standard lot (conservative)
        risk_amount = (risk_pct / 100.0) * account_balance
        lot = (risk_amount / (stop_pips * pip_value)) if stop_pips > 0 else 0.0
        lot = max(lot, 0.0)
    else:
        lot = 0.0

    result["sl"] = round(sl, 6) if sl is not None else None
    result["tp"] = round(tp, 6) if tp is not None else None
    result["lot"] = round(lot, 6)
    if sl is not None and tp is not None:
        reward = abs(tp - price)
        risk = abs(price - sl) if abs(price - sl) > 0 else 1e-9
        result["rr"] = round(reward / risk, 2)
    else:
        result["rr"] = None

    result["htf_confirm"] = htf_confirm
    return result

# ---------------------------
# Save signal history
# ---------------------------
def save_signal_history(row: dict):
    file_path = "signal_history.csv"
    df_row = pd.DataFrame([row])
    if os.path.exists(file_path):
        df_row.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(file_path, index=False)

# ---------------------------
# TradingView embed (large)
# ---------------------------
def tradingview_embed(tv_symbol: str, interval: str):
    # Build a TradingView widget embed snippet. interval param uses TradingView interval (1,5,15,60,240,D)
    interval_map = {"1m":"1","5m":"5","15m":"15","1h":"60","4h":"240","1d":"D"}
    tv_interval = interval_map.get(interval, "1")
    html = f"""
    <div class="tradingview-widget-container" style="width:100%; height:680px;">
      <div id="tradingview_widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%",
        "height": 680,
        "symbol": "{tv_symbol}",
        "interval": "{tv_interval}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "container_id": "tradingview_widget"
      }});
      </script>
    </div>
    """
    return html

# ---------------------------
# Main layout: Chart (big) + Generate button + Signal output
# ---------------------------
tv_symbol = MARKETS[selected_market][1]
chart_col, output_col = st.columns([3,1])  # big chart area left, small details right

with chart_col:
    st.markdown("### Live Chart")
    # embed TradingView
    tv_html = tradingview_embed(tv_symbol, timeframe)
    components.html(tv_html, height=700)

    st.markdown("")  # spacer
    # Generate button under chart
    gen_clicked = st.button("📤 Generate Signal", key="generate_button")

with output_col:
    st.markdown("### Signal Summary")
    placeholder = st.empty()
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.write(f"Market: **{selected_market}**")
    st.write(f"Timeframe: **{timeframe}**")
    st.write(f"Account balance: **${account_balance:.2f}**")
    st.write(f"Risk / trade: **{risk_percent:.2f}%**")
    st.markdown("---")
    st.markdown("### Recent History")
    if os.path.exists("signal_history.csv"):
        try:
            hist = pd.read_csv("signal_history.csv")
            st.dataframe(hist.tail(show_history_rows).reset_index(drop=True))
        except Exception:
            st.info("Unable to read history file.")
    else:
        st.info("No saved signal history yet. Use Generate Signal and Save.")

# ---------------------------
# When Generate clicked: fetch data, compute signal, show
# ---------------------------
if gen_clicked:
    with st.spinner("Fetching data and running strategy..."):
        yf_ticker = MARKETS[selected_market][0]
        df = fetch_ohlc_yf(yf_ticker, timeframe)
        if df is None or df.empty:
            st.error("Unable to fetch live OHLC data for the selected market/timeframe. Try again or pick a different timeframe.")
        else:
            sig = compute_master_signal(df, selected_market, account_balance, risk_percent, enable_htf, timeframe)
            # Display signal in the right column placeholder
            with output_col:
                st.markdown("## Result")
                if sig["signal"] == "WAIT":
                    st.warning(f"Signal: WAIT  •  Confidence: {sig['confidence']}%")
                elif sig["signal"] == "BUY":
                    st.success(f"Signal: BUY  •  Confidence: {sig['confidence']}%")
                elif sig["signal"] == "SELL":
                    st.error(f"Signal: SELL  •  Confidence: {sig['confidence']}%")
                st.markdown("**Entry / SL / TP / Lot / R:R**")
                st.write(f"Entry: `{sig['entry']}`")
                st.write(f"SL: `{sig['sl']}`")
                st.write(f"TP: `{sig['tp']}`")
                st.write(f"Lot (approx): `{sig['lot']}`")
                st.write(f"Risk: `${(risk_percent/100.0)*account_balance:.2f}`  •  R:R: `{sig['rr']}`")
                st.markdown("**Indicators**")
                st.json(sig.get("indicators", {}))
                if sig.get("reasons"):
                    st.markdown("**Reasons / confirmations:**")
                    for r in sig["reasons"]:
                        st.write("- " + r)

                # Save option
                if st.button("💾 Save Signal to History"):
                    row = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Market": selected_market,
                        "Timeframe": timeframe,
                        "Signal": sig["signal"],
                        "Entry": sig["entry"],
                        "SL": sig["sl"],
                        "TP": sig["tp"],
                        "R:R": sig["rr"],
                        "Lot": sig["lot"],
                        "Confidence": sig["confidence"],
                        "Reasons": " | ".join(sig.get("reasons", []))
                    }
                    try:
                        save_signal_history(row)
                        st.success("Saved to signal_history.csv")
                    except Exception as e:
                        st.error(f"Failed to save history: {e}")

# ---------------------------
# Auto-refresh handling (simple)
# ---------------------------
if auto_refresh and auto_refresh > 0:
    st.experimental_rerun()
