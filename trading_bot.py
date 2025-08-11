# trading_bot.py
# Pro Signal Bot (fixed): TradingView embed + Generate Signal + robust yfinance fetch
# Copy-paste this entire file and run with: streamlit run trading_bot.py
#
# Minimal dependencies:
#   streamlit, pandas, numpy, yfinance
#
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
import os
from datetime import datetime
import streamlit.components.v1 as components
import time

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Pro Signal Bot", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background-color: #0b0e13; color:#dbe6ff; }
      .sidebar .sidebar-content { background-color: #111217; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("📈 Pro Signal Bot — Master Strategy (Generate Signal)")

# ---------------------------
# Market map and timeframe mapping
# ---------------------------
MARKETS = {
    "EUR/USD": ("EURUSD=X", "OANDA:EURUSD"),
    "GBP/JPY": ("GBPJPY=X", "OANDA:GBPJPY"),
    "USD/JPY": ("JPY=X", "OANDA:USDJPY"),
    "AUD/USD": ("AUDUSD=X", "OANDA:AUDUSD"),
    "XAU/USD (Gold)": ("XAUUSD=X", "OANDA:XAUUSD"),
    "BTC/USD": ("BTC-USD", "BINANCE:BTCUSDT"),
    "ETH/USD": ("ETH-USD", "BINANCE:ETHUSDT"),
}

# intervals and default period suggestions for yfinance
TIMEFRAME_MAP = {
    "1m": ("1m", "7d"),
    "5m": ("5m", "30d"),
    "15m": ("15m", "60d"),
    "1h": ("60m", "120d"),
    "4h": ("60m", "180d"),  # yfinance has limited intervals; 60m often used
    "1d": ("1d", "2y"),
}

# ---------------------------
# Sidebar: user settings
# ---------------------------
with st.sidebar:
    st.header("Account & Settings")
    account_balance = st.number_input("Account Balance ($)", min_value=1.0, value=1000.0, step=10.0, format="%.2f")
    risk_percent = st.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.2f")
    timeframe = st.selectbox("Chart timeframe", options=list(TIMEFRAME_MAP.keys()), index=0)
    auto_refresh = st.number_input("Auto-refresh every N seconds (0 = off)", min_value=0, value=0, step=1)
    enable_htf = st.checkbox("Enable Higher-Timeframe Confirmation", value=True)
    st.markdown("---")
    st.info("Manual trading only. No auto-execution included.")

# Top controls
col_top_left, col_top_mid, col_top_right = st.columns([2, 2, 1])
with col_top_left:
    selected_market = st.selectbox("Select Market", list(MARKETS.keys()))
with col_top_mid:
    history_rows = st.number_input("History rows to show", min_value=5, max_value=200, value=20, step=5)
with col_top_right:
    if st.button("🔄 Refresh Chart / Data (manual)"):
        # button press will automatically trigger a rerun; no explicit experimental_rerun() call (some environments lack it)
        st.success("Refreshed (if data available).")

# ---------------------------
# Indicator helpers (safe)
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

# ---------------------------
# Robust OHLC fetch with retries/fallbacks (yfinance)
# ---------------------------
def fetch_ohlc_yf(yf_ticker: str, timeframe_key: str, max_attempts: int = 3):
    """
    Try multiple combos to fetch OHLC. Returns DataFrame or None.
    """
    attempts = []
    # prioritize requested mapping
    if timeframe_key in TIMEFRAME_MAP:
        attempts.append(TIMEFRAME_MAP[timeframe_key])
    # fallback intervals list (common)
    attempts.extend([("5m","30d"), ("15m","60d"), ("60m","120d"), ("1d","2y")])
    tried = []
    for interval, period in attempts:
        try:
            # yfinance returns empty sometimes - try a few times
            df = yf.download(tickers=yf_ticker, interval=interval, period=period, progress=False, threads=False)
            # if columns present and not empty:
            if df is not None and not df.empty:
                # normalize columns to Titlecase if necessary
                df = df.rename(columns=lambda s: s.capitalize())
                # ensure required columns exist
                if all(c in df.columns for c in ("Open","High","Low","Close")):
                    return df.dropna()
            tried.append((interval, period, False))
        except Exception as ex:
            tried.append((interval, period, str(ex)))
        # small backoff
        time.sleep(0.4)
    # nothing worked
    return None

# ---------------------------
# Strategy engine (ensemble)
# ---------------------------
def compute_master_signal(df, market_name, account_balance, risk_pct, enable_htf_confirm=True, timeframe_sel="1m"):
    result = {"signal":"WAIT","confidence":0,"reasons":[],"entry":None,"sl":None,"tp":None,"lot":0.0,"rr":None,"indicators":{}}
    if df is None or df.empty or len(df) < 12:
        return result
    close = df['Close']
    price = float(close.iloc[-1])
    result['entry'] = round(price, 6)

    # indicators
    ema20 = float(ema(close, 20).iloc[-1]) if len(close) >= 20 else float(close.iloc[-1])
    ema50 = float(ema(close, 50).iloc[-1]) if len(close) >= 50 else ema20
    rsi14 = float(rsi(close, 14).iloc[-1])
    macd_line, macd_sig, macd_hist = macd(close)
    macd_hist_val = float(macd_hist.iloc[-1])
    bb_u, bb_l, bb_w = bollinger(close, 20, 2.0)
    bb_width = float(bb_w.iloc[-1]) if not math.isnan(bb_w.iloc[-1]) else 0.0
    atrv = float(atr(df, 14).iloc[-1]) if 'High' in df.columns else 0.0

    # basic scoring
    score = 0
    reasons = []
    trend = "side"
    if ema20 > ema50:
        score += 25; reasons.append("EMA bullish")
        trend = "up"
    elif ema20 < ema50:
        score += 25; reasons.append("EMA bearish")
        trend = "down"

    # RSI
    if rsi14 < 30: score += 20; reasons.append("RSI oversold")
    elif rsi14 > 70: score += 20; reasons.append("RSI overbought")
    else: score += 5

    # MACD
    if macd_hist_val > 0: score += 15; reasons.append("MACD positive")
    else: score += 5; reasons.append("MACD negative")

    # Bollinger width
    if bb_width < 0.01: score += 5; reasons.append("Bollinger squeeze")
    else: score += 6

    # candles
    if is_bullish_engulfing(df): score += 12; reasons.append("Bullish engulfing")
    if is_bearish_engulfing(df): score += 12; reasons.append("Bearish engulfing")

    confidence = min(120, int(score))
    result['confidence'] = confidence
    result['reasons'] = reasons
    result['indicators'] = {"ema20": round(ema20,6),"ema50":round(ema50,6),"rsi14":round(rsi14,2),
                            "macd_hist":round(macd_hist_val,6),"atr":round(atrv,6),"bb_width":round(bb_width,6)}

    # conditions
    buy_ok = (ema20 > ema50) and (macd_hist_val > 0) and (rsi14 < 65)
    sell_ok = (ema20 < ema50) and (macd_hist_val < 0) and (rsi14 > 35)
    extra_buy = is_bullish_engulfing(df) or (macd_hist_val>0 and bb_width>0.005)
    extra_sell = is_bearish_engulfing(df) or (macd_hist_val<0 and bb_width>0.005)

    threshold = 60
    if buy_ok and confidence >= threshold and extra_buy:
        decision = "BUY"
    elif sell_ok and confidence >= threshold and extra_sell:
        decision = "SELL"
    else:
        decision = "WAIT"

    # ATR based SL/TP
    atr_factor = 1.5
    if atrv <= 0:
        sl_dist = max(price * 0.0005, 0.0001)
    else:
        sl_dist = atrv * atr_factor

    if decision == "BUY":
        sl = price - sl_dist
        tp = price + sl_dist * 3
    elif decision == "SELL":
        sl = price + sl_dist
        tp = price - sl_dist * 3
    else:
        sl = None; tp = None

    # approximate lot sizing (very rough)
    def pip_unit(label):
        if "JPY" in label: return 0.01
        if "XAU" in label: return 0.1
        if "BTC" in label or "ETH" in label: return 1.0
        return 0.0001
    pip = pip_unit(market_name)
    lot = 0.0
    if sl is not None and abs(price - sl) > 0 and pip > 0:
        stop_pips = abs(price - sl) / pip
        pip_value = 10.0  # rough USD per pip per standard lot
        risk_amount = (risk_pct / 100.0) * account_balance
        lot = (risk_amount / (stop_pips * pip_value)) if stop_pips>0 else 0.0
    rr = None
    if sl is not None and tp is not None:
        rr = round(abs(tp - price) / (abs(price - sl) if abs(price - sl)>0 else 1e-9), 2)

    result.update({"signal":decision,"sl": round(sl,6) if sl else None,"tp":round(tp,6) if tp else None,
                   "lot": round(lot,6),"rr": rr})
    return result

# ---------------------------
# Save history
# ---------------------------
def save_signal_history(row: dict):
    path = "signal_history.csv"
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)

# ---------------------------
# TradingView embed helper
# ---------------------------
def tradingview_embed(tv_symbol: str, interval: str):
    interval_map = {"1m":"1","5m":"5","15m":"15","1h":"60","4h":"240","1d":"D"}
    tv_interval = interval_map.get(interval, "1")
    html = f"""
    <div class="tradingview-widget-container" style="width:100%; height:680px;">
      <div id="tv_{int(time.time())}"></div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
      new TradingView.widget({{
        "width":"100%",
        "height":680,
        "symbol":"{tv_symbol}",
        "interval":"{tv_interval}",
        "timezone":"Etc/UTC",
        "theme":"dark",
        "style":"1",
        "locale":"en",
        "container_id":"tv_{int(time.time())}"
      }});
      </script>
    </div>
    """
    return html

# ---------------------------
# Layout: chart (big) + RHS info
# ---------------------------
tv_symbol = MARKETS[selected_market][1]
left_col, right_col = st.columns([3,1])

with left_col:
    st.subheader("Live Chart")
    tv_html = tradingview_embed(tv_symbol, timeframe)
    components.html(tv_html, height=700)
    gen_clicked = st.button("📤 Generate Signal", key="generate_button")

with right_col:
    st.markdown("### Signal Summary")
    st.write(f"Market: **{selected_market}**")
    st.write(f"Timeframe: **{timeframe}**")
    st.write(f"Account: **${account_balance:,.2f}**")
    st.write(f"Risk: **{risk_percent:.2f}%**")
    st.markdown("---")
    st.markdown("### Recent History")
    if os.path.exists("signal_history.csv"):
        try:
            hist = pd.read_csv("signal_history.csv")
            st.dataframe(hist.tail(history_rows).reset_index(drop=True))
        except Exception as e:
            st.write("Unable to show history:", e)
    else:
        st.info("No history file yet (save signals after generating).")

# ---------------------------
# On Generate: fetch, compute, show
# ---------------------------
if gen_clicked:
    st.info("Fetching data... please wait (may take a few seconds).")
    yf_ticker = MARKETS[selected_market][0]
    df = fetch_ohlc_yf(yf_ticker, timeframe)
    if df is None or df.empty:
        st.error("Unable to fetch live OHLC data for the selected market/timeframe. Try a different timeframe (5m,15m,1h) or run locally.")
    else:
        st.success("Data fetched. Running strategy...")
        sig = compute_master_signal(df, selected_market, account_balance, risk_percent, enable_htf, timeframe)
        st.markdown("## Result")
        if sig["signal"] == "WAIT":
            st.warning(f"Signal: WAIT  •  Confidence: {sig['confidence']}%")
        elif sig["signal"] == "BUY":
            st.success(f"Signal: BUY  •  Confidence: {sig['confidence']}%")
        else:
            st.error(f"Signal: SELL  •  Confidence: {sig['confidence']}%")

        st.markdown("**Entry / SL / TP / Lot / R:R**")
        st.write(f"Entry: `{sig['entry']}`")
        st.write(f"SL: `{sig['sl']}`")
        st.write(f"TP: `{sig['tp']}`")
        st.write(f"Lot (approx): `{sig['lot']}`")
        st.write(f"R:R: `{sig['rr']}`")
        st.markdown("**Indicators & reasons**")
        st.json(sig.get("indicators", {}))
        if sig.get("reasons"):
            for r in sig["reasons"]:
                st.write("- " + r)

        # save
        if st.button("💾 Save this signal"):
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market": selected_market,
                "timeframe": timeframe,
                "signal": sig["signal"],
                "entry": sig["entry"],
                "sl": sig["sl"],
                "tp": sig["tp"],
                "rr": sig["rr"],
                "lot": sig["lot"],
                "confidence": sig["confidence"],
                "reasons": " | ".join(sig.get("reasons", []))
            }
            try:
                save_signal_history(row)
                st.success("Saved to signal_history.csv")
            except Exception as e:
                st.error("Failed to save: " + str(e))

# ---------------------------
# Auto-refresh (optional simple)
# ---------------------------
if auto_refresh and auto_refresh > 0:
    # clicking the input changes the app and causes a rerun; to make it automatic you'd need `st_autorefresh`
    st.info(f"Auto-refresh requested every {auto_refresh}s. This environment may not support automatic rerun; use manual refresh if it does not auto-update.")

