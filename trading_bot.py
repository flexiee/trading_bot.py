# trading_bot.py
import os
import io
import time
from datetime import datetime, timedelta
import base64

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import streamlit as st

# Use TA indicators helper (if installed). If not, simple implementations are provided below.
try:
    import ta
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False

# ---------------------------
# Configuration / Symbols
# ---------------------------

# Map friendly name -> (tradingview_symbol, yfinance_ticker, market_type)
# market_type: "forex", "crypto", "stock", "index", "commodity"
MARKETS = {
    "EUR/USD": ("OANDA:EURUSD", "EURUSD=X", "forex"),
    "GBP/USD": ("OANDA:GBPUSD", "GBPUSD=X", "forex"),
    "USD/JPY": ("OANDA:USDJPY", "JPY=X", "forex"),  # note: yfinance uses JPY=X (USDJPY reversed sometimes)
    "AUD/USD": ("OANDA:AUDUSD", "AUDUSD=X", "forex"),
    "USD/CAD": ("OANDA:USDCAD", "CAD=X", "forex"),
    "BTC/USD": ("BINANCE:BTCUSDT", "BTC-USD", "crypto"),
    "ETH/USD": ("BINANCE:ETHUSDT", "ETH-USD", "crypto"),
    "XAU/USD": ("OANDA:XAUUSD", "XAUUSD=X", "commodity"),
    "NIFTY 50": ("NSE:NIFTY", "^NSEI", "index"),
    "S&P 500": ("INDEX:SPX", "^GSPC", "index"),
}

# Timeframe mapping: user selects string -> yfinance interval, and lookback n bars
TF_MAP = {
    "1m": ("1m", 120),
    "5m": ("5m", 240),
    "15m": ("15m", 240),
    "1h": ("60m", 240),
    "1d": ("1d", 365),
}

# File for history
HISTORY_FILE = "signal_history.csv"

# ---------------------------
# Utility functions
# ---------------------------

def ensure_history_file():
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=["timestamp","market","timeframe","signal","entry","sl","tp","rr","confidence","lot_size","notes"])
        df.to_csv(HISTORY_FILE, index=False)

def save_signal_to_history(row: dict):
    ensure_history_file()
    df = pd.read_csv(HISTORY_FILE)
    df = df.append(row, ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

@st.cache_data(ttl=30)
def fetch_ohlc(ticker: str, interval: str, n_bars: int):
    """
    Fetch OHLC with yfinance. Returns DataFrame with columns: Open, High, Low, Close, Volume and index = Datetime
    interval: yfinance interval string ("1m","5m","15m","60m","1d")
    """
    # yfinance requires period for minute intervals (e.g., '2d' for 1m)
    # Calculate period from n_bars * interval approximate
    if interval.endswith("m"):
        # minutes
        minutes = int(interval[:-1])
        total_minutes = minutes * n_bars
        if total_minutes <= 60:
            period = "1d"
        elif total_minutes <= 60*24:
            period = "7d"
        else:
            period = "30d"
    elif interval.endswith("h") or interval.endswith("60m"):
        period = "60d"
    else:
        period = "1y"

    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        # keep last n_bars
        if len(df) > n_bars:
            df = df.tail(n_bars)
        return df
    except Exception as e:
        st.debug(f"yfinance error: {e}")
        return None

# ---------------------------
# Indicator helpers
# ---------------------------

def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, signal)
    return macd_line, macd_signal

def bollinger_bands(series: pd.Series, period=20, std_mult=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return upper, lower

# Candlestick pattern: engulfing (simple)
def is_bullish_engulfing(df):
    # needs at least 2 candles
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    cur = df.iloc[-1]
    # prev red, cur green and cur body engulfs prev body
    prev_body = abs(prev['Close'] - prev['Open'])
    cur_body = abs(cur['Close'] - cur['Open'])
    cond = (prev['Close'] < prev['Open']) and (cur['Close'] > cur['Open']) and (cur['Close'] - cur['Open'] > prev_body)
    return bool(cond)

def is_bearish_engulfing(df):
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    cur = df.iloc[-1]
    prev_body = abs(prev['Close'] - prev['Open'])
    cur_body = abs(cur['Close'] - cur['Open'])
    cond = (prev['Close'] > prev['Open']) and (cur['Close'] < cur['Open']) and (cur['Open'] - cur['Close'] > prev_body)
    return bool(cond)

# ---------------------------
# Core strategy / signal
# ---------------------------

def compute_indicators(df: pd.DataFrame):
    close = df['Close']
    df = df.copy()
    # EMAs
    df['EMA8'] = ema(close, 8)
    df['EMA20'] = ema(close, 20)
    df['EMA50'] = ema(close, 50)
    # RSI
    df['RSI14'] = rsi(close, 14)
    # MACD
    macd_line, macd_signal = macd(close)
    df['MACD'] = macd_line
    df['MACD_SIGNAL'] = macd_signal
    # Bollinger
    upper, lower = bollinger_bands(close, 20, 2)
    df['BB_UP'] = upper
    df['BB_LOW'] = lower
    return df

def compute_master_signal(df: pd.DataFrame, df_htf: pd.DataFrame = None, market_type="forex"):
    """
    Returns dict with signal: BUY/SELL/WAIT, entry, sl, tp, confidence (0-100), reasons list
    df: timeframe df (most recent)
    df_htf: higher timeframe df for confirmation (optional)
    """
    reasons = []
    sig_score = 0
    last = df.iloc[-1]
    close = last['Close']

    # trend via EMA50 and EMA20
    trend = "up" if last['EMA20'] > last['EMA50'] else "down"

    # momentum via EMA8 vs EMA20
    momentum = "strong" if last['EMA8'] > last['EMA20'] and trend == "up" else ("strong" if last['EMA8'] < last['EMA20'] and trend == "down" else "weak")

    # RSI filter
    rsi_val = last['RSI14']
    if rsi_val < 30:
        reasons.append("RSI oversold")
        sig_score += 5
    elif rsi_val > 70:
        reasons.append("RSI overbought")
        sig_score += 5

    # MACD
    if last['MACD'] > last['MACD_SIGNAL']:
        sig_score += 8
        reasons.append("MACD bullish")
    else:
        sig_score += 0

    # Candlestick
    if is_bullish_engulfing(df):
        sig_score += 12
        reasons.append("Bullish engulfing")
    if is_bearish_engulfing(df):
        sig_score += 12
        reasons.append("Bearish engulfing")

    # Bollinger breakout
    if close > last['BB_UP']:
        sig_score += 7
        reasons.append("Price above BB upper")
    if close < last['BB_LOW']:
        sig_score += 7
        reasons.append("Price below BB lower")

    # HTF confirmation
    htf_confirm = False
    if df_htf is not None:
        df_htf = compute_indicators(df_htf)
        last_htf = df_htf.iloc[-1]
        if (trend == "up" and last_htf['EMA20'] > last_htf['EMA50']) or (trend == "down" and last_htf['EMA20'] < last_htf['EMA50']):
            htf_confirm = True
            sig_score += 10
            reasons.append("Higher TF trend confirmed")

    # final signal decision
    # buy conditions
    buy = False
    sell = False
    if trend == "up" and momentum == "strong" and (is_bullish_engulfing(df) or last['Close'] > last['EMA20']):
        buy = True
    if trend == "down" and momentum == "strong" and (is_bearish_engulfing(df) or last['Close'] < last['EMA20']):
        sell = True

    # require some minimal sig_score (tuneable)
    confidence = min(100, max(10, int(sig_score * 4)))  # scale to 0-100

    # If HTF enabled and not confirmed, lower confidence
    if df_htf is not None and not htf_confirm:
        confidence = int(confidence * 0.7)
        reasons.append("Higher timeframe NOT confirmed")

    # Compose signal and SL/TP (basic methodology)
    signal = "WAIT"
    entry = round(close, 5)
    sl = None
    tp = None

    # Define pip/point logic
    if market_type == "forex":
        pip_unit = 0.0001 if not entry < 0.01 else 0.000001
    elif market_type == "crypto":
        pip_unit = 0.01 if entry > 1 else 0.0001
    else:
        pip_unit = 0.01

    # Default risk distance: use recent ATR-like (stddev)
    recent_std = df['Close'].diff().abs().rolling(10).mean().iloc[-5:]
    # handle empty or NaN
    if recent_std is None or recent_std.empty or recent_std.isna().all():
        sl_distance = pip_unit * 30
    else:
        recent_std_val = recent_std.dropna().mean() if not recent_std.dropna().empty else pip_unit * 10
        sl_distance = max(abs(recent_std_val) * 1.5, pip_unit * 10)

    # set SL/TP based on direction
    if buy:
        signal = "BUY"
        sl = round(entry - sl_distance, 5)
        tp = round(entry + sl_distance * 3, 5)
    elif sell:
        signal = "SELL"
        sl = round(entry + sl_distance, 5)
        tp = round(entry - sl_distance * 3, 5)
    else:
        signal = "WAIT"

    # final package
    return {
        "signal": signal,
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "confidence": confidence,
        "reasons": reasons,
    }

# ---------------------------
# Risk sizing
# ---------------------------

def calculate_lot_size(account_balance, risk_percent, entry, stop_loss, market_type="forex"):
    """
    For forex: 1 standard lot = 100,000 units base currency.
    pip_value depends on pair; we approximate.
    For crypto: return quantity units to buy
    """
    if stop_loss is None:
        return 0
    risk_amount = account_balance * (risk_percent / 100.0)
    # price distance
    dist = abs(entry - stop_loss)
    if dist == 0:
        return 0
    # For forex approximate pip value: $10 per pip per lot for most major pairs (approx)
    if market_type == "forex":
        # lot_size = (risk_amount) / (pip_value * pip_distance)
        # assume pip_value_per_lot = 10 USD per pip (for 1 lot)
        pip_value_per_lot = 10.0
        pip_distance = dist / 0.0001 if entry > 0.01 else dist / 0.000001
        lot = (risk_amount) / (pip_value_per_lot * pip_distance + 1e-9)
        lot = max(0, round(lot, 2))
        return lot
    else:
        # For crypto/others, return asset quantity: qty = risk_amount / dist
        qty = risk_amount / (dist + 1e-9)
        qty = round(max(0, qty), 6)
        return qty

# ---------------------------
# Chart drawing: candlestick + RR box
# ---------------------------

def plot_candles_with_rr(df: pd.DataFrame, entry=None, sl=None, tp=None):
    """
    Returns PNG image bytes of Matplotlib chart of last portion of df with rectangle for SL/TP
    """
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    df_plot = df.copy().tail(60)  # last 60 bars
    fig, ax = plt.subplots(figsize=(8, 4))
    # plot candles (simple)
    o = df_plot['Open'].values
    h = df_plot['High'].values
    l = df_plot['Low'].values
    c = df_plot['Close'].values
    dates = mdates.date2num(df_plot.index.to_pydatetime())

    width = 0.0008 * (dates[-1] - dates[0]) * 100  # adaptive
    for i in range(len(dates)):
        color = 'green' if c[i] >= o[i] else 'red'
        ax.plot([dates[i], dates[i]], [l[i], h[i]], color='black', linewidth=0.6)
        ax.add_patch(Rectangle((dates[i]-width/2, min(o[i], c[i])),
                               width, abs(c[i]-o[i]), facecolor=color, edgecolor=color, alpha=0.8))

    ax.xaxis_date()
    ax.set_title("Price + R/R Box")
    ax.tick_params(axis='x', rotation=20)
    ymin = df_plot['Low'].min()
    ymax = df_plot['High'].max()
    rng = ymax - ymin
    ax.set_ylim(ymin - rng*0.1, ymax + rng*0.1)

    # draw RR box if provided
    if entry is not None and sl is not None and tp is not None:
        # rectangle spanning recent timeframe (width)
        x_left = dates[int(len(dates) * 0.2)]
        x_width = dates[-1] - x_left
        # top and bottom depending on direction
        top = max(entry, sl, tp)
        bottom = min(entry, sl, tp)
        ax.add_patch(Rectangle((x_left, bottom), x_width, top-bottom, facecolor='blue', alpha=0.08, edgecolor='blue'))
        # plot lines
        ax.axhline(entry, color='blue', linestyle='--', linewidth=1, label='Entry')
        ax.axhline(sl, color='red', linestyle='--', linewidth=1, label='SL')
        ax.axhline(tp, color='green', linestyle='--', linewidth=1, label='TP')
        ax.legend(loc='upper left')

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=90)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(layout="wide", page_title="Pro Signal Bot")

st.title("📊 Pro Signal Bot — Basic Live Signal Helper")

# Sidebar: account, settings
with st.sidebar:
    st.header("Account & Settings")
    account_balance = st.number_input("Account Balance ($)", value=1000.0, min_value=1.0, step=10.0)
    risk_percent = st.slider("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    timeframe = st.selectbox("Chart timeframe", list(TF_MAP.keys()), index=0)
    auto_refresh = st.number_input("Auto-refresh every N seconds (0=off)", value=0, min_value=0, step=1)
    enable_htf = st.checkbox("Enable higher-timeframe confirmation (TF above selected)")
    st.markdown("---")
    st.markdown("Made for manual execution. No auto-trading included.")

# main layout: left column for chart + controls, right column for signal info + history
col1, col2 = st.columns([1.4, 1.0])

with col1:
    st.subheader("Live TradingView Chart")
    # Market selection
    market_choice = st.selectbox("Select Market", list(MARKETS.keys()), index=0)
    tv_symbol = MARKETS[market_choice][0]
    yf_ticker = MARKETS[market_choice][1]
    market_type = MARKETS[market_choice][2]

    # TradingView iframe
    iframe_url = f"https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{yf_ticker}&symbol={tv_symbol}&interval={timeframe}&hidesidetoolbar=1&theme=dark"
    st.components.v1.iframe(iframe_url, height=520)

    # Buttons
    cols = st.columns([1,1,1])
    if cols[0].button("🔄 Refresh Chart / Data (manual)"):
        # just a trigger; we'll fetch below
        st.experimental_rerun()
    generate_pressed = cols[1].button("🔮 Generate Signal")
    if cols[2].button("⭐ Add/Remove Favorite"):
        # manage favorites in session_state
        if "favorites" not in st.session_state:
            st.session_state.favorites = []
        if market_choice in st.session_state.favorites:
            st.session_state.favorites.remove(market_choice)
            st.success(f"Removed {market_choice} from favorites")
        else:
            st.session_state.favorites.append(market_choice)
            st.success(f"Added {market_choice} to favorites")

    st.markdown("---")
    st.write("Favorites:")
    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if st.session_state.favorites:
        for f in st.session_state.favorites:
            st.write(f"- {f}")
    else:
        st.write("No favorites yet")

with col2:
    st.subheader("Signal Summary")
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    history_placeholder = st.empty()

# Fetch OHLC
interval, default_bars = TF_MAP[timeframe]
n_bars = default_bars

status_placeholder.info("Fetching OHLC data...")
df = fetch_ohlc(yf_ticker, interval, n_bars)
if df is None or df.empty:
    status_placeholder.error("Unable to fetch live OHLC data for the selected market/timeframe. Try a different timeframe or market.")
    st.stop()

# Compute indicators for selected TF
df_ind = compute_indicators(df)

# Get HTF if enabled
df_htf = None
if enable_htf:
    # pick higher timeframe: if 1m -> 5m, 5m->15m, 15m->1h, 1h->4h (approx) else skip
    htf_map = {
        "1m": "5m",
        "5m": "15m",
        "15m": "1h",
        "1h": "4h",
        "1d": "5d"
    }
    if timeframe in htf_map:
        htf_tf = htf_map[timeframe]
        if htf_tf in TF_MAP:
            interval_htf, bars_htf = TF_MAP.get(htf_tf, ("15m", 240))
        else:
            # fallback
            interval_htf = htf_tf
            bars_htf = 240
        df_htf = fetch_ohlc(yf_ticker, interval_htf, bars_htf)
        if df_htf is not None and not df_htf.empty:
            df_htf = compute_indicators(df_htf)
    else:
        df_htf = None

status_placeholder.success("Data fetched. Running analysis...")

# If generate_pressed or auto-refresh triggered, compute signal
if generate_pressed or (auto_refresh > 0 and st.experimental_get_query_params().get("last_refresh")):
    pass  # we'll compute below

# Compute master signal (every render)
sig = compute_master_signal(df_ind, df_htf, market_type=market_type)
# calculate lot size
lot_size = calculate_lot_size(account_balance, risk_percent, sig['entry'], sig['stop_loss'], market_type=market_type)
sig['lot_size'] = lot_size

# Show results
with result_placeholder.container():
    st.markdown(f"**Market:** {market_choice}  |  **TF:** {timeframe}")
    st.markdown(f"**Signal:**   :blue[{sig['signal']}]   |   **Confidence:** {sig['confidence']}%")
    st.progress(min(100, max(0, sig['confidence'])))
    st.write(f"Entry: {sig['entry']}   |   SL: {sig['stop_loss']}   |   TP: {sig['take_profit']}")
    rr = None
    if sig['stop_loss'] and sig['take_profit']:
        risk = abs(sig['entry'] - sig['stop_loss'])
        reward = abs(sig['take_profit'] - sig['entry'])
        rr = round((reward / risk) if risk != 0 else 0, 2)
        st.write(f"Risk: {round(risk,6)}  |  Reward: {round(reward,6)}  |  R:R = {rr}:1")
    st.write(f"Recommended lot / qty: {sig['lot_size']}")
    if sig['reasons']:
        st.markdown("**Reasons:**")
        for r in sig['reasons']:
            st.write(f"- {r}")

    # draw chart with rr box
    try:
        img_bytes = plot_candles_with_rr(df, entry=sig['entry'], sl=sig['stop_loss'], tp=sig['take_profit'])
        st.image(img_bytes, use_column_width=True)
    except Exception as e:
        st.warning("Failed to render R/R chart image.")

    # Save to history button
    if st.button("💾 Save Signal to History"):
        row = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "market": market_choice,
            "timeframe": timeframe,
            "signal": sig['signal'],
            "entry": sig['entry'],
            "sl": sig['stop_loss'],
            "tp": sig['take_profit'],
            "rr": rr,
            "confidence": sig['confidence'],
            "lot_size": sig['lot_size'],
            "notes": ";".join(sig['reasons'])
        }
        save_signal_to_history(row)
        st.success("Signal saved to history.")

# Show signal history live
ensure_history_file()
hist_df = pd.read_csv(HISTORY_FILE)
with history_placeholder.container():
    st.subheader("Recent History")
    if hist_df is None or hist_df.empty:
        st.info("No saved signal history yet.")
    else:
        st.dataframe(hist_df.tail(20).sort_values("timestamp", ascending=False))

# Auto refresh logic (optional)
if auto_refresh and auto_refresh > 0:
    st.experimental_set_query_params(last_refresh=int(time.time()))
    time.sleep(auto_refresh)
    st.experimental_rerun()

st.caption("Note: This bot is a signal helper. Always validate signals on your own charts and risk-manage trades.")
