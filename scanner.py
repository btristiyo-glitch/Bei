import os
import math
import csv
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TICKERS_FILE = "tickers.txt"
SETUPS_CSV = "setups.csv"
ACTIVE_SETUPS_CSV = "active_setups.csv"
POSITION_STATE_FILE = "position_state.csv"
ERROR_LOG_FILE = "error.log"

TIMEFRAME = "1d"
PERIOD = 120

# Longgarin filter
MIN_GAP_PCT = 1.5
MAX_GAP_PCT = 12.0
MIN_AVG_VOLUME = 500000
MIN_DOLLAR_VOLUME = 1000000
MIN_ATR_PCT = 2.0
MIN_RSI = 35
MAX_RSI = 78
MIN_SCORE = 35

# =========================
# TELEGRAM
# =========================
def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/chat_id belum diisi")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# =========================
# DATA HELPERS
# =========================
def load_tickers():
    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def fetch_ohlcv(ticker, period=PERIOD, timeframe=TIMEFRAME):
    import yfinance as yf
    df = yf.download(ticker, period=f"{period}d", interval=timeframe, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    df = df.dropna().copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df

def calculate_indicators(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["sma20"] = SMAIndicator(close, window=20).sma_indicator()
    df["ema20"] = EMAIndicator(close, window=20).ema_indicator()
    df["rsi14"] = RSIIndicator(close, window=14).rsi()
    atr = AverageTrueRange(high, low, close, window=14)
    df["atr14"] = atr.average_true_range()

    df["avg_volume20"] = volume.rolling(20).mean()
    df["dollar_volume"] = df["close"] * df["volume"]
    df["avg_dollar_volume20"] = df["dollar_volume"].rolling(20).mean()

    return df

def score_setup(row):
    score = 0

    gap_pct = row.get("gap_pct", 0)
    rsi = row.get("rsi14", 0)
    atr_pct = row.get("atr_pct", 0)
    rel_vol = row.get("rel_volume", 0)
    close = row.get("close", 0)
    ema20 = row.get("ema20", np.nan)
    sma20 = row.get("sma20", np.nan)

    if gap_pct >= 2:
        score += 10
    elif gap_pct >= 1.5:
        score += 6

    if rel_vol >= 1.5:
        score += 10
    elif rel_vol >= 1.2:
        score += 6

    if atr_pct >= 3:
        score += 8
    elif atr_pct >= 2:
        score += 5

    if 40 <= rsi <= 65:
        score += 10
    elif 35 <= rsi < 40 or 65 < rsi <= 75:
        score += 6

    if pd.notna(ema20) and close > ema20:
        score += 5
    if pd.notna(sma20) and close > sma20:
        score += 5

    return score

def create_signal_text(ticker, row, score):
    return (
        f"📈 <b>Setup Found</b>\n"
        f"Ticker: {ticker}\n"
        f"Close: {row['close']:.2f}\n"
        f"Gap: {row.get('gap_pct', 0):.2f}%\n"
        f"Rel Vol: {row.get('rel_volume', 0):.2f}x\n"
        f"ATR%: {row.get('atr_pct', 0):.2f}%\n"
        f"RSI: {row.get('rsi14', 0):.2f}\n"
        f"Score: {score}/100\n"
    )

# =========================
# CSV HELPERS
# =========================
def append_to_csv(path, rows, headers):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# =========================
# SCAN
# =========================
def scan():
    tickers = load_tickers()
    setups = []
    signals_sent = 0

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"Scanning {i}/{len(tickers)} - {ticker}")
            df = fetch_ohlcv(ticker)
            if df is None or len(df) < 30:
                continue

            df = calculate_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            gap_pct = ((latest["open"] - prev["close"]) / prev["close"]) * 100 if prev["close"] else 0
            atr_pct = (latest["atr14"] / latest["close"]) * 100 if latest["close"] else 0
            rel_volume = latest["volume"] / latest["avg_volume20"] if latest["avg_volume20"] and latest["avg_volume20"] > 0 else 0
            dollar_volume = latest["close"] * latest["volume"]
            avg_dollar_volume20 = latest["avg_dollar_volume20"] if pd.notna(latest["avg_dollar_volume20"]) else 0

            row = {
                "ticker": ticker,
                "timestamp": datetime.utcnow().isoformat(),
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
                "volume": float(latest["volume"]),
                "gap_pct": float(gap_pct),
                "atr_pct": float(atr_pct),
                "rel_volume": float(rel_volume),
                "dollar_volume": float(dollar_volume),
                "avg_dollar_volume20": float(avg_dollar_volume20),
                "rsi14": float(latest["rsi14"]),
                "ema20": float(latest["ema20"]) if pd.notna(latest["ema20"]) else "",
                "sma20": float(latest["sma20"]) if pd.notna(latest["sma20"]) else "",
            }

            if gap_pct < MIN_GAP_PCT or gap_pct > MAX_GAP_PCT:
                continue
            if dollar_volume < MIN_DOLLAR_VOLUME:
                continue
            if atr_pct < MIN_ATR_PCT:
                continue
            if latest["rsi14"] < MIN_RSI or latest["rsi14"] > MAX_RSI:
                continue

            score = score_setup(row)
            if score < MIN_SCORE:
                continue

            setups.append(row)
            msg = create_signal_text(ticker, row, score)
            if send_telegram_message(msg):
                signals_sent += 1

        except Exception as e:
            with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.utcnow().isoformat()}] {ticker} - {repr(e)}\n")
            continue

    headers = [
        "ticker", "timestamp", "open", "high", "low", "close", "volume",
        "gap_pct", "atr_pct", "rel_volume", "dollar_volume",
        "avg_dollar_volume20", "rsi14", "ema20", "sma20"
    ]
    pd.DataFrame(setups).to_csv(SETUPS_CSV, index=False)

    print(f"Done - setups: {len(setups)} | signals: {signals_sent}")

if __name__ == "__main__":
    scan()
