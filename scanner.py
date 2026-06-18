import pandas as pd
import yfinance as yf
import requests
import os
import csv
import time
from datetime import datetime
from ta.momentum import RSIIndicator

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CSV_FILE = "signals.csv"
BATCH_SIZE = 3
REQUEST_DELAY = 2

def send(msg, parse_mode="Markdown"):
    max_len = 4000
    for i in range(0, len(msg), max_len):
        chunk = msg[i:i+max_len]
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                json={
                    "chat_id": CHAT_ID,
                    "text": chunk,
                    "disable_web_page_preview": False,
                    "parse_mode": parse_mode
                },
                timeout=30
            )
        except Exception as e:
            print(f"Telegram send error: {e}")

def fetch_data(symbol, period="6mo"):
    try:
        df = yf.download(symbol, period=period, interval="1d",
                         auto_adjust=True, progress=False, timeout=15)
        if df.empty or len(df) < 50:
            return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df
    except Exception:
        return None

def trend_regime(symbol, label):
    df = fetch_data(symbol, period="6mo")
    if df is None:
        return f"{label}: DATA ERR"
    close = df["Close"].squeeze()
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]
    if price > ema20 > ema50:
        return f"{label}: 🟢 BULLISH"
    elif price < ema20 < ema50:
        return f"{label}: 🔴 BEARISH"
    return f"{label}: 🟡 NEUTRAL"

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker + ".JK")
        info = stock.info or {}
        pbv = info.get("priceToBook", None)
        per = info.get("trailingPE", None)
        mcap = info.get("marketCap", None)
        return pbv, per, mcap
    except Exception:
        return None, None, None

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["date", "time", "ticker", "score",
                                "rsi", "price", "rvol", "breakout"])

def save_signal(ticker, score, rsi, price, rvol, breakout):
    now = datetime.now()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [now.strftime("%Y-%m-%d"), now.strftime("%H:%M"),
             ticker, score, rsi, price, rvol, breakout]
        )

with open("stocks.txt", "r") as f:
    STOCKS = [x.strip() for x in f if x.strip()]
    STOCKS = [s if s.endswith(".JK") else s + ".JK" for s in STOCKS]

sector_map = {}
if os.path.exists("sectors.csv"):
    with open("sectors.csv", "r") as f:
        for row in csv.DictReader(f):
            sector_map[row["ticker"].upper().replace(".JK","")] = row["sector"].upper()

# ── AMBIH REGIME ──
regime_lines = []
for ticker, label in [("^JKSE", "IHSG"), ("USDIDR=X", "USDIDR"),
                       ("GC=F", "GOLD"), ("CL=F", "OIL")]:
    result = trend_regime(ticker, label)
    print(f"  {result}")
    regime_lines.append(result)
    time.sleep(REQUEST_DELAY)

IHSG_line = regime_lines[0]
USDIDR_bull = "🟢 BULLISH" in regime_lines[1]
GOLD_bull = "🟢 BULLISH" in regime_lines[2]
OIL_bull = "🟢 BULLISH" in regime_lines[3]

results = []
high_conviction = []
extreme_oversold = []
failed_tickers = []

for i, stock in enumerate(STOCKS):
    print(f"Scanning {i+1}/{len(STOCKS)}: {stock}")

    df = fetch_data(stock, period="12mo")
    if df is None:
        failed_tickers.append(stock)
        continue

    try:
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        price = float(close.iloc[-1])
        rsi = RSIIndicator(close, window=4).rsi()
        rsi_now = float(rsi.iloc[-1])
        rsi_prev = float(rsi.iloc[-2])

        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

        avg_vol = float(volume.tail(20).mean())
        if avg_vol <= 0 or price <= 0:
            continue

        rvol = float(volume.iloc[-1]) / avg_vol

        prev_20_high = float(close.shift(1).tail(20).max())
        prev_20_low = float(close.shift(1).tail(20).min())
        breakout_high = price > prev_20_high

        # ── FUNDAMENTAL (optional filter, tidak mematikan sinyal) ──
        pbv, per, mcap = get_fundamentals(stock)
        fundament_flag = ""
        if pbv and per and mcap:
            if pbv < 2 and per < 25 and 500_000_000_000 < mcap < 50_000_000_000_000:
                fundament_flag = "FUNDAMENTAL OK"

        # ── SCORING ──
        # Kategori oversold reversal
        if rsi_now < 30:
            score = 0
            score += max(0, min(30, (30 - rsi_now) * 2.5))
        else:
            score = 0
            # RSI > 50 sebagai bonus momentum
            score += max(0, min(20, rsi_now - 50) * 1.2)

        # Kenaikan RSI (bullish divergence / impulse)
        score += min(15, max(0, (rsi_now - rsi_prev) * 4))

        # Volume eksplosif
        score += min(20, rvol * 4)

        # Trend alignment
        if price > ema20:
            score += 8
        if ema20 > ema50:
            score += 8

        # Breakout valid
        if breakout_high:
            score += 18

        # Regime booster
        if "🟢 BULLISH" in IHSG_line:
            score += 8

        # Potential upside harian (5-20% daily mover)
        range_pct = ((prev_20_high - prev_20_low) / prev_20_low) * 100
        if 3 < range_pct < 25:
            score += 12

        # Fundamental bonus
        if fundament_flag:
            score += 5

        ticker_clean = stock.replace(".JK", "")
        sector = sector_map.get(ticker_clean, "OTHER")

        item = {
            "ticker": ticker_clean,
            "score": round(score, 1),
            "rsi": round(rsi_now, 2),
            "price": round(price, 0),
            "rvol": round(rvol, 2),
            "breakout": breakout_high,
            "range_pct": round(range_pct, 1),
            "fundamental": fundament_flag,
            "sector": sector
        }

        results.append(item)

        if score >= 80:
            high_conviction.append(item)
        if rsi_now < 22:
            extreme_oversold.append(item)

    except Exception as e:
        failed_tickers.append(f"{stock}: {e}")
        continue

    if (i + 1) % BATCH_SIZE == 0:
        time.sleep(REQUEST_DELAY)

results = sorted(results, key=lambda x: x["score"], reverse=True)

for item in results[:15]:
    save_signal(item["ticker"], item["score"], item["rsi"],
                item["price"], item["rvol"], item["breakout"])

# ── KIRIM TELEGRAM ──
regime_header = "🌏 **MARKET REGIME**\n"
for line in regime_lines:
    regime_header += f"  {line}\n"
regime_header += f"\n  Total saham discan: {len(results)}\n"
regime_header += f"  Gagal di-load: {len(failed_tickers)}\n"
send(regime_header)

# TOP 15
msg1 = "🏆 **TOP 15 SETUP HARIAN – BEI**\n_5-20% daily mover target_\n\n"
for item in results[:15]:
    tv = f"https://www.tradingview.com/chart/?symbol=IDX:{item['ticker']}"
    bflag = " 🚀BREAKOUT" if item["breakout"] else ""
    fflag = f" ({item['fundamental']})" if item["fundamental"] else ""
    emoji = "🔥" if item["score"] >= 80 else "⭐"
    msg1 += (
        f"{emoji} #{item['ticker']}{fflag}{bflag}\n"
        f"Score: {item['score']} | RSI(4): {item['rsi']}\n"
        f"RVOL: {item['rvol']}x | Range: {item['range_pct']}%\n"
        f"Harga: Rp {item['price']:,.0f}\n"
        f"[TradingView]({tv})\n\n"
    )
    if len(msg1) > 3800:
        break
send(msg1, parse_mode="Markdown")

# HIGH CONVICTION
if high_conviction:
    msg2 = "🔥 **HIGH CONVICTION (Score >= 80)**\n\n"
    for item in sorted(high_conviction, key=lambda x: x["score"], reverse=True)[:10]:
        msg2 += (
            f"#{item['ticker']} | Score: {item['score']}\n"
            f"RSI: {item['rsi']} | RVOL: {item['rvol']}x\n"
            f"Harga: Rp {item['price']:,.0f}\n\n"
        )
    send(msg2)

# EXTREME OVERSOLD
if extreme_oversold:
    msg3 = "🚨 **EXTREME OVERSOLD (RSI < 22)**\n_Potensi reversal jangka pendek_\n\n"
    for item in sorted(extreme_oversold, key=lambda x: x["rsi"])[:10]:
        msg3 += (
            f"#{item['ticker']} | RSI: {item['rsi']}\n"
            f"Harga: Rp {item['price']:,.0f} | RVOL: {item['rvol']}x\n\n"
        )
    send(msg3)

# FAILED LOG
if failed_tickers:
    fail_msg = "⚠️ **Gagal di-load**\n\n"
    for ft in failed_tickers[:10]:
        fail_msg += f"{ft}\n"
    if len(failed_tickers) > 10:
        fail_msg += f"...dan {len(failed_tickers)-10} lainnya"
    send(fail_msg)

print(f"\n✅ Scan selesai. Total saham berhasil: {len(results)}")
print(f"⚠️  Gagal: {len(failed_tickers)}")
print(f"🔥 High conviction: {len(high_conviction)}")
print(f"🚨 Extreme oversold: {len(extreme_oversold)}")
          
