import pandas as pd
import yfinance as yf
import requests
import os
import csv
from datetime import datetime
from ta.momentum import RSIIndicator

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CSV_FILE = "signals.csv"

def send(msg):
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg, "disable_web_page_preview": False},
        timeout=30
    )

def trend_status(symbol):
    try:
        df = yf.download(symbol, period="6mo", auto_adjust=True, progress=False)
        close = df["Close"].squeeze()
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        price = close.iloc[-1]

        if price > ema20 > ema50:
            return "BULLISH"
        elif price < ema20 < ema50:
            return "BEARISH"
        return "NEUTRAL"
    except:
        return "UNKNOWN"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["date","ticker","score","rsi","price","rvol"])

def save_signal(ticker, score, rsi, price, rvol):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [datetime.now().strftime("%Y-%m-%d"), ticker, score, rsi, price, rvol]
        )

with open("stocks.txt","r") as f:
    STOCKS = [x.strip() for x in f if x.strip()]

sector_map = {}
if os.path.exists("sectors.csv"):
    with open("sectors.csv","r") as f:
        for row in csv.DictReader(f):
            sector_map[row["ticker"].upper()] = row["sector"].upper()

IHSG = trend_status("^JKSE")
USDIDR = trend_status("USDIDR=X")
GOLD = trend_status("GC=F")
OIL = trend_status("CL=F")
COAL = trend_status("KOL")

results = []
high_conviction = []
extreme = []

for stock in STOCKS:
    try:
        df = yf.download(stock, period="12mo", interval="1d",
                         auto_adjust=True, progress=False)

        if len(df) < 60:
            continue

        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        price = float(close.iloc[-1])
        rsi = RSIIndicator(close, window=4).rsi()
        rsi_now = float(rsi.iloc[-1])
        rsi_prev = float(rsi.iloc[-2])

        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

        avg_vol = float(volume.tail(20).mean())
        if avg_vol <= 0:
            continue

        rvol = float(volume.iloc[-1]) / avg_vol
        breakout = price >= float(close.tail(20).max())

        score = 0

        score += max(0, min(40, (50 - rsi_now) * 0.8))

        if rsi_now > rsi_prev:
            score += min(20, (rsi_now - rsi_prev) * 5)

        score += min(20, rvol * 5)

        if price > ema20:
            score += 10

        if ema20 > ema50:
            score += 10

        if breakout:
            score += 15

        ticker = stock.replace(".JK","")
        sector = sector_map.get(ticker, "OTHER")

        if IHSG == "BULLISH":
            score += 10
        elif IHSG == "BEARISH":
            score -= 10

        if USDIDR == "BULLISH" and sector in ["COAL","NICKEL","OIL","GAS"]:
            score += 5

        if GOLD == "BULLISH" and sector == "NICKEL":
            score += 10

        if OIL == "BULLISH" and sector in ["OIL","GAS"]:
            score += 10

        if COAL == "BULLISH" and sector == "COAL":
            score += 10

        item = {
            "ticker": ticker,
            "score": round(score,1),
            "rsi": round(rsi_now,2),
            "price": round(price,2),
            "rvol": round(rvol,2),
            "breakout": breakout
        }

        results.append(item)

        if score >= 80:
            high_conviction.append(item)

        if rsi_now < 20:
            extreme.append(item)

    except Exception:
        continue

results = sorted(results, key=lambda x: x["score"], reverse=True)

for item in results[:10]:
    save_signal(item["ticker"], item["score"], item["rsi"], item["price"], item["rvol"])

market_msg = (
    "🌏 MARKET REGIME\n\n"
    f"IHSG : {IHSG}\n"
    f"USDIDR : {USDIDR}\n"
    f"GOLD : {GOLD}\n"
    f"OIL : {OIL}\n"
    f"COAL : {COAL}\n"
)
send(market_msg)

msg1 = "🏆 TOP 10 SETUP BEI\n\n"
for item in results[:10]:
    tv = f"https://www.tradingview.com/chart/?symbol=IDX:{item['ticker']}"
    msg1 += (
        f"#{item['ticker']}\n"
        f"Score : {item['score']}\n"
        f"RSI(4): {item['rsi']}\n"
        f"RVOL : {item['rvol']}x\n"
        f"Harga : Rp {item['price']:,.0f}\n"
        f"{'🚀 Breakout20' if item['breakout'] else '-'}\n"
        f"{tv}\n\n"
    )
send(msg1)

if high_conviction:
    msg2 = "🔥 HIGH CONVICTION\n\n"
    for item in sorted(high_conviction, key=lambda x:x["score"], reverse=True)[:10]:
        msg2 += f"{item['ticker']}\nScore : {item['score']}\nRSI : {item['rsi']}\nRVOL : {item['rvol']}x\n\n"
    send(msg2)

if extreme:
    msg3 = "🚨 EXTREME OVERSOLD\n\n"
    for item in sorted(extreme, key=lambda x:x["rsi"])[:10]:
        msg3 += f"{item['ticker']}\nRSI(4): {item['rsi']}\nHarga : Rp {item['price']:,.0f}\n\n"
    send(msg3)

print(f"Scan selesai. Total saham: {len(results)}")
