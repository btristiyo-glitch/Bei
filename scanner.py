
import pandas as pd
import yfinance as yf
import requests
import os

from ta.momentum import RSIIndicator

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

with open("stocks.txt", "r") as f:
    STOCKS = [x.strip() for x in f if x.strip()]

def send(msg):
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={
            "chat_id": CHAT_ID,
            "text": msg
        },
        timeout=30
    )

results = []

for stock in STOCKS:

    try:

        df = yf.download(
            stock,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if len(df) < 30:
            continue

        close = df["Close"]

        rsi = RSIIndicator(
            close,
            window=4
        ).rsi()

        rsi_now = float(rsi.iloc[-1])

        if rsi_now >= 25:
            continue

        price = float(close.iloc[-1])

        if price < 50:
            continue

        volume_now = float(df["Volume"].iloc[-1])
        avg_volume = float(df["Volume"].tail(20).mean())

        
        results.append({
            "ticker": stock.replace(".JK", ""),
            "rsi": round(rsi_now, 2),
            "price": round(price, 2),
            "volume": int(volume_now)
        })

    except Exception:
        continue

results = sorted(
    results,
    key=lambda x: x["rsi"]
)

if results:

    message = "🇮🇩 BEI RSI(4) OVERSOLD\n\n"

    for item in results[:10]:

        tv_link = (
            f"https://www.tradingview.com/chart/?symbol=IDX:"
            f"{item['ticker']}"
        )

        message += (
            f"📉 {item['ticker']}\n"
            f"RSI(4): {item['rsi']}\n"
            f"Harga: Rp {item['price']:,.0f}\n"
            f"Volume: {item['volume']:,}\n"
            f"{tv_link}\n\n"
        )

    send(message)
