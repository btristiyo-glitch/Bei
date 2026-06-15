import pandas as pd
import yfinance as yf
import requests
import os

from ta.momentum import RSIIndicator

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send(msg):
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={
            "chat_id": CHAT_ID,
            "text": msg
        },
        timeout=30
    )

with open("stocks.txt", "r") as f:
    STOCKS = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]

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

        close = df["Close"].squeeze()

        rsi = RSIIndicator(
            close,
            window=4
        ).rsi()

        rsi_now = float(rsi.iloc[-1])

        price = float(close.iloc[-1])

        results.append({
            "ticker": stock.replace(".JK", ""),
            "rsi": round(rsi_now, 2),
            "price": round(price, 2)
        })

    except:
        continue

results = sorted(
    results,
    key=lambda x: x["rsi"]
)

message = "🇮🇩 TOP 20 RSI(4) TERENDAH BEI\n\n"

for item in results[:20]:

    message += (
        f"{item['ticker']}\n"
        f"RSI(4): {item['rsi']}\n"
        f"Harga: Rp {item['price']:,.0f}\n\n"
    )

send(message)
