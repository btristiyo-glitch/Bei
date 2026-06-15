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
            "text": msg,
            "disable_web_page_preview": True
        },
        timeout=30
    )

# Baca daftar saham dari stocks.txt
with open("stocks.txt", "r") as f:
    STOCKS = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]

all_rsi = []
rebound = []

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
        volume = df["Volume"].squeeze()

        rsi_series = RSIIndicator(
            close,
            window=4
        ).rsi()

        rsi_now = float(rsi_series.iloc[-1])
        rsi_prev = float(rsi_series.iloc[-2])

        price = float(close.iloc[-1])

        volume_now = float(volume.iloc[-1])
        avg_volume = float(volume.tail(20).mean())

        ticker = stock.replace(".JK", "")

        # Simpan untuk ranking RSI
        all_rsi.append({
            "ticker": ticker,
            "rsi": round(rsi_now, 2),
            "price": round(price, 2)
        })

        # Scanner rebound
        if (
            rsi_now < 30 and
            rsi_now > rsi_prev and
            volume_now > avg_volume
        ):

            score = (
                (30 - rsi_now) * 2
                + (volume_now / avg_volume) * 10
            )

            rebound.append({
                "ticker": ticker,
                "rsi": round(rsi_now, 2),
                "price": round(price, 2),
                "vol_ratio": round(volume_now / avg_volume, 2),
                "score": round(score, 2)
            })

    except Exception:
        continue

# ==========================
# TOP RSI TERENDAH
# ==========================

all_rsi = sorted(
    all_rsi,
    key=lambda x: x["rsi"]
)

msg1 = "🇮🇩 TOP 20 RSI(4) TERENDAH BEI\n\n"

for item in all_rsi[:20]:

    msg1 += (
        f"📉 {item['ticker']}\n"
        f"RSI(4): {item['rsi']}\n"
        f"Harga: Rp {item['price']:,.0f}\n\n"
    )

send(msg1)

# ==========================
# REBOUND SCANNER
# ==========================

rebound = sorted(
    rebound,
    key=lambda x: x["score"],
    reverse=True
)

if rebound:

    msg2 = "🚀 REBOUND SCANNER BEI\n\n"

    for item in rebound[:10]:

        tv_link = (
            f"https://www.tradingview.com/chart/?symbol=IDX:"
            f"{item['ticker']}"
        )

        msg2 += (
            f"⭐ {item['ticker']}\n"
            f"Score: {item['score']}\n"
            f"RSI(4): {item['rsi']}\n"
            f"Vol: {item['vol_ratio']}x Avg20\n"
            f"Harga: Rp {item['price']:,.0f}\n"
            f"{tv_link}\n\n"
        )

    send(msg2)

else:

    send(
        "🚀 REBOUND SCANNER BEI\n\n"
        "Belum ada saham yang memenuhi kriteria hari ini."
    )
