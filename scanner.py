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
            "disable_web_page_preview": False
        },
        timeout=30
    )

# ==========================
# LOAD TICKERS
# ==========================

with open("stocks.txt", "r") as f:
    STOCKS = [
        x.strip()
        for x in f.readlines()
        if x.strip()
    ]

results = []
high_conviction = []
extreme = []

for stock in STOCKS:

    try:

        df = yf.download(
            stock,
            period="12mo",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if len(df) < 60:
            continue

        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        price = float(close.iloc[-1])

        # RSI(4)
        rsi_series = RSIIndicator(
            close,
            window=4
        ).rsi()

        rsi_now = float(rsi_series.iloc[-1])
        rsi_prev = float(rsi_series.iloc[-2])

        # EMA
        ema20 = float(
            close.ewm(
                span=20,
                adjust=False
            ).mean().iloc[-1]
        )

        ema50 = float(
            close.ewm(
                span=50,
                adjust=False
            ).mean().iloc[-1]
        )

        # Volume
        volume_now = float(volume.iloc[-1])
        avg_volume = float(
            volume.tail(20).mean()
        )

        if avg_volume <= 0:
            continue

        rvol = volume_now / avg_volume

        # Breakout 20 hari
        high20 = float(
            close.tail(20).max()
        )

        breakout = price >= high20

        # ==========================
        # SCORING
        # ==========================

        score = 0

        # RSI rendah (0-40)
        score += max(
            0,
            min(
                40,
                (50 - rsi_now) * 0.8
            )
        )

        # RSI rebound (0-20)
        if rsi_now > rsi_prev:

            score += min(
                20,
                (rsi_now - rsi_prev) * 5
            )

        # RVOL (0-20)
        score += min(
            20,
            rvol * 5
        )

        # Harga > EMA20
        if price > ema20:
            score += 10

        # EMA20 > EMA50
        if ema20 > ema50:
            score += 10

        # Breakout 20 hari
        if breakout:
            score += 15

        ticker = stock.replace(
            ".JK",
            ""
        )

        item = {
            "ticker": ticker,
            "score": round(score, 1),
            "rsi": round(rsi_now, 2),
            "price": round(price, 2),
            "rvol": round(rvol, 2),
            "breakout": breakout
        }

        results.append(item)

        # High conviction
        if score >= 80:
            high_conviction.append(item)

        # Extreme oversold
        if rsi_now < 20:
            extreme.append(item)

    except Exception:
        continue

# ==========================
# TOP SETUP
# ==========================

results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

msg1 = "🏆 TOP 10 SETUP BEI\n\n"

for item in results[:10]:

    tv = (
        f"https://www.tradingview.com/chart/?symbol=IDX:"
        f"{item['ticker']}"
    )

    breakout_text = (
        "🚀 Breakout20"
        if item["breakout"]
        else "-"
    )

    msg1 += (
        f"#{item['ticker']}\n"
        f"Score : {item['score']}\n"
        f"RSI(4): {item['rsi']}\n"
        f"RVOL : {item['rvol']}x\n"
        f"Harga : Rp {item['price']:,.0f}\n"
        f"{breakout_text}\n"
        f"{tv}\n\n"
    )

send(msg1)

# ==========================
# HIGH CONVICTION
# ==========================

if high_conviction:

    high_conviction = sorted(
        high_conviction,
        key=lambda x: x["score"],
        reverse=True
    )

    msg2 = (
        "🔥 HIGH CONVICTION SETUP\n\n"
    )

    for item in high_conviction[:10]:

        msg2 += (
            f"{item['ticker']}\n"
            f"Score : {item['score']}\n"
            f"RSI : {item['rsi']}\n"
            f"RVOL : {item['rvol']}x\n\n"
        )

    send(msg2)

# ==========================
# EXTREME OVERSOLD
# ==========================

if extreme:

    extreme = sorted(
        extreme,
        key=lambda x: x["rsi"]
    )

    msg3 = (
        "🚨 EXTREME OVERSOLD\n\n"
    )

    for item in extreme[:10]:

        msg3 += (
            f"{item['ticker']}\n"
            f"RSI(4): {item['rsi']}\n"
            f"Harga : Rp {item['price']:,.0f}\n\n"
        )

    send(msg3)
