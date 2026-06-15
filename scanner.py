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
        }
    )

stocks = [
    "BBCA.JK",
    "BBRI.JK",
    "BMRI.JK",
    "TLKM.JK",
    "ANTM.JK",
    "ASII.JK",
    "BRIS.JK",
    "MDKA.JK"
]

msg = "RSI CHECK\n\n"

for stock in stocks:

    try:

        df = yf.download(
            stock,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        close = df["Close"].squeeze()

        rsi = RSIIndicator(
            close,
            window=4
        ).rsi()

        rsi_now = float(rsi.iloc[-1])

        msg += f"{stock} = RSI {rsi_now:.2f}\n"

    except Exception as e:

        msg += f"{stock} ERROR\n"

send(msg)
