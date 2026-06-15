import yfinance as yf
import requests
import os

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
    "ANTM.JK"
]

msg = "TEST DATA\n\n"

for stock in stocks:

    try:

        df = yf.download(
            stock,
            period="3mo",
            progress=False,
            auto_adjust=True
        )

        msg += (
            f"{stock}\n"
            f"Rows: {len(df)}\n"
            f"Last Close: {float(df['Close'].iloc[-1])}\n\n"
        )

    except Exception as e:

        msg += (
            f"{stock}\n"
            f"ERROR: {str(e)}\n\n"
        )

send(msg)
