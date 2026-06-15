
import pandas as pd
import yfinance as yf
import requests
import os

from ta.momentum import RSIIndicator

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STOCKS = [
    "AADI.JK","ACES.JK","ADMR.JK","ADRO.JK","AKRA.JK","AMMN.JK","AMRT.JK","ANTM.JK",
    "ARTO.JK","ASII.JK","AVIA.JK","BBCA.JK","BBNI.JK","BBRI.JK","BBTN.JK","BMRI.JK",
    "BRIS.JK","BRMS.JK","BREN.JK","BRPT.JK","BSDE.JK","BUKA.JK","BUMI.JK","CBDK.JK",
    "CMRY.JK","CPIN.JK","CTRA.JK","CUAN.JK","DEWA.JK","DSSA.JK","DSNG.JK","ELSA.JK",
    "EMTK.JK","ENRG.JK","ERAA.JK","ESSA.JK","EXCL.JK","GGRM.JK","GOTO.JK","HEAL.JK",
    "HRTA.JK","HRUM.JK","ICBP.JK","INCO.JK","INDF.JK","INDY.JK","INKP.JK","INTP.JK",
    "ISAT.JK","ITMG.JK","JPFA.JK","JSMR.JK","KIJA.JK","KLBF.JK","KPIG.JK","MAPA.JK",
    "MAPI.JK","MBMA.JK","MDKA.JK","MEDC.JK","MIKA.JK","MYOR.JK","PANI.JK","PGAS.JK",
    "PGEO.JK","PNLF.JK","PTBA.JK","PTRO.JK","PWON.JK","RAJA.JK","RATU.JK","SCMA.JK",
    "SIDO.JK","SMGR.JK","SMRA.JK","SSIA.JK","TAPG.JK","TLKM.JK","TOWR.JK","TPIA.JK",
    "UNTR.JK","AALI.JK","ADHI.JK","BBYB.JK","BKSL.JK","DGIK.JK","DMAS.JK","LSIP.JK",
    "MIDI.JK","MTEL.JK","PGJO.JK","PNBN.JK","PTPP.JK","SMDR.JK","TKIM.JK","UNVR.JK",
    "WIKA.JK","WSKT.JK"
]

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

        if rsi_now >= 15:
            continue

        price = float(close.iloc[-1])

        if price < 50:
            continue

        volume_now = float(df["Volume"].iloc[-1])
        avg_volume = float(df["Volume"].tail(20).mean())

        if volume_now < avg_volume:
            continue

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
