import os
import csv
import time
import requests
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_IDS = [
    6262086905,
    6003751935,
]

SETUPS_FILE = "setups.csv"
ACTIVE_FILE = "active_setups.csv"
ERROR_LOG_FILE = "error.log"

ALERT_EXPIRY_HOURS = 3


def log_error(text):
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {text}\n")
    except Exception:
        pass


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clean_columns(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def read_setups():
    if not os.path.exists(SETUPS_FILE):
        return []
    with open(SETUPS_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_active(setups):
    with open(ACTIVE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scan_time", "ticker", "signal", "score", "gap_pct",
            "price", "rsi", "rvol", "atr_pct", "entry_limit",
            "stop_loss", "tp1", "tp2", "tp3", "support",
            "vwap", "first_candle_rvol", "first_candle_body_pct", "reason"
        ])
        for s in setups:
            writer.writerow([
                s.get("scan_time", ""),
                s.get("ticker", ""),
                s.get("signal", ""),
                s.get("score", ""),
                s.get("gap_pct", ""),
                s.get("price", ""),
                s.get("rsi", ""),
                s.get("rvol", ""),
                s.get("atr_pct", ""),
                s.get("entry_limit", ""),
                s.get("stop_loss", ""),
                s.get("tp1", ""),
                s.get("tp2", ""),
                s.get("tp3", ""),
                s.get("support", ""),
                s.get("vwap", ""),
                s.get("first_candle_rvol", ""),
                s.get("first_candle_body_pct", ""),
                s.get("reason", ""),
            ])


def check_active_setups():
    setups = read_setups()
    if not setups:
        print("No setups.")
        return

    now = datetime.now()
    active = []

    for s in setups:
        scan_time_str = s.get("scan_time", "")
        if scan_time_str:
            try:
                scan_dt = datetime.strptime(scan_time_str, "%Y-%m-%d %H:%M")
                if (now - scan_dt) > timedelta(hours=ALERT_EXPIRY_HOURS):
                    continue
            except Exception:
                pass

        ticker = s.get("ticker", "")
        entry = safe_float(s.get("entry_limit"))
        if not ticker or entry is None:
            continue

        try:
            df = yf.download(
                ticker + ".JK",
                period="1d",
                interval="15m",
                auto_adjust=True,
                progress=False,
                timeout=10
            )
            df = clean_columns(df)
            if df is None or df.empty:
                active.append(s)
                continue

            current = safe_float(df["Close"].iloc[-1])
            low_today = safe_float(df["Low"].min())

            if current is None or low_today is None:
                active.append(s)
                continue

            # Tidak ada alert entry Telegram lagi - hanya update daftar aktif
            if low_today <= entry <= current:
                pass

            active.append(s)

        except Exception as e:
            log_error(f"check_active_setups {ticker} | {e}")
            active.append(s)

    write_active(active)
    print(f"Active setups: {len(active)} / {len(setups)}")


def main():
    print(f"Alert checker start - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    check_active_setups()


if __name__ == "__main__":
    main()
