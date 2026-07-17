import os
import csv
import time
import requests
import pandas as pd
import yfinance as yf

from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_IDS = [
    6262086905,
    6003751935,
]

STOCKS_FILE = "stocks.txt"
SECTORS_FILE = "sectors.csv"
SETUPS_FILE = "setups.csv"
SIGNALS_FILE = "signals.csv"
ERROR_LOG_FILE = "error.log"

BATCH_SIZE = 3
REQUEST_DELAY = 3

MIN_GAP_PCT = 2.0
MAX_GAP_PCT = 5.0
MIN_ATR_PCT = 3.0
MIN_DAILY_VALUE = 5_000_000_000
MIN_PRICE = 50

MIN_FIRST_CANDLE_RVOL = 1.5
MIN_FIRST_CANDLE_BODY_PCT = 0.35
VWAP_BUFFER_PCT = 0.0025


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


def send(msg, parse_mode="Markdown"):
    if not TOKEN:
        return
    chunks = [msg[i:i + 4000] for i in range(0, len(msg), 4000)]
    for chunk in chunks:
        for chat_id in CHAT_IDS:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "disable_web_page_preview": False,
                        "parse_mode": parse_mode,
                    },
                    timeout=10
                )
            except Exception as e:
                log_error(f"Telegram error chat_id={chat_id} | {e}")


def clean_columns(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def read_stocks():
    if not os.path.exists(STOCKS_FILE):
        return []
    with open(STOCKS_FILE, "r", encoding="utf-8") as f:
        stocks = [x.strip() for x in f if x.strip()]
    return [s if s.endswith(".JK") else s + ".JK" for s in stocks]


def read_sector_map():
    sector_map = {}
    if not os.path.exists(SECTORS_FILE):
        return sector_map

    with open(SECTORS_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row.get("ticker", "").upper().replace(".JK", "")
            sector = row.get("sector", "").upper()
            if ticker:
                sector_map[ticker] = sector
    return sector_map


def init_files():
    for file_path, header in [
        (SETUPS_FILE, [
            "scan_time", "ticker", "signal", "score", "gap_pct",
            "price", "rsi", "rvol", "atr_pct", "entry_limit",
            "stop_loss", "tp1", "tp2", "tp3", "support",
            "vwap", "first_candle_rvol", "first_candle_body_pct", "reason"
        ]),
        (SIGNALS_FILE, [
            "date", "time", "ticker", "score", "signal",
            "rsi", "price", "rvol", "gap_pct", "atr_pct",
            "entry_limit", "stop_loss", "tp1", "tp2", "tp3"
        ]),
    ]:
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def fetch_daily(symbol):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", auto_adjust=True, progress=False, timeout=15)
        df = clean_columns(df)
        if df is None or df.empty or len(df) < 30:
            return None
        return df
    except Exception as e:
        log_error(f"fetch_daily {symbol} | {e}")
        return None


def fetch_intraday(symbol):
    try:
        df = yf.download(symbol, period="5d", interval="15m", auto_adjust=True, progress=False, timeout=15)
        df = clean_columns(df)
        if df is None or df.empty or len(df) < 20:
            return None
        return df
    except Exception as e:
        log_error(f"fetch_intraday {symbol} | {e}")
        return None


def trend_regime(symbol):
    df = fetch_daily(symbol)
    if df is None:
        return "DATA ERR"

    close = df["Close"].squeeze()
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]

    if price > ema20 > ema50:
        return "BULLISH"
    if price < ema20 < ema50:
        return "BEARISH"
    return "NEUTRAL"


def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker + ".JK")
        info = stock.info or {}
        return info.get("priceToBook"), info.get("trailingPE"), info.get("marketCap")
    except Exception as e:
        log_error(f"get_fundamentals {ticker} | {e}")
        return None, None, None


def calc_vwap(intraday_df):
    try:
        tp = (intraday_df["High"] + intraday_df["Low"] + intraday_df["Close"]) / 3
        vol = intraday_df["Volume"].replace(0, pd.NA)
        vwap = (tp * vol).cumsum() / vol.cumsum()
        return safe_float(vwap.iloc[-1])
    except Exception as e:
        log_error(f"calc_vwap | {e}")
        return None


def first_candle_filter(intraday_df):
    try:
        if intraday_df is None or intraday_df.empty or len(intraday_df) < 2:
            return False, None, None

        first = intraday_df.iloc[0]
        second = intraday_df.iloc[1]

        first_open = safe_float(first["Open"])
        first_close = safe_float(first["Close"])
        first_high = safe_float(first["High"])
        first_low = safe_float(first["Low"])
        first_vol = safe_float(first["Volume"], 0)
        second_vol = safe_float(second["Volume"], 0)

        if None in [first_open, first_close, first_high, first_low] or first_open == 0:
            return False, None, None

        body_pct = abs(first_close - first_open) / first_open * 100
        avg_15m_vol = safe_float(intraday_df["Volume"].head(4).mean())
        if avg_15m_vol is None or avg_15m_vol <= 0:
            return False, None, None

        rvol_first = first_vol / avg_15m_vol

        candle_up = first_close > first_open
        follow_through = second_vol >= first_vol * 0.7

        ok = (
            candle_up and
            body_pct >= MIN_FIRST_CANDLE_BODY_PCT and
            rvol_first >= MIN_FIRST_CANDLE_RVOL and
            follow_through
        )
        return ok, round(rvol_first, 2), round(body_pct, 2)
    except Exception as e:
        log_error(f"first_candle_filter | {e}")
        return False, None, None


def compute_stop_and_targets(df, entry_price):
    try:
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_val = safe_float(atr.iloc[-1])
        swing_low = safe_float(low.tail(10).min())

        if atr_val is None or swing_low is None:
            return None, None, None, None, None, None

        stop_loss = swing_low - (0.5 * atr_val)
        if stop_loss >= entry_price:
            stop_loss = entry_price - atr_val

        risk = entry_price - stop_loss
        if risk <= 0:
            return None, None, None, None, None, None

        tp1 = entry_price + (1.5 * risk)
        tp2 = entry_price + (2.5 * risk)
        tp3 = entry_price + (3.5 * risk)

        atr_pct = (atr_val / entry_price) * 100

        return (
            round(stop_loss, 0),
            round(tp1, 0),
            round(tp2, 0),
            round(tp3, 0),
            round(atr_val, 2),
            round(atr_pct, 2),
        )
    except Exception as e:
        log_error(f"compute_stop_and_targets | {e}")
        return None, None, None, None, None, None


def compute_vwap_retest_entry(df_intraday, support, price, vwap):
    try:
        if df_intraday is None or df_intraday.empty:
            return None

        recent = df_intraday.tail(8)
        last_low = safe_float(recent["Low"].min())
        last_close = safe_float(recent["Close"].iloc[-1])

        if None in [last_low, last_close, vwap]:
            return None

        if support and support < price:
            entry = max(support, vwap * 0.999)
        else:
            entry = max(last_low, vwap * (1 - VWAP_BUFFER_PCT))

        if last_close < vwap:
            return None

        return round(entry, 0)
    except Exception as e:
        log_error(f"compute_vwap_retest_entry | {e}")
        return None


def scan_symbol(symbol, sector_map, market_bullish):
    daily = fetch_daily(symbol)
    intraday = fetch_intraday(symbol)
    if daily is None or intraday is None:
        return [], []

    try:
        close = daily["Close"].squeeze()
        high = daily["High"].squeeze()
        low = daily["Low"].squeeze()
        volume = daily["Volume"].squeeze()

        price = safe_float(close.iloc[-1])
        if price is None:
            return [], []

        rsi = RSIIndicator(close, window=4).rsi()
        rsi_now = safe_float(rsi.iloc[-1])
        rsi_prev = safe_float(rsi.iloc[-2])

        ema20 = safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = safe_float(close.ewm(span=50, adjust=False).mean().iloc[-1])

        avg_vol = safe_float(volume.tail(20).mean())
        daily_value = price * safe_float(volume.iloc[-1], 0)
        if avg_vol is None or avg_vol <= 0:
            return [], []

        rvol = safe_float(volume.iloc[-1], 0) / avg_vol

        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_val = safe_float(atr.iloc[-1])
        atr_pct = (atr_val / price * 100) if atr_val and price else None
        if atr_pct is None:
            return [], []

        prev_20_high = safe_float(close.shift(1).tail(20).max())
        prev_20_low = safe_float(close.shift(1).tail(20).min())
        breakout_high = safe_float(close.iloc[-2]) > prev_20_high if prev_20_high else False

        daily_intraday = intraday.resample("D").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

        if len(daily_intraday) < 2:
            return [], []

        open_today = safe_float(daily_intraday["Open"].iloc[-1])
        close_yest = safe_float(daily_intraday["Close"].iloc[-2])
        gap_pct = ((open_today - close_yest) / close_yest * 100) if open_today and close_yest else 0

        vwap = calc_vwap(intraday)
        first_ok, first_rvol, first_body_pct = first_candle_filter(intraday)

        pbv, per, mcap = get_fundamentals(symbol.replace(".JK", ""))
        fundament_flag = ""
        if pbv is not None and per is not None and mcap is not None:
            if pbv < 2 and per < 25 and 500_000_000_000 < mcap < 50_000_000_000_000:
                fundament_flag = "FUNDAMENTAL OK"

        if atr_pct < MIN_ATR_PCT or daily_value < MIN_DAILY_VALUE or price < MIN_PRICE:
            return [], []

        if not (MIN_GAP_PCT <= gap_pct <= MAX_GAP_PCT):
            return [], []

        ticker_clean = symbol.replace(".JK", "")
        sector = sector_map.get(ticker_clean, "OTHER")

        score = 0
        score += 30  # gap wajib
        if rvol > 5:
            score += 25
        elif rvol > 3:
            score += 15
        elif rvol > 2:
            score += 8

        if breakout_high:
            score += 10
        if rsi_now and rsi_prev and rsi_now > rsi_prev:
            score += 8
        if ema20 and price > ema20:
            score += 5
        if ema20 and ema50 and ema20 > ema50:
            score += 5
        if market_bullish:
            score += 5
        if 3 < ((prev_20_high - prev_20_low) / prev_20_low * 100) < 25 if prev_20_high and prev_20_low else False:
            score += 5
        if first_ok:
            score += 10
        if vwap and price > vwap:
            score += 5

        if score < 40:
            return [], []

        entry = compute_vwap_retest_entry(intraday, safe_float(intraday["Low"].tail(16).min()), price, vwap)
        if entry is None:
            return [], []

        stop_loss, tp1, tp2, tp3, atr_used, atr_used_pct = compute_stop_and_targets(daily, entry)
        if stop_loss is None:
            return [], []

        if entry <= stop_loss:
            return [], []

        rr = round((tp1 - entry) / (entry - stop_loss), 2) if entry > stop_loss else 0
        if rr < 2:
            return [], []

        row = {
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ticker": ticker_clean,
            "signal": "BREAKOUT",
            "score": round(score, 1),
            "gap_pct": round(gap_pct, 2),
            "price": round(price, 0),
            "rsi": round(rsi_now, 1) if rsi_now else 0,
            "rvol": round(rvol, 2),
            "atr_pct": round(atr_pct, 2),
            "entry_limit": round(entry, 0),
            "stop_loss": round(stop_loss, 0),
            "tp1": round(tp1, 0),
            "tp2": round(tp2, 0),
            "tp3": round(tp3, 0),
            "support": round(safe_float(intraday["Low"].tail(16).min()) or 0, 0),
            "vwap": round(vwap, 0) if vwap else 0,
            "first_candle_rvol": first_rvol if first_rvol else 0,
            "first_candle_body_pct": first_body_pct if first_body_pct else 0,
            "reason": "Gap 2-5% + VWAP retest + first candle volume confirmation",
        }

        return [row], [row]
    except Exception as e:
        log_error(f"scan_symbol {symbol} | {e}")
        return [], []


def main():
    init_files()

    stocks = read_stocks()
    sector_map = read_sector_map()

    if not stocks:
        print("stocks.txt kosong.")
        return

    rows_setup = []
    rows_signal = []

    for i, symbol in enumerate(stocks):
        print(f"Scanning {i + 1}/{len(stocks)} - {symbol}")
        market_bullish = trend_regime("^JKSE") == "BULLISH"
        setup_rows, signal_rows = scan_symbol(symbol, sector_map, market_bullish)

        if setup_rows:
            rows_setup.extend(setup_rows)
        if signal_rows:
            rows_signal.extend(signal_rows)

        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(REQUEST_DELAY)

    with open(SETUPS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scan_time", "ticker", "signal", "score", "gap_pct",
            "price", "rsi", "rvol", "atr_pct", "entry_limit",
            "stop_loss", "tp1", "tp2", "tp3", "support",
            "vwap", "first_candle_rvol", "first_candle_body_pct", "reason"
        ])
        for row in rows_setup:
            writer.writerow(row)

    with open(SIGNALS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "time", "ticker", "score", "signal",
            "rsi", "price", "rvol", "gap_pct", "atr_pct",
            "entry_limit", "stop_loss", "tp1", "tp2", "tp3"
        ])
        now = datetime.now()
        for row in rows_signal:
            writer.writerow({
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M"),
                "ticker": row["ticker"],
                "score": row["score"],
                "signal": row["signal"],
                "rsi": row["rsi"],
                "price": row["price"],
                "rvol": row["rvol"],
                "gap_pct": row["gap_pct"],
                "atr_pct": row["atr_pct"],
                "entry_limit": row["entry_limit"],
                "stop_loss": row["stop_loss"],
                "tp1": row["tp1"],
                "tp2": row["tp2"],
                "tp3": row["tp3"],
            })

    print(f"Done - setups: {len(rows_setup)} | signals: {len(rows_signal)}")


if __name__ == "__main__":
    main()
