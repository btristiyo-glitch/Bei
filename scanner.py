import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# =====================
# CONFIG
# =====================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_IDS = [6262086905]

STOCKS_FILE = "stocks.txt"
ERROR_LOG_FILE = "error.log"

LOOKBACK = 120
MIN_PRICE = 50
VOLUME_MULTIPLIER = 2.0
REQUEST_DELAY = 2


# =====================
# UTIL
# =====================
def log_error(text):
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {text}\n")
    except Exception:
        pass


def send_telegram(text):
    if not TOKEN:
        print("TELEGRAM_TOKEN belum diset.")
        return

    chunks = [text[i:i + 3800] for i in range(0, len(text), 3800)]
    for chunk in chunks:
        for chat_id in CHAT_IDS:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True
                    },
                    timeout=15
                )
            except Exception as e:
                log_error(f"Telegram error | {e}")


def clean_df(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna()
    return df


def fetch_data(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        df = clean_df(df)
        if df is None or len(df) < 50:
            return None
        return df
    except Exception as e:
        log_error(f"fetch_data {symbol} | {e}")
        return None


def read_stocks():
    if not os.path.exists(STOCKS_FILE):
        return []
    with open(STOCKS_FILE, "r", encoding="utf-8") as f:
        stocks = [x.strip() for x in f if x.strip()]
    return [s if s.endswith(".JK") else s + ".JK" for s in stocks]


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rvol(df, length=20):
    vol = df["Volume"].astype(float)
    ma = vol.rolling(length).mean()
    if pd.isna(ma.iloc[-1]) or ma.iloc[-1] == 0:
        return 0.0
    return float(vol.iloc[-1] / ma.iloc[-1])


def get_trend(df):
    close = df["Close"].astype(float)
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1]
    last = close.iloc[-1]

    if last > e20 > e50 > e200:
        return "🟢 Strong Bullish", e20, e50, e200
    elif last < e20 < e50 < e200:
        return "🔴 Strong Bearish", e20, e50, e200
    elif e20 > e50:
        return "🟢 Bullish", e20, e50, e200
    elif e20 < e50:
        return "🔴 Bearish", e20, e50, e200
    else:
        return "⚪ Sideways", e20, e50, e200


def candle_patterns(df):
    if len(df) < 3:
        return []

    last = df.iloc[-1]
    prev = df.iloc[-2]
    patterns = []

    body = abs(last["Close"] - last["Open"])
    rng = max(last["High"] - last["Low"], 1e-9)
    upper_wick = last["High"] - max(last["Close"], last["Open"])
    lower_wick = min(last["Close"], last["Open"]) - last["Low"]

    if last["Close"] > last["Open"] and prev["Close"] < prev["Open"] and last["Close"] > prev["Open"] and last["Open"] < prev["Close"]:
        patterns.append("Bullish Engulfing")

    if lower_wick > body * 2 and body / rng < 0.4:
        patterns.append("Hammer")

    if upper_wick > body * 2 and last["Close"] < last["Open"]:
        patterns.append("Bearish Rejection")

    if last["Close"] > prev["High"] and last["Volume"] > df["Volume"].rolling(20).mean().iloc[-1]:
        patterns.append("Breakout Candle")

    return patterns


def find_supply_demand(df):
    zones = []
    vol_ma = df["Volume"].rolling(20).mean()

    for i in range(3, min(LOOKBACK, len(df) - 2)):
        prev = df.iloc[i - 1]
        base = df.iloc[i]

        if pd.isna(vol_ma.iloc[i]) or vol_ma.iloc[i] == 0:
            continue

        vol_ratio = base["Volume"] / vol_ma.iloc[i]
        if vol_ratio < VOLUME_MULTIPLIER:
            continue

        prev_body = abs(prev["Close"] - prev["Open"])
        base_body = abs(base["Close"] - base["Open"])
        prev_range = max(prev["High"] - prev["Low"], 1e-9)

        prev_body_pct = prev_body / prev_range
        base_body_pct = base_body / prev_range

        # DEMAND
        if (
            prev["Close"] < prev["Open"] and
            base["Close"] > base["Open"] and
            prev_body_pct > 0.5 and
            base_body_pct > 0.3
        ):
            zones.append({
                "type": "DEMAND",
                "top": max(base["Close"], prev["Open"]),
                "bot": min(base["Open"], prev["Close"]),
                "strength": "Strong" if vol_ratio >= 4 else "Moderate",
                "vol_ratio": round(vol_ratio, 1)
            })

        # SUPPLY
        if (
            prev["Close"] > prev["Open"] and
            base["Close"] < base["Open"] and
            prev_body_pct > 0.5 and
            base_body_pct > 0.3
        ):
            zones.append({
                "type": "SUPPLY",
                "top": max(prev["Close"], base["Open"]),
                "bot": min(base["Close"], prev["Open"]),
                "strength": "Strong" if vol_ratio >= 4 else "Moderate",
                "vol_ratio": round(vol_ratio, 1)
            })

    return zones


def latest_zones(zones):
    demand = None
    supply = None

    for z in reversed(zones):
        if z["type"] == "DEMAND" and demand is None:
            demand = z
        if z["type"] == "SUPPLY" and supply is None:
            supply = z
        if demand and supply:
            break

    return supply, demand


def confidence_score(trend, rvol_value, patterns, in_zone, near_zone):
    score = 0

    if "Strong Bullish" in trend:
        score += 30
    elif "Bullish" in trend:
        score += 20
    elif "Strong Bearish" in trend:
        score += 30
    elif "Bearish" in trend:
        score += 20

    if rvol_value >= 3:
        score += 25
    elif rvol_value >= 2:
        score += 18
    elif rvol_value >= 1.5:
        score += 10

    if "Breakout Candle" in patterns or "Bullish Engulfing" in patterns or "Bearish Rejection" in patterns:
        score += 20
    if in_zone:
        score += 15
    elif near_zone:
        score += 8

    return min(100, score)


def calculate_signal(df):
    close = float(df["Close"].iloc[-1])
    trend, e20, e50, e200 = get_trend(df)
    volume_ratio = rvol(df)
    patterns = candle_patterns(df)
    zones = find_supply_demand(df)
    supply, demand = latest_zones(zones)

    in_demand = False
    near_demand = False
    in_supply = False
    near_supply = False

    if demand:
        if demand["bot"] <= close <= demand["top"]:
            in_demand = True
        elif 0 < ((demand["bot"] - close) / demand["bot"]) * 100 <= 2:
            near_demand = True

    if supply:
        if supply["bot"] <= close <= supply["top"]:
            in_supply = True
        elif 0 < ((close - supply["top"]) / supply["top"]) * 100 <= 2:
            near_supply = True

    bullish = trend in ["🟢 Strong Bullish", "🟢 Bullish"]
    bearish = trend in ["🔴 Strong Bearish", "🔴 Bearish"]
    breakout = "Breakout Candle" in patterns
    bounce = "Bullish Engulfing" in patterns or "Hammer" in patterns
    rejection = "Bearish Rejection" in patterns

    if in_demand and bullish and volume_ratio >= 1.8 and (breakout or bounce):
        signal = "🟢 BUY"
    elif in_supply and bearish and volume_ratio >= 1.8 and (breakout or rejection):
        signal = "🔴 SELL"
    else:
        signal = "WATCH"

    score = confidence_score(trend, volume_ratio, patterns, in_demand or in_supply, near_demand or near_supply)

    return {
        "signal": signal,
        "confidence": score,
        "trend": trend,
        "volume": round(volume_ratio, 1),
        "patterns": patterns,
        "supply": supply,
        "demand": demand,
        "close": close,
        "e20": e20,
        "e50": e50,
        "e200": e200
    }


def build_plan(result):
    close = result["close"]
    atr_like = max(close * 0.02, 1)

    if result["signal"] == "🟢 BUY":
        entry_low = result["demand"]["bot"] if result["demand"] else close - atr_like
        entry_high = result["demand"]["top"] if result["demand"] else close
        stop_loss = round(entry_low - atr_like * 0.6)
        tp1 = round(close + atr_like * 1.0)
        tp2 = round(close + atr_like * 2.0)
        tp3 = round(close + atr_like * 3.0)
        rr = round((tp3 - close) / max(1, close - stop_loss), 1)
        return {
            "entry": f"{round(entry_low)} - {round(entry_high)}",
            "sl": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "rr": rr
        }

    if result["signal"] == "🔴 SELL":
        entry_low = result["supply"]["bot"] if result["supply"] else close
        entry_high = result["supply"]["top"] if result["supply"] else close + atr_like
        stop_loss = round(entry_high + atr_like * 0.6)
        tp1 = round(close - atr_like * 1.0)
        tp2 = round(close - atr_like * 2.0)
        tp3 = round(close - atr_like * 3.0)
        rr = round((close - tp3) / max(1, stop_loss - close), 1)
        return {
            "entry": f"{round(entry_low)} - {round(entry_high)}",
            "sl": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "rr": rr
        }

    return None


def format_buy(ticker, result, plan):
    return (
        f"📊 {ticker}\n\n"
        f"Signal : {result['signal']}\n"
        f"Confidence : {result['confidence']}%\n\n"
        f"━━━━━━━━━━━━\n\n"
        f"📍Entry\n{plan['entry']}\n\n"
        f"🛑 Stop Loss\n{plan['sl']}\n\n"
        f"🎯 Target\n"
        f"TP1 {plan['tp1']}\n"
        f"TP2 {plan['tp2']}\n"
        f"TP3 {plan['tp3']}\n\n"
        f"Risk Reward\n1 : {plan['rr']}\n\n"
        f"Breakout Chance\n{result['confidence']}%\n\n"
        f"Smart Money\n🟢 Accumulation\n\n"
        f"Trend\n{result['trend']}\n\n"
        f"Volume\n{result['volume']}x Average\n\n"
        f"Aksi:\n✅ Boleh mulai cicil beli\n"
    )


def format_sell(ticker, result, plan):
    return (
        f"📊 {ticker}\n\n"
        f"Signal : {result['signal']}\n"
        f"Confidence : {result['confidence']}%\n\n"
        f"━━━━━━━━━━━━\n\n"
        f"📍Entry\n{plan['entry']}\n\n"
        f"🛑 Stop Loss\n{plan['sl']}\n\n"
        f"🎯 Target\n"
        f"TP1 {plan['tp1']}\n"
        f"TP2 {plan['tp2']}\n"
        f"TP3 {plan['tp3']}\n\n"
        f"Risk Reward\n1 : {plan['rr']}\n\n"
        f"Breakout Chance\n{result['confidence']}%\n\n"
        f"Smart Money\n🔴 Distribution\n\n"
        f"Trend\n{result['trend']}\n\n"
        f"Volume\n{result['volume']}x Average\n\n"
        f"Aksi:\n❌ Jangan entry\n"
    )


def format_watch(ticker, result):
    supply = "-"
    demand = "-"
    if result["supply"]:
        supply = f"{round(result['supply']['bot'])} - {round(result['supply']['top'])}"
    if result["demand"]:
        demand = f"{round(result['demand']['bot'])} - {round(result['demand']['top'])}"

    action = "❌ Jangan entry\n✔ Tunggu breakout atau pullback"
    if result["signal"] == "WATCH":
        if result["demand"]:
            action = f"❌ Jangan entry\n✔ Tunggu breakout di atas {round(result['demand']['top'])}\natau pullback ke {round(result['demand']['bot'])}"
        elif result["supply"]:
            action = f"❌ Jangan entry\n✔ Tunggu breakdown di bawah {round(result['supply']['bot'])}\natau pullback ke {round(result['supply']['top'])}"

    return (
        f"⚠ {ticker}\n\n"
        f"Signal : WATCH\n\n"
        f"Confidence : {result['confidence']}%\n\n"
        f"Smart Money\n⚪ Neutral\n\n"
        f"Breakout Chance\n{result['confidence']}%\n\n"
        f"Supply : {supply}\n"
        f"Demand : {demand}\n\n"
        f"Aksi:\n{action}\n"
    )


def scan_one(symbol):
    df = fetch_data(symbol)
    if df is None:
        return None

    if float(df["Close"].iloc[-1]) < MIN_PRICE:
        return None

    result = calculate_signal(df)
    plan = build_plan(result)
    return result, plan


def main():
    stocks = read_stocks()
    if not stocks:
        print("stocks.txt kosong")
        return

    buys, sells, watches = [], [], []

    for s in stocks:
        try:
            scanned = scan_one(s)
            if scanned is None:
                continue

            result, plan = scanned
            ticker = s.replace(".JK", "")

            if result["signal"] == "🟢 BUY":
                buys.append(format_buy(ticker, result, plan))
            elif result["signal"] == "🔴 SELL":
                sells.append(format_sell(ticker, result, plan))
            else:
                watches.append(format_watch(ticker, result))

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            log_error(f"scan_one {s} | {e}")

    summary = (
        f"📡 S&D INTRADAY SCAN — {datetime.now().strftime('%d %b %H:%M')} WIB\n\n"
        f"🟢 BUY : {len(buys)} | 🔴 SELL : {len(sells)} | 👀 WATCH : {len(watches)}\n\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
    )

    send_telegram(summary)

    if buys:
        send_telegram("🟢 BUY SIGNALS\n\n" + "\n━━━━━━━━━━━━━━━━━━\n\n".join(buys))
    if sells:
        send_telegram("🔴 SELL SIGNALS\n\n" + "\n━━━━━━━━━━━━━━━━━━\n\n".join(sells))
    if watches:
        send_telegram("👀 WATCH LIST\n\n" + "\n━━━━━━━━━━━━━━━━━━\n\n".join(watches))


if __name__ == "__main__":
    main()
