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
TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_IDS_RAW = os.getenv("TELEGRAM_CHAT_ID", "").strip()
CHAT_IDS = [x.strip() for x in CHAT_IDS_RAW.split(",") if x.strip()]

STOCKS_FILE = "stocks.txt"
ERROR_LOG_FILE = "error.log"
OUTPUT_CSV = "scan_results.csv"
TOP10_CSV = "top10_results.csv"

LOOKBACK = 120
MIN_PRICE = 50
VOLUME_MULTIPLIER = 1.8
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
    if not CHAT_IDS:
        print("TELEGRAM_CHAT_ID belum diset.")
        return

    chunks = [text[i:i + 3800] for i in range(0, len(text), 3800)]
    for chunk in chunks:
        for chat_id in CHAT_IDS:
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True
                    },
                    timeout=15
                )
                if not r.ok:
                    log_error(f"Telegram HTTP {r.status_code} | {r.text}")
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
        if df is None or len(df) < 60:
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


def detect_structure(df, lookback=20):
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    if len(df) < lookback + 5:
        return {
            "bias": "neutral",
            "bos": False,
            "choch": False,
            "last_swing_high": None,
            "last_swing_low": None
        }

    recent_high = np.max(highs[-lookback - 1:-1])
    recent_low = np.min(lows[-lookback - 1:-1])

    prev_high = np.max(highs[-lookback * 2:-lookback]) if len(df) >= lookback * 2 else recent_high
    prev_low = np.min(lows[-lookback * 2:-lookback]) if len(df) >= lookback * 2 else recent_low

    last_close = closes[-1]
    prev_close = closes[-2]

    bos = False
    choch = False
    bias = "neutral"

    if last_close > recent_high:
        bos = True
        bias = "bullish"
    elif last_close < recent_low:
        bos = True
        bias = "bearish"

    if prev_close <= prev_high and last_close > prev_high:
        choch = True
        bias = "bullish"
    elif prev_close >= prev_low and last_close < prev_low:
        choch = True
        bias = "bearish"

    return {
        "bias": bias,
        "bos": bos,
        "choch": choch,
        "last_swing_high": float(recent_high),
        "last_swing_low": float(recent_low)
    }


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

        if prev["Close"] < prev["Open"] and base["Close"] > base["Open"]:
            zones.append({
                "type": "DEMAND",
                "top": max(base["Close"], prev["Open"]),
                "bot": min(base["Open"], prev["Close"]),
                "strength": "Strong" if vol_ratio >= 4 else "Moderate",
                "vol_ratio": round(vol_ratio, 1)
            })

        if prev["Close"] > prev["Open"] and base["Close"] < base["Open"]:
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


def confidence_score(trend, rvol_value, patterns, in_zone, near_zone, structure):
    score = 0

    if "Strong Bullish" in trend or "Strong Bearish" in trend:
        score += 25
    elif "Bullish" in trend or "Bearish" in trend:
        score += 15

    if rvol_value >= 3:
        score += 20
    elif rvol_value >= 2:
        score += 15
    elif rvol_value >= 1.5:
        score += 10

    if "Breakout Candle" in patterns or "Bullish Engulfing" in patterns or "Hammer" in patterns or "Bearish Rejection" in patterns:
        score += 15

    if in_zone:
        score += 10
    elif near_zone:
        score += 5

    if structure["bos"]:
        score += 15
    if structure["choch"]:
        score += 10

    if structure["bias"] in ["bullish", "bearish"]:
        score += 5

    return min(100, score)


def calculate_signal(df):
    close = float(df["Close"].iloc[-1])
    trend, e20, e50, e200 = get_trend(df)
    volume_ratio = rvol(df)
    patterns = candle_patterns(df)
    zones = find_supply_demand(df)
    supply, demand = latest_zones(zones)
    structure = detect_structure(df)

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

    bullish = trend in ["🟢 Strong Bullish", "🟢 Bullish"] or structure["bias"] == "bullish"
    bearish = trend in ["🔴 Strong Bearish", "🔴 Bearish"] or structure["bias"] == "bearish"

    breakout = "Breakout Candle" in patterns or structure["bos"]
    reversal_bull = "Bullish Engulfing" in patterns or "Hammer" in patterns or structure["choch"]
    reversal_bear = "Bearish Rejection" in patterns or structure["choch"]

    if (in_demand or near_demand) and bullish and volume_ratio >= 1.8 and (breakout or reversal_bull):
        signal = "🟢 BUY"
    elif (in_supply or near_supply) and bearish and volume_ratio >= 1.8 and (breakout or reversal_bear):
        signal = "🔴 SELL"
    else:
        signal = "WATCH"

    score = confidence_score(
        trend,
        volume_ratio,
        patterns,
        in_demand or in_supply,
        near_demand or near_supply,
        structure
    )

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
        "e200": e200,
        "structure": structure
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


def score_to_rank(score, signal, volume):
    base = score
    if signal in ["🟢 BUY", "🔴 SELL"]:
        base += 15
    if volume >= 2:
        base += 10
    elif volume >= 1.5:
        base += 5
    return min(100, base)


def format_signal(ticker, result, plan):
    if result["signal"] == "🟢 BUY":
        return (
            f"📊 {ticker}\n\n"
            f"Signal : {result['signal']}\n"
            f"Confidence : {result['confidence']}%\n"
            f"Rank Score : {result['rank_score']}\n\n"
            f"Entry : {plan['entry']}\n"
            f"SL : {plan['sl']}\n"
            f"TP1 : {plan['tp1']}\n"
            f"TP2 : {plan['tp2']}\n"
            f"TP3 : {plan['tp3']}\n"
            f"R:R : 1:{plan['rr']}\n"
            f"Trend : {result['trend']}\n"
            f"Structure : BOS {result['structure']['bos']} | CHOCH {result['structure']['choch']} | Bias {result['structure']['bias']}\n"
            f"Volume : {result['volume']}x\n"
            f"Aksi : ✅ Boleh mulai cicil beli\n"
        )

    if result["signal"] == "🔴 SELL":
        return (
            f"📊 {ticker}\n\n"
            f"Signal : {result['signal']}\n"
            f"Confidence : {result['confidence']}%\n"
            f"Rank Score : {result['rank_score']}\n\n"
            f"Entry : {plan['entry']}\n"
            f"SL : {plan['sl']}\n"
            f"TP1 : {plan['tp1']}\n"
            f"TP2 : {plan['tp2']}\n"
            f"TP3 : {plan['tp3']}\n"
            f"R:R : 1:{plan['rr']}\n"
            f"Trend : {result['trend']}\n"
            f"Structure : BOS {result['structure']['bos']} | CHOCH {result['structure']['choch']} | Bias {result['structure']['bias']}\n"
            f"Volume : {result['volume']}x\n"
            f"Aksi : ❌ Jangan entry\n"
        )

    supply = "-"
    demand = "-"
    if result["supply"]:
        supply = f"{round(result['supply']['bot'])} - {round(result['supply']['top'])}"
    if result["demand"]:
        demand = f"{round(result['demand']['bot'])} - {round(result['demand']['top'])}"

    return (
        f"⚠ {ticker}\n\n"
        f"Signal : WATCH\n"
        f"Confidence : {result['confidence']}%\n"
        f"Rank Score : {result['rank_score']}\n"
        f"Trend : {result['trend']}\n"
        f"Structure : BOS {result['structure']['bos']} | CHOCH {result['structure']['choch']} | Bias {result['structure']['bias']}\n"
        f"Supply : {supply}\n"
        f"Demand : {demand}\n"
        f"Volume : {result['volume']}x\n"
        f"Aksi : ❌ Tunggu level valid\n"
    )


def scan_one(symbol):
    df = fetch_data(symbol)
    if df is None:
        return None

    if float(df["Close"].iloc[-1]) < MIN_PRICE:
        return None

    result = calculate_signal(df)
    plan = build_plan(result)
    result["rank_score"] = score_to_rank(result["confidence"], result["signal"], result["volume"])
    return result, plan


def main():
    stocks = read_stocks()
    if not stocks:
        print("stocks.txt kosong")
        return

    all_rows = []
    text_signals = []

    for s in stocks:
        try:
            scanned = scan_one(s)
            if scanned is None:
                continue

            result, plan = scanned
            ticker = s.replace(".JK", "")
            structure = result["structure"]

            row = {
                "ticker": ticker,
                "signal": result["signal"],
                "confidence": result["confidence"],
                "rank_score": result["rank_score"],
                "trend": result["trend"],
                "volume": result["volume"],
                "bos": structure["bos"],
                "choch": structure["choch"],
                "bias": structure["bias"]
            }
            all_rows.append(row)

            text_signals.append(format_signal(ticker, result, plan))

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            log_error(f"scan_one {s} | {e}")

    if not all_rows:
        send_telegram("Tidak ada sinyal valid hari ini.")
        return

    df_all = pd.DataFrame(all_rows).sort_values(
        ["rank_score", "confidence", "volume"],
        ascending=False
    )
    df_top10 = df_all.head(10).copy()

    df_all.to_csv(OUTPUT_CSV, index=False)
    df_top10.to_csv(TOP10_CSV, index=False)

    now = datetime.now().strftime("%d %b %H:%M")
    summary = (
        f"📡 DAILY STOCK RANKING - {now} WIB\n\n"
        f"Top 10 saham terbaik hari ini:\n\n"
    )

    for i, row in enumerate(df_top10.itertuples(index=False), start=1):
        summary += (
            f"{i}. {row.ticker} - {row.signal} - Score {row.rank_score} - Conf {row.confidence}% - Vol {row.volume}x\n"
        )

    send_telegram(summary)

    detailed_text = "\n━━━━━━━━━━━━━━━━━━\n\n".join(text_signals)
    if detailed_text:
        send_telegram(detailed_text)


if __name__ == "__main__":
    main()
