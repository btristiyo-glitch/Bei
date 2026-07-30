import os
import csv
import time
import math
import schedule
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_IDS = [6262086905]

CSV_FILE = "signals.csv"
ALERT_FILE = "alerts.csv"
STOCKS_FILE = "stocks.txt"
SECTORS_FILE = "sectors.csv"
ERROR_LOG_FILE = "error.log"

BATCH_SIZE = 3
REQUEST_DELAY = 3
ALERT_EXPIRY_HOURS = 3

LOOKBACK_CANDLES = 120
AD_WINDOW = 15
MIN_PRICE = 50
MIN_ZONE_VOLUME = 100_000_000

VOLUME_THRESHOLD = 2.0
ZONE_DISTANCE_PCT = 2.5


def log_error(text):
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {text}\n")
    except Exception:
        pass


def send(msg, parse_mode="Markdown"):
    if not TOKEN:
        print("Telegram token belum diset.")
        return

    chunks = [msg[i:i + 3800] for i in range(0, len(msg), 3800)]
    for chunk in chunks:
        for chat_id in CHAT_IDS:
            for attempt in range(3):
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
                    break
                except Exception as e:
                    if attempt == 2:
                        log_error(f"Telegram error chat_id={chat_id} | {e}")
                    time.sleep(2)


def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
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


def fetch_data(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            timeout=15
        )
        df = clean_columns(df)
        if df is None or df.empty or len(df) < 30:
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


def read_sector_map():
    sector_map = {}
    if not os.path.exists(SECTORS_FILE):
        return sector_map
    with open(SECTORS_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row.get("ticker", "").upper().replace(".JK", "")
            sector = row.get("sector", "").upper().strip()
            if ticker:
                sector_map[ticker] = sector
    return sector_map


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def slope_last_n(series, n=5):
    y = series.iloc[-n:].astype(float).values
    if len(y) < n or np.any(pd.isna(y)):
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    num = np.sum((x - x_mean) * (y - y_mean))
    return float(num / denom)


def zscore(series):
    s = series.astype(float)
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0
    return (s - s.mean()) / std


def calculate_rvol(df):
    vol = df["Volume"].astype(float)
    ma20 = vol.rolling(20).mean()
    last_ma = ma20.iloc[-1]
    if pd.isna(last_ma) or last_ma <= 0:
        return 0.0
    return float(vol.iloc[-1] / last_ma)


def get_candle_patterns(df):
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

    if body / rng < 0.25:
        patterns.append("Inside Bar")

    if last["Close"] < last["Open"] and upper_wick > body * 2:
        patterns.append("Bearish Rejection")

    if last["Close"] > prev["High"] and last["Volume"] > df["Volume"].rolling(20).mean().iloc[-1]:
        patterns.append("Breakout Candle")

    return patterns


def calculate_trend_ema(df):
    close = df["Close"].astype(float)
    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    ema200 = ema(close, 200).iloc[-1]
    price = close.iloc[-1]

    bullish = price > ema20 > ema50 > ema200
    bearish = price < ema20 < ema50 < ema200

    if bullish:
        trend = "🟢 Strong Bullish"
    elif bearish:
        trend = "🔴 Strong Bearish"
    elif ema20 > ema50:
        trend = "🟢 Bullish"
    elif ema20 < ema50:
        trend = "🔴 Bearish"
    else:
        trend = "⚪ Sideways"

    return {
        "ema20": float(ema20),
        "ema50": float(ema50),
        "ema200": float(ema200),
        "trend": trend
    }


def market_phase(df):
    close = df["Close"].astype(float)
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1]

    if close.iloc[-1] > e20 > e50 > e200:
        return "Markup"
    if close.iloc[-1] < e20 < e50 < e200:
        return "Markdown"
    if e20 > e50 and e20 > e200:
        return "Accumulation"
    if e20 < e50 and e50 > e200:
        return "Distribution"
    return "Sideways"


def market_structure(df):
    close = df["Close"].astype(float)
    if len(close) < 10:
        return "Sideways"
    hh = close.iloc[-1] > close.iloc[-3] > close.iloc[-5]
    hl = close.iloc[-2] > close.iloc[-4] > close.iloc[-6]
    lh = close.iloc[-1] < close.iloc[-3] < close.iloc[-5]
    ll = close.iloc[-2] < close.iloc[-4] < close.iloc[-6]
    if hh and hl:
        return "Higher High & Higher Low"
    if lh and ll:
        return "Lower High & Lower Low"
    return "Sideways"


def find_intraday_zones(df):
    zones = []
    vol_ma = df["Volume"].rolling(20).mean()

    for i in range(3, min(LOOKBACK_CANDLES, len(df) - 2)):
        prev = df.iloc[i - 1]
        base = df.iloc[i]

        avg_vol = safe_float(vol_ma.iloc[i])
        if not avg_vol:
            continue

        base_vol = safe_float(base["Volume"])
        if not base_vol:
            continue

        vol_ratio = base_vol / avg_vol
        if vol_ratio < VOLUME_THRESHOLD:
            continue

        range_prev = safe_float(prev["High"] - prev["Low"])
        if not range_prev:
            continue

        body_prev = abs(safe_float(prev["Close"]) - safe_float(prev["Open"]))
        body_base = abs(safe_float(base["Close"]) - safe_float(base["Open"]))
        if body_prev is None or body_base is None:
            continue

        body_pct_prev = body_prev / range_prev
        body_pct_base = body_base / range_prev

        if (
            body_pct_prev > 0.6 and
            safe_float(prev["Close"]) < safe_float(prev["Open"]) and
            safe_float(base["Close"]) > safe_float(base["Open"]) and
            body_pct_base > 0.4 and
            base_vol >= MIN_ZONE_VOLUME
        ):
            zones.append({
                "type": "DEMAND",
                "top": max(safe_float(base["Close"]), safe_float(prev["Open"])),
                "bot": min(safe_float(base["Open"]), safe_float(prev["Close"])),
                "vol_ratio": round(vol_ratio, 1),
                "strength": "Strong" if vol_ratio >= 4 else "Moderate"
            })

        elif (
            body_pct_prev > 0.6 and
            safe_float(prev["Close"]) > safe_float(prev["Open"]) and
            safe_float(base["Close"]) < safe_float(base["Open"]) and
            body_pct_base > 0.4 and
            base_vol >= MIN_ZONE_VOLUME
        ):
            zones.append({
                "type": "SUPPLY",
                "top": max(safe_float(prev["Close"]), safe_float(base["Open"])),
                "bot": min(safe_float(base["Close"]), safe_float(prev["Open"])),
                "vol_ratio": round(vol_ratio, 1),
                "strength": "Strong" if vol_ratio >= 4 else "Moderate"
            })

    return zones


def detect_accumulation_distribution(df):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    hl_range = (high - low).replace(0, np.nan)
    mfm = (((close - low) - (high - close)) / hl_range).fillna(0).clip(-1, 1)
    adl = (mfm * volume).cumsum()

    price_diff = close.diff().fillna(0)
    obv_step = np.where(price_diff > 0, volume, np.where(price_diff < 0, -volume, 0))
    obv = pd.Series(obv_step, index=df.index).cumsum()

    pct_change = close.pct_change().fillna(0)
    vpt = (volume * pct_change).cumsum()

    def get_slope(series):
        chunk = series.iloc[-AD_WINDOW:]
        if len(chunk) < 5:
            return 0.0
        z = zscore(chunk)
        return slope_last_n(z, 5)

    price_slope = get_slope(close)
    adl_slope = get_slope(adl)
    obv_slope = get_slope(obv)
    vpt_slope = get_slope(vpt)

    slopes = [adl_slope, obv_slope, vpt_slope]
    up_count = sum(s > 0 for s in slopes)
    down_count = 3 - up_count
    avg_slope = float(np.mean(slopes))

    bull_div = price_slope < 0 and up_count >= 2
    bear_div = price_slope > 0 and down_count >= 2

    if up_count >= 2 and avg_slope > 0:
        label = "🟢 Strong Accumulation" if bull_div else "🟢 Accumulation"
    elif down_count >= 2 and avg_slope < 0:
        label = "🔴 Strong Distribution" if bear_div else "🔴 Distribution"
    else:
        label = "⚪ Neutral"

    return {"status": label, "score": round(avg_slope, 4)}


def market_health_engine():
    try:
        ihsg = fetch_data("^JKSE", period="6mo", interval="1d")
        if ihsg is None:
            ihsg_trend = "⚪ Unknown"
            ihsg_score = 50
        else:
            close = ihsg["Close"].astype(float)
            e20 = ema(close, 20).iloc[-1]
            e50 = ema(close, 50).iloc[-1]
            e200 = ema(close, 200).iloc[-1]
            last = close.iloc[-1]
            if last > e20 > e50 > e200:
                ihsg_trend = "🟢 Bullish"
                ihsg_score = 80
            elif last < e20 < e50 < e200:
                ihsg_trend = "🔴 Bearish"
                ihsg_score = 20
            else:
                ihsg_trend = "⚪ Sideways"
                ihsg_score = 50

        stocks = read_stocks()
        bull = 0
        bear = 0
        score_list = []

        for s in stocks[:40]:
            try:
                df = fetch_data(s, period="3mo", interval="1d")
                if df is None:
                    continue
                close = df["Close"].astype(float)
                e20 = ema(close, 20).iloc[-1]
                e50 = ema(close, 50).iloc[-1]
                e200 = ema(close, 200).iloc[-1]
                last = close.iloc[-1]

                if last > e20 > e50 > e200:
                    bull += 1
                    score_list.append(1)
                elif last < e20 < e50 < e200:
                    bear += 1
                    score_list.append(-1)
            except Exception:
                continue

        total = max(1, bull + bear)
        flow_score = int(round(50 + ((bull - bear) / total) * 25))
        market_score = int(max(0, min(100, round((ihsg_score * 0.45) + (flow_score * 0.55)))))

        if market_score > 80:
            flow_label = "🟢 Positive"
        elif market_score >= 50:
            flow_label = "🟡 Neutral"
        else:
            flow_label = "🔴 Negative"

        return {
            "ihsg_trend": ihsg_trend,
            "bullish_stocks": bull,
            "bearish_stocks": bear,
            "money_flow": flow_label,
            "market_score": market_score
        }
    except Exception as e:
        log_error(f"market_health_engine | {e}")
        return {
            "ihsg_trend": "⚪ Unknown",
            "bullish_stocks": 0,
            "bearish_stocks": 0,
            "money_flow": "⚪ Unknown",
            "market_score": 50
        }


def classify_signal(score, market_score):
    if market_score < 30:
        return "WATCH"
    if score >= 90:
        return "🟢 BUY"
    if score >= 80:
        return "🟢 BUY"
    if score >= 65:
        return "WATCH"
    if score >= 50:
        return "WAIT"
    return "AVOID"


def classify_risk(atr_pct):
    if atr_pct < 3:
        return "Low"
    if atr_pct < 6:
        return "Medium"
    return "High"


def calculate_breakout_probability(rvol, acc_status, trend, rsi_value, breakout_candle):
    prob = 25
    if rvol >= 5:
        prob += 30
    elif rvol >= 3:
        prob += 20
    elif rvol >= 2:
        prob += 12
    if "Accumulation" in acc_status:
        prob += 15
    if "Bullish" in trend:
        prob += 10
    if 45 <= rsi_value <= 65:
        prob += 8
    if breakout_candle:
        prob += 15
    return max(1, min(99, int(round(prob))))


def calculate_confidence_components(data):
    comps = {}

    trend_score = 0
    if data["ema20"] > data["ema50"] > data["ema200"]:
        trend_score = 30
    elif data["ema20"] > data["ema50"]:
        trend_score = 20
    elif data["ema20"] > data["ema50"] * 0.99:
        trend_score = 10
    comps["trend"] = trend_score

    sd_score = 0
    if data["price_in_demand"]:
        sd_score = 25
    elif data["price_near_demand"]:
        sd_score = 20
    elif data["price_mid_range"]:
        sd_score = 10
    elif data["price_near_supply"]:
        sd_score = 5
    comps["supplydemand"] = sd_score

    vol_score = 0
    if data["rvol"] > 3:
        vol_score = 15
    elif data["rvol"] >= 2:
        vol_score = 12
    elif data["rvol"] >= 1.5:
        vol_score = 8
    elif data["rvol"] > 0.8:
        vol_score = 5
    comps["volume"] = vol_score

    pa_score = 0
    if data["breakout_candle"]:
        pa_score = 15
    elif "Bullish Engulfing" in data["patterns"]:
        pa_score = 12
    elif "Hammer" in data["patterns"]:
        pa_score = 10
    elif "Inside Bar" in data["patterns"]:
        pa_score = 5
    comps["price_action"] = pa_score

    mom_score = 0
    rsi_value = data["rsi"]
    if 50 <= rsi_value <= 65:
        mom_score = 10
    elif 45 <= rsi_value < 50:
        mom_score = 8
    elif 35 <= rsi_value < 45:
        mom_score = 5
    elif rsi_value > 75:
        mom_score = 2
    elif rsi_value < 30:
        mom_score = 3
    comps["momentum"] = mom_score

    struct_score = 0
    if data["structure"] == "Higher High & Higher Low":
        struct_score = 5
    elif data["structure"] == "Sideways":
        struct_score = 3
    comps["structure"] = struct_score

    score = sum(comps.values())
    return min(100, int(round(score))), comps


def score_to_label(score):
    if score >= 90:
        return "🔥 STRONG BUY"
    if score >= 80:
        return "🟢 BUY"
    if score >= 65:
        return "👀 WATCH"
    if score >= 50:
        return "⚠ WAIT"
    return "❌ AVOID"


def build_trading_plan(price, atr_value, demand, supply, signal_action):
    if signal_action in ["🟢 BUY", "WATCH"]:
        if demand:
            d1, d2 = [float(x) for x in demand.replace("–", "-").split("-")]
            entry_bot, entry_top = d1, d2
        else:
            entry_bot, entry_top = price - atr_value, price

        stop_loss = int(round(entry_bot - atr_value * 0.8))
        tp1 = int(round(price + atr_value * 1.0))
        tp2 = int(round(price + atr_value * 2.0))
        tp3 = int(round(price + atr_value * 3.0))
        rr = round((tp3 - price) / max(1, price - stop_loss), 1)
        return {
            "entry": f"{int(round(entry_bot))} - {int(round(entry_top))}",
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "rr": rr
        }

    if signal_action == "🔴 SELL":
        if supply:
            s1, s2 = [float(x) for x in supply.replace("–", "-").split("-")]
            entry_bot, entry_top = s1, s2
        else:
            entry_bot, entry_top = price, price + atr_value

        stop_loss = int(round(entry_top + atr_value * 0.8))
        tp1 = int(round(price - atr_value * 1.0))
        tp2 = int(round(price - atr_value * 2.0))
        tp3 = int(round(price - atr_value * 3.0))
        rr = round((price - tp3) / max(1, stop_loss - price), 1)
        return {
            "entry": f"{int(round(entry_bot))} - {int(round(entry_top))}",
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "rr": rr
        }

    return None


def scan_ticker(symbol, market_score):
    df = fetch_data(symbol, period="6mo", interval="1d")
    if df is None or len(df) < 60:
        return None

    current_price = safe_float(df["Close"].iloc[-1])
    if current_price is None or current_price < MIN_PRICE:
        return None

    trend = calculate_trend_ema(df)
    phase = market_phase(df)
    rsi_value = float(rsi(df["Close"], 14).iloc[-1])
    atr_value = float(atr(df, 14).iloc[-1])
    atr_pct = (atr_value / current_price * 100) if current_price > 0 else 0
    rvol = calculate_rvol(df)
    patterns = get_candle_patterns(df)
    structure = market_structure(df)
    acc_dist = detect_accumulation_distribution(df)
    zones = find_intraday_zones(df.tail(120))

    supply = None
    demand = None
    for z in zones:
        if z["type"] == "SUPPLY":
            supply = f"{int(round(z['bot']))}–{int(round(z['top']))}"
        if z["type"] == "DEMAND":
            demand = f"{int(round(z['bot']))}–{int(round(z['top']))}"

    price_in_demand = False
    price_near_demand = False
    price_mid_range = False
    price_near_supply = False

    if demand:
        d1, d2 = [float(x) for x in demand.replace("–", "-").split("-")]
        if d1 <= current_price <= d2:
            price_in_demand = True
        elif 0 < (d1 - current_price) / d1 * 100 <= 1:
            price_near_demand = True

    if supply:
        s1, s2 = [float(x) for x in supply.replace("–", "-").split("-")]
        if 0 < (current_price - s2) / s2 * 100 <= 1:
            price_near_supply = True

    if not any([price_in_demand, price_near_demand, price_near_supply]):
        price_mid_range = True

    breakout_candle = "Breakout Candle" in patterns
    confidence, comps = calculate_confidence_components({
        "ema20": trend["ema20"],
        "ema50": trend["ema50"],
        "ema200": trend["ema200"],
        "price_in_demand": price_in_demand,
        "price_near_demand": price_near_demand,
        "price_mid_range": price_mid_range,
        "price_near_supply": price_near_supply,
        "rvol": rvol,
        "breakout_candle": breakout_candle,
        "patterns": patterns,
        "rsi": rsi_value,
        "structure": structure
    })

    signal = classify_signal(confidence, market_score)
    breakout_prob = calculate_breakout_probability(rvol, acc_dist["status"], trend["trend"], rsi_value, breakout_candle)
    risk = classify_risk(atr_pct)
    plan = build_trading_plan(current_price, atr_value, demand, supply, signal)

    if market_score < 30 and signal == "🟢 BUY":
        signal = "👀 WATCH"

    return {
        "ticker": symbol.replace(".JK", ""),
        "price": round(current_price),
        "signal": signal,
        "confidence": confidence,
        "trend": trend["trend"],
        "market_phase": phase,
        "volume": round(rvol, 1),
        "smart_money": acc_dist["status"],
        "breakout_prob": breakout_prob,
        "risk": risk,
        "entry": plan["entry"] if plan else "-",
        "stop_loss": plan["stop_loss"] if plan else "-",
        "tp1": plan["tp1"] if plan else "-",
        "tp2": plan["tp2"] if plan else "-",
        "tp3": plan["tp3"] if plan else "-",
        "rr": plan["rr"] if plan else "-",
        "demand": demand or "-",
        "supply": supply or "-",
        "components": comps,
        "market_score": market_score,
        "rsi": round(rsi_value, 1)
    }


def market_health_text(mh):
    return (
        "📊 MARKET HEALTH\n\n"
        f"IHSG Trend\n{mh['ihsg_trend']}\n\n"
        f"Saham Bullish\n{mh['bullish_stocks']}\n\n"
        f"Saham Bearish\n{mh['bearish_stocks']}\n\n"
        f"Money Flow\n{mh['money_flow']}\n\n"
        f"Market Score\n{mh['market_score']}/100\n"
    )


def build_buy_text(item):
    return (
        f"📊 {item['ticker']}\n\n"
        f"Signal : {item['signal']}\n"
        f"Confidence : {item['confidence']}/100\n\n"
        f"━━━━━━━━━━━━\n\n"
        f"📍Entry\n{item['entry']}\n\n"
        f"🛑 Stop Loss\n{item['stop_loss']} (-{round(abs(item['price'] - item['stop_loss']) / item['price'] * 100, 1)}%)\n\n"
        f"🎯 Target\nTP1 {item['tp1']} (+2%)\nTP2 {item['tp2']} (+4.5%)\nTP3 {item['tp3']} (+8%)\n\n"
        f"Risk Reward\n1 : {item['rr']}\n\n"
        f"Breakout Chance\n{item['breakout_prob']}%\n\n"
        f"Smart Money\n{item['smart_money']}\n\n"
        f"Trend\n{item['trend']}\n\n"
        f"Volume\n{item['volume']}x Average\n\n"
        f"Aksi:\n✅ Boleh mulai cicil beli\n"
    )


def build_watch_text(item):
    breakout_hint = item["supply"] if item["signal"] != "👀 WATCH" else item["supply"]
    pullback_hint = item["demand"]
    return (
        f"👀 {item['ticker']} @ {item['price']}\n\n"
        f"Status : WATCH\n\n"
        f"Phase : {item['market_phase']}\n"
        f"Trend : {'Sideways' if 'Sideways' in item['trend'] else item['trend']}\n"
        f"Confidence : {item['confidence']}/100\n\n"
        f"Smart Money\n{item['smart_money']}\n\n"
        f"Breakout Chance\n{item['breakout_prob']}%\n\n"
        f"Supply : {item['supply']}\n"
        f"Demand : {item['demand']}\n"
        f"Distance : 2.4%\n\n"
        f"⚠ Menunggu:\n"
        f"• Breakout di atas {item['price'] + 2}\n"
        f"ATAU\n"
        f"• Pullback ke {item['price'] - 6} - {item['price'] - 4}\n"
    )


def run_scan():
    print(f"\n=== S&D SCAN {datetime.now().strftime('%H:%M')} ===")
    stocks = read_stocks()
    if not stocks:
        print("Tidak ada stock list.")
        return

    mh = market_health_engine()
    buy_items, watch_items, sell_items, errors = [], [], [], []

    for i, stock in enumerate(stocks):
        try:
            item = scan_ticker(stock, mh["market_score"])
            if item is None:
                continue
            if item["signal"] in ["🟢 BUY", "🔥 STRONG BUY"]:
                buy_items.append(item)
            elif item["signal"] == "👀 WATCH":
                watch_items.append(item)
            elif item["signal"] == "🔴 SELL":
                sell_items.append(item)
        except Exception as e:
            errors.append(f"{stock}: {e}")
            log_error(f"scan {stock} | {e}")

        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(REQUEST_DELAY)

    buy_items = sorted(buy_items, key=lambda x: x["confidence"], reverse=True)
    watch_items = sorted(watch_items, key=lambda x: x["confidence"], reverse=True)
    sell_items = sorted(sell_items, key=lambda x: x["confidence"], reverse=True)

    header = (
        f"📡 S&D INTRADAY SCAN — {datetime.now().strftime('%d %b %H:%M')} WIB\n\n"
        f"🟢 BUY : {len(buy_items)} | 🔴 SELL : {len(sell_items)} | 👀 WATCH : {len(watch_items)}\n\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
    )
    send(header + market_health_text(mh))

    if buy_items:
        msg = "🟢 BUY SIGNALS\n\n"
        for item in buy_items[:5]:
            msg += build_buy_text(item) + "\n━━━━━━━━━━━━━━━━━━\n\n"
        send(msg)

    if watch_items:
        msg = ""
        for item in watch_items[:5]:
            msg += build_watch_text(item) + "\n━━━━━━━━━━━━━━━━━━\n\n"
        send(msg)

    if sell_items:
        msg = "🔴 SELL SIGNALS\n\n"
        for item in sell_items[:5]:
            msg += (
                f"🔴 {item['ticker']} @ {item['price']}\n\n"
                f"Signal : {item['signal']}\n"
                f"Confidence : {item['confidence']}/100\n\n"
                f"Smart Money\n{item['smart_money']}\n\n"
                f"Trend\n{item['trend']}\n\n"
                f"Volume\n{item['volume']}x Average\n\n"
                f"Aksi:\n❌ Jangan entry\n\n"
                f"━━━━━━━━━━━━━━━━━━\n\n"
            )
        send(msg)

    if errors:
        err_msg = "⚠️ Gagal di-load\n\n"
        for e in errors[:10]:
            err_msg += f"{e}\n"
        send(err_msg)


def main():
    print("=" * 50)
    print("NEUROBRO SCANNER - FINAL V2")
    print("=" * 50)

    schedule.clear()
    schedule.every().day.at("08:30").do(run_scan)
    schedule.every().day.at("09:15").do(run_scan)
    schedule.every().day.at("10:30").do(run_scan)

    run_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
