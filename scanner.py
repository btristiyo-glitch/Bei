import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# =========================================================
# CONFIG
# =========================================================
TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_IDS_RAW = os.getenv("TELEGRAM_CHAT_ID", "").strip()
CHAT_IDS = [x.strip() for x in CHAT_IDS_RAW.split(",") if x.strip()]

STOCKS_FILE = "stocks.txt"
ERROR_LOG_FILE = "error.log"
OUTPUT_CSV = "scan_results.csv"
TOP10_CSV = "top10_results.csv"
GAINER_CSV = "gainer_candidates.csv"

LOOKBACK = 180
MIN_PRICE = 50
VOLUME_MULTIPLIER = 1.8
REQUEST_DELAY = 1.5
ATR_PERIOD = 14
REGIME_LOOKBACK = 20
PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# Threshold khusus kandidat top gainer intraday
GAINER_CONFIG = {
    "atr_min": 3.0,        # minimal ATR % biar bisa gerak besar
    "gap_min": 1.0,        # minimal gap open %
    "rvol_min": 2.0,       # harga harus dikejar volume
    "gain_min": 1.5,       # minimal gain intraday berjalan
    "flow_min": 0,         # skema (opsional)
    "top_n": 10,           # jumlah kandidat
    "min_price": MIN_PRICE,
}

# =========================================================
# UTIL
# =========================================================
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

def fetch_daily(symbol, period="9mo"):
    try:
        df = yf.download(symbol, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        df = clean_df(df)
        if df is None or len(df) < 80:
            return None
        return df
    except Exception as e:
        log_error(f"fetch_daily {symbol} | {e}")
        return None

def read_stocks():
    if not os.path.exists(STOCKS_FILE):
        return []
    with open(STOCKS_FILE, "r", encoding="utf-8") as f:
        stocks = [x.strip() for x in f if x.strip()]
    return [s if s.endswith(".JK") else s + ".JK" for s in stocks]

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def atr(df, period=14):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rvol(df, length=20):
    vol = df["Volume"].astype(float)
    ma = vol.rolling(length).mean()
    if pd.isna(ma.iloc[-1]) or ma.iloc[-1] == 0:
        return 0.0
    return float(vol.iloc[-1] / ma.iloc[-1])

# =========================================================
# LAPISAN 1: KANDIDAT TOP GAINER INTRADAY (BARU - PRIORITAS)
# =========================================================
def fetch_intraday(symbol):
    """
    Ambil batang 15-menit hari ini + close kemarin.
    Return None kalau data intraday nggak valid/kosong.
    """
    try:
        df = yf.download(symbol, period="3d", interval="15m",
                         auto_adjust=True, progress=False)
        df = clean_df(df)
        if df is None or len(df) < 10:
            return None

        last_date = df.index.date[-1]
        todays = df[df.index.date == last_date]
        prev_rows = df[df.index.date < last_date]
        if todays.empty or prev_rows.empty:
            return None

        prev_close = float(prev_rows["Close"].iloc[-1])
        now = float(todays["Close"].iloc[-1])
        open_p = float(todays["Open"].iloc[0])
        high = float(todays["High"].max())
        low = float(todays["Low"].min())

        # RVOL intraday: volume batang terakhir vs rata2 20 batang
        vol_ref = float(df["Volume"].rolling(20).mean().iloc[-1])
        rv = float(todays["Volume"].iloc[-1] / vol_ref) if vol_ref > 0 else 0

        return {
            "price": now,
            "open": open_p,
            "high": high,
            "low": low,
            "prev_close": prev_close,
            "gain_pct": (now / prev_close - 1) * 100,
            "gap_pct": (open_p / prev_close - 1) * 100,
            "rv": round(rv, 2),
            "bars": len(todays),
        }
    except Exception as e:
        log_error(f"fetch_intraday {symbol} | {e}")
        return None

def is_gainer_candidate(snap):
    if snap is None:
        return False
    if snap["price"] < GAINER_CONFIG["min_price"]:
        return False
    if snap["rv"] < GAINER_CONFIG["rvol_min"]:
        return False
    gap_ok = snap["gap_pct"] >= GAINER_CONFIG["gap_min"] or \
             snap["gain_pct"] >= GAINER_CONFIG["gain_min"]
    return gap_ok and snap["gain_pct"] > 0

def gainer_score(snap):
    """Skor 0-100 - makin tinggi makin berpotensi jadi top gainer."""
    score = 0
    if snap["gap_pct"] >= 3:   score += 30
    elif snap["gap_pct"] >= 1: score += 15
    if snap["rv"] >= 3:        score += 25
    elif snap["rv"] >= 2:      score += 15
    if snap["gain_pct"] >= 4:  score += 25
    elif snap["gain_pct"] >= 2: score += 15
    if snap["price"] > snap["open"]: score += 20   # momentum naik dari open
    return int(min(100, score))

def percent_from_range(snap):
    """Seberapa dekat harga sekarang ke range high sesi (% buat target)."""
    rng = snap["high"] - snap["low"]
    if rng <= 0:
        return 100.0
    return (snap["high"] - snap["price"]) / rng * 100

def scan_intraday_gainers(symbols):
    candidates = []
    for i, s in enumerate(symbols):
        try:
            snap = fetch_intraday(s)
            if is_gainer_candidate(snap):
                snap["ticker"] = s.replace(".JK", "")
                snap["score"] = gainer_score(snap)
                snap["pct_from_high"] = round(percent_from_range(snap), 1)
                candidates.append(snap)
            if (i + 1) % 3 == 0:
                time.sleep(REQUEST_DELAY)
        except Exception as e:
            log_error(f"intraday {s} | {e}")
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:GAINER_CONFIG["top_n"]]

def format_intraday_gainers(candidates):
    if not candidates:
        return "🚫 Tidak ada kandidat top gainer intraday yang memenuhi syarat."
    lines = ["🚀 *KANDIDAT TOP GAINER INTRADAY*", "```"]
    for i, s in enumerate(candidates, 1):
        lines.append(
            f"{i}. {s['ticker']}  Score:{s['score']}\n"
            f"   Price: {s['price']:,.0f}  Gain: {s['gain_pct']:.2f}%\n"
            f"   Gap: {s['gap_pct']:.2f}%  RVOL: {s['rv']:.2f}x\n"
            f"   Dist ke High: {s['pct_from_high']:.1f}%"
        )
    lines.append("```")
    return "\n".join(lines)

# =========================================================
# LAPISAN 2: SWING HARIAN (DARi KODE AWAL KAMU)
# =========================================================
def get_trend(df):
    close = df["Close"].astype(float)
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1]
    last = close.iloc[-1]
    if last > e20 > e50 > e200:
        trend = "🟢 Strong Bullish"
    elif last > e20 > e50:
        trend = "🟢 Bullish"
    elif last < e20 < e50 < e200:
        trend = "🔴 Strong Bearish"
    elif last < e20 < e50:
        trend = "🔴 Bearish"
    else:
        trend = "⚪ Neutral"
    return trend, e20, e50, e200

# (Fungsi harian lainnya: market_regime_filter, find_pivots, detect_structure,
#  detect_liquidity_sweep, candle_patterns, find_supply_demand, latest_zones,
#  score_to_rank, confidence_score, calculate_signal, build_plan, format_signal,
#  scan_one — DIPERTAHANKAN IDENTIK dengan kode asli kamu.
#  Potong-penuh dari pastebin; di sini saya ringkas biar file jelas.)

def market_regime_filter(df):
    close = df["Close"].astype(float)
    e20 = ema(close, 20); e50 = ema(close, 50); e200 = ema(close, 200)
    last = close.iloc[-1]
    vol_ma20 = df["Volume"].astype(float).rolling(20).mean().iloc[-1]
    price_above_200 = last > e200.iloc[-1]
    trend_up = e20.iloc[-1] > e50.iloc[-1] > e200.iloc[-1]
    trend_down = e20.iloc[-1] < e50.iloc[-1] < e200.iloc[-1]
    if price_above_200 and trend_up:
        regime, bias, weight = "risk-on", "bullish", 1.20
    elif (not price_above_200) and trend_down:
        regime, bias, weight = "risk-off", "bearish", 0.85
    else:
        regime, bias, weight = "mixed", "neutral", 1.00
    if vol_ma20 and df["Volume"].iloc[-1] < vol_ma20 * 0.7:
        weight *= 0.90
    return {"regime": regime, "bias": bias, "weight": weight,
            "e20": float(e20.iloc[-1]), "e50": float(e50.iloc[-1]),
            "e200": float(e200.iloc[-1])}

def find_pivots(df, left=3, right=3):
    highs = df["High"].astype(float).values
    lows = df["Low"].astype(float).values
    pivot_highs, pivot_lows = [], []
    for i in range(left, len(df) - right):
        whl = highs[i - left:i]; whr = highs[i + 1:i + right + 1]
        wll = lows[i - left:i]; wlr = lows[i + 1:i + right + 1]
        if highs[i] > whl.max() and highs[i] > whr.max():
            pivot_highs.append((i, float(highs[i])))
        if lows[i] < wll.min() and lows[i] < wlr.min():
            pivot_lows.append((i, float(lows[i])))
    return pivot_highs, pivot_lows

def detect_structure(df):
    if len(df) < 30:
        return {"bias": "neutral", "bos": False, "choch": False,
                "swing_high": None, "swing_low": None}
    pivot_highs, pivot_lows = find_pivots(df, PIVOT_LEFT, PIVOT_RIGHT)
    if not pivot_highs or not pivot_lows:
        return {"bias": "neutral", "bos": False, "choch": False,
                "swing_high": None, "swing_low": None}
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    last_swing_high = pivot_highs[-1][1]
    last_swing_low = pivot_lows[-1][1]
    bos, choch, bias = False, False, "neutral"
    if last_close > last_swing_high:
        bos, bias = True, "bullish"
    elif last_close < last_swing_low:
        bos, bias = True, "bearish"
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        prev_swing_high = pivot_highs[-2][1]
        prev_swing_low = pivot_lows[-2][1]
        if prev_close <= prev_swing_high < last_close:
            choch, bias = True, "bullish"
        elif prev_close >= prev_swing_low > last_close:
            choch, bias = True, "bearish"
    return {"bias": bias, "bos": bos, "choch": choch,
            "swing_high": last_swing_high, "swing_low": last_swing_low}

def detect_liquidity_sweep(df, structure):
    if len(df) < 20 or structure["swing_high"] is None or structure["swing_low"] is None:
        return {"bull_sweep": False, "bear_sweep": False, "score": 0}
    last = df.iloc[-1]; prev = df.iloc[-2]
    sh = structure["swing_high"]; sl = structure["swing_low"]
    bull = (last["High"] > sh and last["Close"] < sh and prev["High"] <= sh)
    bear = (last["Low"] < sl and last["Close"] > sl and prev["Low"] >= sl)
    return {"bull_sweep": bull, "bear_sweep": bear, "score": 1 if (bull or bear) else 0}

def candle_patterns(df):
    if len(df) < 3:
        return []
    last = df.iloc[-1]; prev = df.iloc[-2]
    patterns = []
    body = abs(last["Close"] - last["Open"])
    rng = max(last["High"] - last["Low"], 1e-9)
    lower_wick = min(last["Close"], last["Open"]) - last["Low"]
    upper_wick = last["High"] - max(last["Close"], last["Open"])
    if last["Close"] > last["Open"] and prev["Close"] < prev["Open"] and \
       last["Close"] > prev["Open"] and last["Open"] < prev["Close"]:
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
    vol_ma = df["Volume"].astype(float).rolling(20).mean()
    for i in range(3, min(LOOKBACK, len(df) - 2)):
        prev = df.iloc[i - 1]; base = df.iloc[i]
        if pd.isna(vol_ma.iloc[i]) or vol_ma.iloc[i] == 0:
            continue
        vol_ratio = base["Volume"] / vol_ma.iloc[i]
        if vol_ratio < VOLUME_MULTIPLIER:
            continue
        strength = "Strong" if vol_ratio >= 4 else "Moderate"
        if prev["Close"] < prev["Open"] and base["Close"] > base["Open"]:
            zones.append({"type": "DEMAND",
                          "top": max(base["Close"], prev["Open"]),
                          "bot": min(base["Open"], prev["Close"]),
                          "strength": strength, "vol_ratio": round(float(vol_ratio), 1)})
        if prev["Close"] > prev["Open"] and base["Close"] < base["Open"]:
            zones.append({"type": "SUPPLY",
                          "top": max(prev["Close"], base["Open"]),
                          "bot": min(base["Close"], prev["Open"]),
                          "strength": strength, "vol_ratio": round(float(vol_ratio), 1)})
    return zones

def latest_zones(zones):
    demand = None; supply = None
    for z in reversed(zones):
        if z["type"] == "DEMAND" and demand is None:
            demand = z
        if z["type"] == "SUPPLY" and supply is None:
            supply = z
        if demand and supply:
            break
    return supply, demand

def score_to_rank(score, signal, volume, regime_weight, sweep_bonus, bos_bonus, choch_bonus):
    base = score
    if signal in ["🟢 BUY", "🔴 SELL"]:
        base += 15
    if volume >= 2: base += 10
    elif volume >= 1.5: base += 5
    base += sweep_bonus + bos_bonus + choch_bonus
    base *= regime_weight
    return int(min(100, round(base)))

def confidence_score(trend, rvol_value, patterns, in_zone, near_zone, structure, regime, sweep):
    score = 0
    if "Strong Bullish" in trend or "Strong Bearish" in trend: score += 22
    elif "Bullish" in trend or "Bearish" in trend: score += 14
    if regime["regime"] == "risk-on": score += 12
    elif regime["regime"] == "risk-off": score += 5
    else: score += 8
    if rvol_value >= 3: score += 18
    elif rvol_value >= 2: score += 14
    elif rvol_value >= 1.5: score += 9
    if any(p in patterns for p in ["Breakout Candle", "Bullish Engulfing", "Hammer", "Bearish Rejection"]):
        score += 12
    if in_zone: score += 10
    elif near_zone: score += 5
    if structure["bos"]: score += 12
    if structure["choch"]: score += 10
    if structure["bias"] in ["bullish", "bearish"]: score += 4
    if sweep["bull_sweep"] or sweep["bear_sweep"]: score += 8
    return min(100, score)

def calculate_signal(df):
    close = float(df["Close"].iloc[-1])
    trend, e20, e50, e200 = get_trend(df)
    volume_ratio = rvol(df)
    patterns = candle_patterns(df)
    supply, demand = latest_zones(find_supply_demand(df))
    structure = detect_structure(df)
    regime = market_regime_filter(df)
    sweep = detect_liquidity_sweep(df, structure)
    atr_val = atr(df, ATR_PERIOD).iloc[-1]
    atr_val = float(atr_val) if not pd.isna(atr_val) and atr_val > 0 else max(close * 0.02, 1)

    in_demand = demand and demand["bot"] <= close <= demand["top"]
    near_demand = demand and 0 < ((demand["bot"] - close) / demand["bot"]) * 100 <= 2
    in_supply = supply and supply["bot"] <= close <= supply["top"]
    near_supply = supply and 0 < ((close - supply["top"]) / supply["top"]) * 100 <= 2

    bullish = trend in ["🟢 Strong Bullish", "🟢 Bullish"] or structure["bias"] == "bullish"
    bearish = trend in ["🔴 Strong Bearish", "🔴 Bearish"] or structure["bias"] == "bearish"
    breakout = "Breakout Candle" in patterns or structure["bos"]
    reversal_bull = "Bullish Engulfing" in patterns or "Hammer" in patterns or structure["choch"] or sweep["bull_sweep"]
    reversal_bear = "Bearish Rejection" in patterns or structure["choch"] or sweep["bear_sweep"]

    if (in_demand or near_demand) and bullish and volume_ratio >= 1.8 and \
       (breakout or reversal_bull) and regime["bias"] != "bearish":
        signal = "🟢 BUY"
    elif (in_supply or near_supply) and bearish and volume_ratio >= 1.8 and \
         (breakout or reversal_bear) and regime["bias"] != "bullish":
        signal = "🔴 SELL"
    else:
        signal = "WATCH"

    score = confidence_score(trend, volume_ratio, patterns, in_demand or in_supply,
                             near_demand or near_supply, structure, regime, sweep)
    return {"signal": signal, "confidence": score, "trend": trend,
            "volume": round(float(volume_ratio), 1), "patterns": patterns,
            "supply": supply, "demand": demand, "close": close,
            "e20": float(e20), "e50": float(e50), "e200": float(e200),
            "atr": atr_val, "structure": structure, "regime": regime, "sweep": sweep}

def build_plan(result):
    close = result["close"]; atr_val = result["atr"]
    if result["signal"] == "🔴 SELL":
        entry_low = result["supply"]["bot"] if result["supply"] else close
        entry_high = result["supply"]["top"] if result["supply"] else close + atr_val
        stop_loss = round(entry_high + atr_val * 0.5)
        tp1 = round(close - atr_val * 1.0); tp2 = round(close - atr_val * 2.0); tp3 = round(close - atr_val * 3.0)
        rr = round((close - tp3) / max(1, stop_loss - close), 1)
        return {"entry": f"{round(entry_low)} - {round(entry_high)}", "sl": stop_loss,
                "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr": rr}
    if result["signal"] == "🟢 BUY":
        entry_low = result["demand"]["bot"] if result["demand"] else close - atr_val
        entry_high = result["demand"]["top"] if result["demand"] else close
        stop_loss = round(entry_low - atr_val * 0.5)
        tp1 = round(close + atr_val * 1.0); tp2 = round(close + atr_val * 2.0); tp3 = round(close + atr_val * 3.0)
        rr = round((tp3 - close) / max(1, close - stop_loss), 1)
        return {"entry": f"{round(entry_low)} - {round(entry_high)}", "sl": stop_loss,
                "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr": rr}
    return None

def format_signal(ticker, result, plan):
    regime = result["regime"]["regime"]; bias = result["regime"]["bias"]
    sweep = result["sweep"]
    common = (
        f"📊 {ticker}\n\n"
        f"Signal : {result['signal']}\n"
        f"Confidence : {result['confidence']}%\n"
        f"Rank Score : {result['rank_score']}\n"
        f"Regime : {regime} | Bias : {bias}\n"
        f"Trend : {result['trend']}\n"
        f"EMA : 20={round(result['e20'])} | 50={round(result['e50'])} | 200={round(result['e200'])}\n"
        f"Structure : BOS {result['structure']['bos']} | CHOCH {result['structure']['choch']} | Bias {result['structure']['bias']}\n"
        f"Liquidity Sweep : Bull {sweep['bull_sweep']} | Bear {sweep['bear_sweep']}\n"
        f"Volume : {result['volume']}x\n"
        f"Supply : {round(result['supply']['bot'])}-{round(result['supply']['top']) if result['supply'] else '-'}\n"
        f"Demand : {round(result['demand']['bot'])}-{round(result['demand']['top']) if result['demand'] else '-'}\n"
    )
    if result["signal"] == "🟢 BUY":
        return (common + f"Entry : {plan['entry']}\nSL : {plan['sl']}\n"
                f"TP1 : {plan['tp1']}\nTP2 : {plan['tp2']}\nTP3 : {plan['tp3']}\n"
                f"R:R : 1:{plan['rr']}\nAksi : ✅ Boleh mulai cicil beli\n")
    if result["signal"] == "🔴 SELL":
        return (common + f"Entry : {plan['entry']}\nSL : {plan['sl']}\n"
                f"TP1 : {plan['tp1']}\nTP2 : {plan['tp2']}\nTP3 : {plan['tp3']}\n"
                f"R:R : 1:{plan['rr']}\nAksi : ❌ Jangan entry\n")
    return (common + "Aksi : ❌ Tunggu level valid\n")

def get_daily_signals(symbols):
    rows, text = [], []
    for s in symbols:
        try:
            df = fetch_daily(s)
            if df is None or float(df["Close"].iloc[-1]) < MIN_PRICE:
                continue
            result = calculate_signal(df)
            plan = build_plan(result)
            result["rank_score"] = score_to_rank(
                result["confidence"], result["signal"], result["volume"],
                result["regime"]["weight"],
                8 if result["sweep"]["score"] else 0,
                10 if result["structure"]["bos"] else 0,
                8 if result["structure"]["choch"] else 0)
            ticker = s.replace(".JK", "")
            rows.append({"ticker": ticker, "signal": result["signal"],
                         "confidence": result["confidence"], "rank_score": result["rank_score"],
                         "regime": result["regime"]["regime"], "bias": result["regime"]["bias"],
                         "trend": result["trend"], "volume": result["volume"],
                         "bos": result["structure"]["bos"], "choch": result["structure"]["choch"],
                         "bull_sweep": result["sweep"]["bull_sweep"],
                         "bear_sweep": result["sweep"]["bear_sweep"]})
            text.append(format_signal(ticker, result, plan))
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log_error(f"daily {s} | {e}")
    return rows, text

# =========================================================
# MAIN
# =========================================================
def main():
    stocks = read_stocks()
    if not stocks:
        send_telegram("stocks.txt kosong.")
        return

    # [1] PRIORITAS: top gainer intraday
    gainer_cands = scan_intraday_gainers(stocks)
    if gainer_cands:
        g_text = format_intraday_gainers(gainer_cands)
        send_telegram(g_text)
        pd.DataFrame(gainer_cands).to_csv(GAINER_CSV, index=False)
    else:
        send_telegram("🚫 Tidak ada kandidat top gainer intraday saat ini.")

    # [2] Lapisan kedua: swing harian (dari kode asli kamu)
    daily_rows, daily_text = get_daily_signals(stocks)
    if not daily_rows:
        send_telegram("Tidak ada sinyal harian valid hari ini.")
        return

    df_all = pd.DataFrame(daily_rows).sort_values(
        ["rank_score", "confidence", "volume"], ascending=False)
    df_top10 = df_all.head(10).copy()
    df_all.to_csv(OUTPUT_CSV, index=False)
    df_top10.to_csv(TOP10_CSV, index=False)

    now = datetime.now().strftime("%d %b %H:%M")
    summary = (f"📡 DAILY STOCK RANKING - {now} WIB\n\nTop 10 saham terbaik hari ini:\n\n")
    for i, row in enumerate(df_top10.itertuples(index=False), start=1):
        summary += (f"{i}. {row.ticker} - {row.signal} - "
                    f"Score {row.rank_score} - Conf {row.confidence}% - Vol {row.volume}x\n")
    send_telegram(summary)

    detailed_text = "\n━━━━━━━━━━━━━━━━━━\n\n".join(daily_text)
    if detailed_text:
        send_telegram(detailed_text)

if __name__ == "__main__":
    main()
