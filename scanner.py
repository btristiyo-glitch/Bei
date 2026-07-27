import os
import csv
import time
import schedule
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_IDS = [
    6262086905
]

CSV_FILE = "signals.csv"
ALERT_FILE = "alerts.csv"
STOCKS_FILE = "stocks.txt"
SECTORS_FILE = "sectors.csv"
ERROR_LOG_FILE = "error.log"

BATCH_SIZE = 3
REQUEST_DELAY = 3
ALERT_EXPIRY_HOURS = 3

VOLUME_THRESHOLD = 2.5
ZONE_DISTANCE_PCT = 2.5
LOOKBACK_CANDLES = 120
AD_WINDOW = 15
MIN_ZONE_VOLUME = 100_000_000
MIN_PRICE = 50


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

    chunks = [msg[i:i + 4000] for i in range(0, len(msg), 4000)]
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
        print(f"{STOCKS_FILE} tidak ditemukan.")
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


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


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


def find_intraday_zones(df):
    zones = []
    vol_ma = df["Volume"].rolling(20).mean()

    for i in range(3, min(LOOKBACK_CANDLES, len(df) - 2)):
        prev = df.iloc[i - 1]
        base = df.iloc[i]

        avg_vol = safe_float(vol_ma.iloc[i])
        if avg_vol is None or avg_vol == 0:
            continue

        base_vol = safe_float(base["Volume"])
        if base_vol is None:
            continue

        vol_ratio = base_vol / avg_vol
        if vol_ratio < VOLUME_THRESHOLD:
            continue

        range_prev = safe_float(prev["High"] - prev["Low"])
        if range_prev is None or range_prev == 0:
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
                "strength": "STRONG" if vol_ratio >= 4.0 else "MODERATE",
                "idx": i,
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
                "strength": "STRONG" if vol_ratio >= 4.0 else "MODERATE",
                "idx": i,
            })

    return zones


def detect_accumulation_distribution(df):
    window = AD_WINDOW

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
        chunk = series.iloc[-window:]
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
        label = "ACCUMULATING"
        if bull_div:
            label = "ACCUMULATING (Bull Div)"
    elif down_count >= 2 and avg_slope < 0:
        label = "DISTRIBUTING"
        if bear_div:
            label = "DISTRIBUTING (Bear Div)"
    else:
        label = "NEUTRAL"

    return {
        "status": label,
        "score": round(avg_slope, 4),
        "up": up_count,
        "down": down_count,
    }


def scan_snd_ticker(ticker_symbol):
    df = fetch_data(ticker_symbol, period="5d", interval="15m")
    if df is None or len(df) < 50:
        return None

    current_price = safe_float(df["Close"].iloc[-1])
    if current_price is None or current_price < MIN_PRICE:
        return None

    zones = find_intraday_zones(df)
    acc_dist = detect_accumulation_distribution(df)

    ticker_clean = ticker_symbol.replace(".JK", "")
    sector_map = read_sector_map()
    sector = sector_map.get(ticker_clean, "OTHER")

    result = {
        "ticker": ticker_clean,
        "price": round(current_price),
        "acc_dist": acc_dist["status"],
        "acc_score": acc_dist["score"],
        "zone_count": len(zones),
        "sector": sector,
        "signals": [],
        "watchlist": [],
    }

    for zone in zones:
        top = safe_float(zone["top"])
        bot = safe_float(zone["bot"])
        if top is None or bot is None:
            continue

        in_zone = bot <= current_price <= top
        distance_pct = 0

        if zone["type"] == "DEMAND":
            if current_price < bot:
                distance_pct = (bot - current_price) / bot * 100
            elif current_price > top:
                continue
        elif zone["type"] == "SUPPLY":
            if current_price > top:
                distance_pct = (current_price - top) / top * 100
            elif current_price < bot:
                continue

        if in_zone:
            is_buy_setup = zone["type"] == "DEMAND" and "ACCUMULATING" in acc_dist["status"]
            is_sell_setup = zone["type"] == "SUPPLY" and "DISTRIBUTING" in acc_dist["status"]

            if is_buy_setup:
                entry = current_price
                stop = bot * 0.995
                risk = entry - stop
                if risk <= 0:
                    continue

                tp1 = entry + (1.5 * risk)
                tp2 = entry + (2.5 * risk)
                tp3 = entry + (4.0 * risk)
                rr = round((tp1 - entry + (tp2 - entry) * 0.5) / (risk * 1.5), 1)

                result["signals"].append({
                    "action": "BUY",
                    "type": "S&D_BREAKOUT",
                    "zone_bot": round(bot),
                    "zone_top": round(top),
                    "entry": round(entry),
                    "stop": round(stop),
                    "tp1": round(tp1),
                    "tp2": round(tp2),
                    "tp3": round(tp3),
                    "vol_ratio": zone["vol_ratio"],
                    "strength": zone["strength"],
                    "rr": rr,
                    "acc_dist": acc_dist["status"],
                })

            elif is_sell_setup:
                entry = current_price
                stop = top * 1.005
                risk = stop - entry
                if risk <= 0:
                    continue

                tp1 = entry - (1.5 * risk)
                tp2 = entry - (2.5 * risk)
                tp3 = entry - (4.0 * risk)
                rr = round((entry - tp1 + (entry - tp2) * 0.5) / (risk * 1.5), 1)

                result["signals"].append({
                    "action": "SELL",
                    "type": "S&D_REJECTION",
                    "zone_bot": round(bot),
                    "zone_top": round(top),
                    "entry": round(entry),
                    "stop": round(stop),
                    "tp1": round(tp1),
                    "tp2": round(tp2),
                    "tp3": round(tp3),
                    "vol_ratio": zone["vol_ratio"],
                    "strength": zone["strength"],
                    "rr": rr,
                    "acc_dist": acc_dist["status"],
                })

        elif 0 < distance_pct <= ZONE_DISTANCE_PCT:
            direction = "mendekati demand" if zone["type"] == "DEMAND" else "mendekati supply"
            result["watchlist"].append({
                "setup": f"{zone['type']} zone {direction} ({distance_pct:.1f}%)",
                "zone": f"{round(bot):,} - {round(top):,}",
                "vol_ratio": f"{zone['vol_ratio']}x",
                "strength": zone["strength"],
            })

    return result


def init_csv_files():
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "time", "ticker", "score", "signal",
            "rsi", "price", "rvol", "gap_pct",
            "atr_pct", "entry_limit", "stop_loss",
            "tp1", "tp2", "tp3"
        ])

    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticker", "entry", "stop_loss", "tp1", "tp2", "tp3",
            "score", "signal", "scan_time"
        ])


def append_signals_csv(rows):
    now = datetime.now()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for item in rows:
            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M"),
                item["ticker"],
                item["score"],
                item["signal"],
                item.get("rsi", 0),
                item.get("price", 0),
                item.get("rvol", 0),
                item.get("gap_pct", 0),
                item.get("atr_pct", 0),
                item.get("entry_limit", 0),
                item.get("stop_loss", 0),
                item.get("tp1", 0),
                item.get("tp2", 0),
                item.get("tp3", 0),
            ])


def run_snd_scan():
    print(f"\n=== S&D SCAN {datetime.now().strftime('%H:%M')} ===")
    init_csv_files()

    stocks = read_stocks()
    if not stocks:
        print("Tidak ada stock list.")
        return

    buy_signals = []
    sell_signals = []
    watchlist = []
    errors = []

    for i, stock in enumerate(stocks):
        print(f"Scanning {i + 1}/{len(stocks)}: {stock}")

        try:
            result = scan_snd_ticker(stock)
            if result is None:
                continue

            if result["signals"]:
                for s in result["signals"]:
                    base_score = 70 if s["strength"] == "STRONG" else 50
                    if "Bull Div" in s["acc_dist"] or "Bear Div" in s["acc_dist"]:
                        base_score += 20
                    s["score"] = base_score
                    s["sector"] = result["sector"]
                    s["price"] = result["price"]
                    s["signal"] = s["action"]
                    if s["action"] == "BUY":
                        buy_signals.append(s)
                    else:
                        sell_signals.append(s)

            elif result["watchlist"]:
                watchlist.append(result)

        except Exception as e:
            errors.append(f"{stock}: {e}")
            log_error(f"scan_snd {stock} | {e}")
            continue

        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(REQUEST_DELAY)

    buy_signals = sorted(buy_signals, key=lambda x: x["score"], reverse=True)
    sell_signals = sorted(sell_signals, key=lambda x: x["score"], reverse=True)

    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ticker", "entry", "stop_loss", "tp1", "tp2", "tp3",
            "score", "signal", "scan_time"
        ])
        for s in buy_signals + sell_signals:
            w.writerow([
                s["ticker"], s["entry"], s["stop"], s["tp1"], s["tp2"], s["tp3"],
                s["score"], s["signal"], datetime.now().strftime("%Y-%m-%d %H:%M")
            ])

    summary = f"📡 **S&D INTRADAY SCAN** — {datetime.now().strftime('%d %b %H:%M')} WIB\n"
    summary += f"_Buy: {len(buy_signals)} | Sell: {len(sell_signals)} | Watch: {len(watchlist)} | Error: {len(errors)}_\n"
    send(summary)

    if buy_signals:
        msg = "🟢 **S&D DEMAND ZONE — BUY SIGNALS**\n_Harga di demand zone + accumulation terkonfirmasi_\n\n"
        for s in buy_signals[:8]:
            tv = f"https://www.tradingview.com/chart/?symbol=IDX:{s['ticker']}"
            msg += (
                f"🟢 **#{s['ticker']}** | Score: {s['score']} | {s['strength']}\n"
                f"┃ Entry: Rp {s['entry']:,}\n"
                f"┃ Stop:  Rp {s['stop']:,}\n"
                f"┃ TP1: Rp {s['tp1']:,}  TP2: Rp {s['tp2']:,}\n"
                f"┃ TP3: Rp {s['tp3']:,}  R:R 1:{s['rr']}\n"
                f"Zone: Rp {s['zone_bot']:,} - Rp {s['zone_top']:,}\n"
                f"Vol: {s['vol_ratio']}x | Acc/Dist: {s['acc_dist']}\n"
                f"[TradingView]({tv})\n\n"
            )
            if len(msg) > 3800:
                break
        send(msg)

    if sell_signals:
        msg = "🔴 **S&D SUPPLY ZONE — SELL SIGNALS**\n_Harga di supply zone + distribution terkonfirmasi_\n\n"
        for s in sell_signals[:6]:
            tv = f"https://www.tradingview.com/chart/?symbol=IDX:{s['ticker']}"
            msg += (
                f"🔴 **#{s['ticker']}** | Score: {s['score']} | {s['strength']}\n"
                f"┃ Entry: Rp {s['entry']:,}\n"
                f"┃ Stop:  Rp {s['stop']:,}\n"
                f"┃ TP1: Rp {s['tp1']:,}  TP2: Rp {s['tp2']:,}\n"
                f"┃ TP3: Rp {s['tp3']:,}  R:R 1:{s['rr']}\n"
                f"Zone: Rp {s['zone_bot']:,} - Rp {s['zone_top']:,}\n"
                f"Vol: {s['vol_ratio']}x | Acc/Dist: {s['acc_dist']}\n"
                f"[TradingView]({tv})\n\n"
            )
            if len(msg) > 3800:
                break
        send(msg)

    if watchlist:
        msg = "📊 **S&D WATCHLIST — Mendekati Zone**\n_Harga dalam 2.5% dari S&D zone_\n\n"
        for w in sorted(watchlist, key=lambda x: x["acc_score"], reverse=True)[:8]:
            acc_icon = "📈" if "ACCUMULATING" in w["acc_dist"] else "📉" if "DISTRIBUTING" in w["acc_dist"] else "➖"
            msg += f"{acc_icon} **#{w['ticker']}** @ Rp {w['price']:,} | {w['acc_dist']}\n"
            for ws in w["watchlist"][:2]:
                msg += f"   ┃ {ws['setup']} | Zone: {ws['zone']} | Vol: {ws['vol_ratio']}\n"
            msg += "\n"
            if len(msg) > 3800:
                break
        send(msg)

    if errors:
        err_msg = "⚠️ **Gagal di-load**\n\n"
        for e in errors[:10]:
            err_msg += f"  {e}\n"
        if len(errors) > 10:
            err_msg += f"  ...dan {len(errors) - 10} lainnya"
        send(err_msg)

    print(f"\n✅ S&D SCAN SELESAI — {datetime.now().strftime('%H:%M')}")
    print(f"  Buy: {len(buy_signals)} | Sell: {len(sell_signals)} | Watch: {len(watchlist)}")


def check_alerts():
    print(f"  ⌛ Alert check {datetime.now().strftime('%H:%M')}...")

    if not os.path.exists(ALERT_FILE):
        return

    with open(ALERT_FILE, "r", encoding="utf-8") as f:
        setups = list(csv.DictReader(f))

    if not setups:
        return

    now = datetime.now()
    active_setups = []

    for setup in setups:
        ticker = setup.get("ticker", "") + ".JK"
        entry = safe_float(setup.get("entry"))
        stop = safe_float(setup.get("stop_loss"))
        scan_time_str = setup.get("scan_time", "")
        ticker_clean = setup.get("ticker", "")

        if entry is None:
            continue

        if scan_time_str:
            try:
                scan_dt = datetime.strptime(scan_time_str, "%Y-%m-%d %H:%M")
                if (now - scan_dt) > timedelta(hours=ALERT_EXPIRY_HOURS):
                    print(f"  ⏰ {ticker_clean} expired")
                    continue
            except Exception:
                pass

        try:
            df = yf.download(
                ticker,
                period="1d",
                interval="15m",
                auto_adjust=True,
                progress=False,
                timeout=10
            )
            df = clean_columns(df)
            if df is None or df.empty:
                active_setups.append(setup)
                continue

            current = safe_float(df["Close"].iloc[-1])
            low_today = safe_float(df["Low"].min())

            if current is None or low_today is None:
                active_setups.append(setup)
                continue

            if low_today <= entry <= current:
                active_setups.append(setup)
                continue

            risk = (stop and abs(entry - stop)) or entry * 0.02
            if stop and entry:
                if current > entry + risk * 3 or current < entry - risk * 3:
                    print(f"  ⏰ {ticker_clean} harga sudah jauh ({current:,.0f}), expire")
                    continue

            active_setups.append(setup)

        except Exception as e:
            log_error(f"check_alerts {ticker} | {e}")
            active_setups.append(setup)
            continue

    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ticker", "entry", "stop_loss", "tp1", "tp2", "tp3",
            "score", "signal", "scan_time"
        ])
        for s in active_setups:
            w.writerow([
                s.get("ticker", ""),
                s.get("entry", ""),
                s.get("stop_loss", ""),
                s.get("tp1", ""),
                s.get("tp2", ""),
                s.get("tp3", ""),
                s.get("score", ""),
                s.get("signal", ""),
                s.get("scan_time", "")
            ])

    print(f"  ✅ Alert check selesai - {len(active_setups)} aktif, {len(setups) - len(active_setups)} expired")


def schedule_jobs():
    schedule.clear()
    schedule.every().day.at("08:30").do(run_snd_scan)
    schedule.every().day.at("09:15").do(run_snd_scan)
    schedule.every().day.at("10:30").do(run_snd_scan)

    for hour in range(9, 17):
        for minute in [0, 15, 30, 45]:
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(check_alerts)


def main():
    print("=" * 50)
    print("NEUROBRO SCANNER - S&D ZONE INTRADAY")
    print(f"Mulai: {datetime.now().strftime('%H:%M')} WIB")
    print("Jadwal:")
    print("  - S&D scan: 08:30, 09:15, 10:30")
    print("  - Alert check: tiap 15 menit (09:00-16:00)")
    print("=" * 50)

    schedule_jobs()
    run_snd_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
