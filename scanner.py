"""
NEUROBRO SCALPING SCANNER - IDX
Target: 3% profit per market open session
Strategy: Gap 1-4% + Volume spike 2x+ di 5 menit pertama market buka
Exit: TP 3% atau cut loss di stop (max 2% risk)
Jadwal: Scan 09:05 & 09:10 WIB, exit check tiap 5 menit sampai 11:00 WIB
"""

import os
import csv
import time
import json
import schedule
import requests
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta

# ============================================================
# KONFIGURASI
# ============================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_IDS = [
    6262086905,
    6003751935,
]

SCALP_CSV = "scalp_setups.csv"
SCALP_LOG = "scalp_log.csv"
STOCKS_FILE = "stocks.txt"
ERROR_LOG = "scalp_error.log"
SCALP_HISTORY = "scalp_history.json"

# Filter scalping
MIN_GAP = 1.0        # gap minimal 1%
MAX_GAP = 4.0        # gap maksimal 4%
MIN_VOL_RATIO = 2.0  # volume spike minimal 2x rata-rata
MAX_RISK_PCT = 2.0   # stop loss max 2% dari entry
TARGET_PCT = 3.0     # target profit 3%
MAX_HOLD_MINUTES = 120  # maksimal hold 2 jam

# Jadwal
SCAN_TIMES = ["09:05", "09:10"]
EXIT_CHECK_START = "09:05"
EXIT_CHECK_END = "11:00"

BATCH_DELAY = 3    # jeda antar request
BATCH_SIZE = 3

# ============================================================
# UTILITY
# ============================================================
def log_error(text):
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%H:%M')} | {text}\n")
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
                        log_error(f"Telegram {chat_id} | {e}")
                    time.sleep(2)

def safe_float(val, default=None):
    try:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default

def clean_columns(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

def read_stocks():
    if not os.path.exists(STOCKS_FILE):
        print(f"{STOCKS_FILE} tidak ditemukan.")
        return []

    with open(STOCKS_FILE, "r", encoding="utf-8") as f:
        stocks = [x.strip() for x in f if x.strip()]

    return [s if s.endswith(".JK") else s + ".JK" for s in stocks]

def load_history():
    """Load history harian buat track win/loss rate"""
    if not os.path.exists(SCALP_HISTORY):
        return {"total_trades": 0, "wins": 0, "losses": 0, "daily_pnl": []}
    
    try:
        with open(SCALP_HISTORY, "r") as f:
            return json.load(f)
    except Exception:
        return {"total_trades": 0, "wins": 0, "losses": 0, "daily_pnl": []}

def save_history(history):
    with open(SCALP_HISTORY, "w") as f:
        json.dump(history, f)

# ============================================================
# CORE SCALPING LOGIC
# ============================================================
def fetch_scalp_data(symbol):
    """
    Ambil data 5 menit untuk 3 hari terakhir
    Fokus: candlestick 30 menit pertama market buka (09:00-09:30)
    """
    try:
        df = yf.download(
            symbol,
            period="3d",
            interval="5m",
            auto_adjust=True,
            progress=False,
            timeout=15
        )
        df = clean_columns(df)
        if df is None or df.empty or len(df) < 15:
            return None
        
        return df
    except Exception as e:
        log_error(f"fetch_scalp {symbol} | {e}")
        return None

def analyze_scalp_setup(stock):
    """
    Analisa setup scalping:
    1. Ambil 3 hari data 5 menit
    2. Cari candle pertama market buka (09:00)
    3. Hitung gap dari close kemarin
    4. Cek volume spike
    5. Entry = close candle pertama
    6. Stop = low candle pertama - 0.5% buffer
    7. TP = entry * 1.03 (3%)
    """
    df = fetch_scalp_data(stock)
    if df is None or len(df) < 15:
        return None
    
    try:
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        
        # Filter candle market open (09:00-09:30 WIB = 02:00-02:30 UTC)
        # IDX di yfinance pake UTC, 09:00 WIB = 02:00 UTC
        market_open = df.between_time("01:55", "02:35")
        
        if market_open.empty:
            # Fallback: ambil candle pertama hari ini
            today = datetime.now().strftime("%Y-%m-%d")
            today_mask = df.index.strftime("%Y-%m-%d") == today
            if today_mask.any():
                today_df = df[today_mask]
                if len(today_df) >= 2:
                    market_open = today_df.iloc[:2].copy()
                else:
                    return None
            else:
                return None
        
        if len(market_open) < 1:
            return None
        
        # Candle pertama market buka
        first_candle = market_open.iloc[0]
        
        # Close kemarin (hari sebelumnya)
        # Cari candle terakhir sebelum market open hari ini
        before_today = df[df.index < market_open.index[0]]
        if before_today.empty:
            return None
        
        prev_close = safe_float(before_today["Close"].iloc[-1])
        if prev_close is None or prev_close == 0:
            return None
        
        # Hitung gap
        first_open = safe_float(first_candle["Open"])
        if first_open is None:
            return None
        
        gap_pct = ((first_open - prev_close) / prev_close) * 100
        
        # Filter gap
        if not (MIN_GAP <= gap_pct <= MAX_GAP):
            return None
        
        # Volume spike
        avg_vol = safe_float(volume.tail(20).mean())
        if avg_vol is None or avg_vol == 0:
            return None
        
        first_vol = safe_float(first_candle["Volume"], 0)
        if first_vol == 0:
            return None
        
        vol_ratio = first_vol / avg_vol
        
        if vol_ratio < MIN_VOL_RATIO:
            return None
        
        # Entry: close candle pertama
        entry_price = safe_float(first_candle["Close"])
        if entry_price is None:
            return None
        
        # Stop loss: di bawah low candle pertama - 0.5%
        first_low = safe_float(first_candle["Low"])
        if first_low is None or first_low == 0:
            return None
        
        stop_loss = first_low - (first_low * 0.005)  # 0.5% buffer
        
        # TP 3%
        tp_price = entry_price * (1 + TARGET_PCT / 100)
        
        # Risk check
        risk_pct = ((entry_price - stop_loss) / entry_price) * 100
        if risk_pct > MAX_RISK_PCT:
            return None
        
        # R:R
        rr = (tp_price - entry_price) / (entry_price - stop_loss)
        
        # Price sekarang
        current_price = safe_float(close.iloc[-1])
        
        # Ticker bersih
        ticker_clean = stock.replace(".JK", "")
        
        return {
            "ticker": ticker_clean,
            "entry": round(entry_price, 0),
            "stop": round(stop_loss, 0),
            "tp": round(tp_price, 0),
            "gap_pct": round(gap_pct, 2),
            "vol_ratio": round(vol_ratio, 2),
            "risk_pct": round(risk_pct, 2),
            "rr": round(rr, 2),
            "current": round(current_price, 0),
            "first_open": round(first_open, 0),
            "first_low": round(first_low, 0),
            "first_vol": round(first_vol, 0),
            "avg_vol": round(avg_vol, 2),
            "prev_close": round(prev_close, 0),
            "scan_time": datetime.now().strftime("%H:%M"),
        }
        
    except Exception as e:
        log_error(f"analyze {stock} | {e}")
        return None

def score_scalp(setup):
    """
    Scoring prioritas scalping:
    - Volume spike: 0-30 poin
    - Gap ideal: 0-25 poin
    - Risk/Reward: 0-25 poin
    - Risk kecil: 0-20 poin
    Total max: 100
    """
    score = 0
    
    # Volume spike (prioritas #1)
    vr = setup["vol_ratio"]
    if vr >= 8:
        score += 30
    elif vr >= 5:
        score += 25
    elif vr >= 3:
        score += 15
    elif vr >= 2:
        score += 10
    
    # Gap ideal 1.5-3% (prioritas #2)
    gap = setup["gap_pct"]
    if 1.5 <= gap <= 3.0:
        score += 25
    elif 1.0 <= gap < 1.5 or 3.0 < gap <= 3.5:
        score += 15
    elif gap <= 4.0:
        score += 5
    
    # R:R >= 2 (prioritas #3)
    rr = setup["rr"]
    if rr >= 3.0:
        score += 25
    elif rr >= 2.0:
        score += 20
    elif rr >= 1.5:
        score += 10
    
    # Risk kecil (prioritas #4)
    risk = setup["risk_pct"]
    if risk <= 0.8:
        score += 20
    elif risk <= 1.2:
        score += 15
    elif risk <= 1.6:
        score += 8
    elif risk <= 2.0:
        score += 3
    
    return score

# ============================================================
# SCALP SCAN
# ============================================================
def scalp_scan():
    """
    Scan pagi: cari setup scalping di 30 menit pertama market buka
    Output: kirim setup ke Telegram + simpan di CSV untuk exit check
    """
    print(f"\n=== SCALP SCAN {datetime.now().strftime('%H:%M')} ===")
    
    stocks = read_stocks()
    if not stocks:
        print("Stock list kosong.")
        return
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # Cek apakah sudah scan hari ini
    if os.path.exists(SCALP_CSV):
        with open(SCALP_CSV, "r", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        
        # Kalo masih ada setup dari scan sebelumnya, hapus
        if existing:
            print(f"  Membersihkan {len(existing)} setup lama sebelum scan baru")
            os.remove(SCALP_CSV)
    
    found_setups = []
    failed = []
    
    for i, stock in enumerate(stocks):
        print(f"  [{i+1}/{len(stocks)}] {stock}")
        
        setup = analyze_scalp_setup(stock)
        
        if setup:
            score = score_scalp(setup)
            setup["score"] = score
            found_setups.append(setup)
            print(f"    ✅ Gap: {setup['gap_pct']}% | Vol: {setup['vol_ratio']}x | Risk: {setup['risk_pct']}% | Score: {score}")
        else:
            failed.append(stock)
        
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)
    
    # Urutkan berdasarkan score descending
    found_setups.sort(key=lambda x: x["score"], reverse=True)
    
    if not found_setups:
        msg = (
            f"⏳ **SCALP SCAN {datetime.now().strftime('%H:%M')}**\n"
            f"Tidak ada setup scalping yang lolos filter.\n"
            f"Saham di-scan: {len(stocks)}\n"
            f"Gagal: {len(failed)}\n\n"
            f"_Mungkin gap terlalu besar/kecil atau volume belum spike_"
        )
        send(msg)
        print("  Tidak ada setup.")
        return
    
    # Kirim results ke Telegram
    msg_header = (
        f"⚡ **SCALPING SETUP - {datetime.now().strftime('%H:%M')} WIB**\n"
        f"Target 3% | Risk max 2% | Exit by 11:00 WIB\n"
        f"Total setup: {len(found_setups)} dari {len(stocks)} saham\n\n"
    )
    
    msg = msg_header
    for s in found_setups[:8]:  # Kirim top 8
        entry_msg = (
            f"{'🟢' if s['score'] >= 60 else '🟡' if s['score'] >= 40 else '🔴'} "
            f"**#{s['ticker']}** | Score: {s['score']}\n"
            f"┃ Entry: Rp {s['entry']:,.0f}\n"
            f"┃ Stop:  Rp {s['stop']:,.0f} ({s['risk_pct']}% risk)\n"
            f"┃ TP 3%: Rp {s['tp']:,.0f}\n"
            f"┃ Gap: {s['gap_pct']}% | Vol: {s['vol_ratio']}x | R:R {s['rr']}\n"
            f"┃ First: {s['first_open']:,.0f} | Prev Close: {s['prev_close']:,.0f}\n"
            f"[Chart](https://www.tradingview.com/chart/?symbol=IDX:{s['ticker']})\n\n"
        )
        
        if len(msg) + len(entry_msg) > 4000:
            send(msg)
            msg = ""
        
        msg += entry_msg
    
    if msg:
        send(msg)
    
    # Simpan ke CSV buat exit check
    with open(SCALP_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ticker", "entry", "stop", "tp", "gap_pct",
            "vol_ratio", "risk_pct", "rr", "score",
            "entry_time", "entry_date", "first_open", "current"
        ])
        for s in found_setups:
            w.writerow([
                s["ticker"], s["entry"], s["stop"], s["tp"],
                s["gap_pct"], s["vol_ratio"], s["risk_pct"],
                s["rr"], s["score"],
                s["scan_time"], today_date,
                s["first_open"], s["current"]
            ])
    
    print(f"\n✅ Scalp scan selesai — {len(found_setups)} setup dikirim")
    
    if failed:
        print(f"  Gagal: {len(failed)} saham")

# ============================================================
# EXIT CHECKER
# ============================================================
def check_scalp_exit():
    """
    Cek exit tiap 5 menit:
    - Kena TP 3%? → Win, kirim notif + catat ke history
    - Kena stop loss? → Loss, kirim notif + catat ke history
    - Lebih dari 2 jam? → Expired, tutup manual
    - Masih diantara? → Hold, kasih tau pergerakan
    """
    if not os.path.exists(SCALP_CSV):
        return
    
    with open(SCALP_CSV, "r", encoding="utf-8") as f:
        setups = list(csv.DictReader(f))
    
    if not setups:
        return
    
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    
    active_setups = []
    history_updates = []
    today = now.strftime("%Y-%m-%d")
    
    print(f"  ⌛ Exit check {current_time} — {len(setups)} aktif")
    
    for setup in setups:
        ticker = setup["ticker"] + ".JK"
        entry = safe_float(setup["entry"])
        stop = safe_float(setup["stop"])
        tp = safe_float(setup["tp"])
        entry_time = setup.get("entry_time", "")
        entry_date = setup.get("entry_date", "")
        score = setup.get("score", "")
        gap_pct = setup.get("gap_pct", "")
        ticker_clean = setup["ticker"]
        
        if entry is None or stop is None or tp is None:
            continue
        
        # Cek expired (>2 jam dari entry)
        if entry_time and entry_date == today:
            try:
                entry_dt = datetime.strptime(f"{today} {entry_time}", "%Y-%m-%d %H:%M")
                elapsed = (now - entry_dt).total_seconds() / 60
                
                if elapsed > MAX_HOLD_MINUTES:
                    print(f"  ⏰ {ticker_clean} expired ({elapsed:.0f}m)")
                    
                    # Kirim notif expired, cek harga terakhir
                    try:
                        df_check = yf.download(ticker, period="1d", interval="5m", auto_adjust=True, progress=False, timeout=10)
                        df_check = clean_columns(df_check)
                        if df_check is not None and not df_check.empty:
                            last_price = safe_float(df_check["Close"].iloc[-1])
                            pnl_pct = ((last_price - entry) / entry) * 100
                            
                            if pnl_pct > 0:
                                msg_exp = (
                                    f"⏳ **EXPIRED - #{ticker_clean}**\n"
                                    f"Hold >2 jam, forced exit\n"
                                    f"Entry: Rp {entry:,.0f} | Exit: Rp {last_price:,.0f}\n"
                                    f"PnL: {pnl_pct:+.2f}% (belum TP 3%)\n"
                                    f"Waktu: {current_time} WIB"
                                )
                                send(msg_exp)
                            else:
                                msg_exp = (
                                    f"⏳ **EXPIRED - #{ticker_clean}**\n"
                                    f"Hold >2 jam, cut\n"
                                    f"Entry: Rp {entry:,.0f} | Exit: Rp {last_price:,.0f}\n"
                                    f"PnL: {pnl_pct:+.2f}%\n"
                                    f"Waktu: {current_time} WIB"
                                )
                                send(msg_exp)
                    except Exception:
                        pass
                    
                    continue  # Hapus dari active
                    
            except Exception:
                pass
        
        try:
            # Fetch 5 menit terakhir
            df = yf.download(ticker, period="1d", interval="5m", auto_adjust=True, progress=False, timeout=10)
            df = clean_columns(df)
            
            if df is None or df.empty:
                active_setups.append(setup)
                continue
            
            high_today = safe_float(df["High"].max())
            low_today = safe_float(df["Low"].min())
            current_price = safe_float(df["Close"].iloc[-1])
            
            if high_today is None or low_today is None or current_price is None:
                active_setups.append(setup)
                continue
            
            # Cek TP 3% kena?
            if high_today >= tp:
                pnl = TARGET_PCT
                msg_win = (
                    f"✅ **TP 3% KENA - #{ticker_clean}**\n"
                    f"Target tercapai: Rp {tp:,.0f}\n"
                    f"Entry: Rp {entry:,.0f} | Return: +{pnl}%\n"
                    f"Gap: {gap_pct}% | Score: {score}\n"
                    f"Waktu: {current_time} WIB\n"
                    f"[Chart](https://www.tradingview.com/chart/?symbol=IDX:{ticker_clean})"
                )
                send(msg_win)
                
                history_updates.append({"ticker": ticker_clean, "result": "win", "pnl": pnl})
                print(f"  ✅ {ticker_clean} — TP kena!")
                continue  # Hapus dari active
            
            # Cek stop loss kena?
            if low_today <= stop:
                pnl_loss = ((stop - entry) / entry) * 100
                msg_loss = (
                    f"❌ **CUT LOSS - #{ticker_clean}**\n"
                    f"Stop kena: Rp {stop:,.0f}\n"
                    f"Entry: Rp {entry:,.0f} | Loss: {pnl_loss:.2f}%\n"
                    f"Gap: {gap_pct}% | Score: {score}\n"
                    f"Waktu: {current_time} WIB\n"
                    f"[Chart](https://www.tradingview.com/chart/?symbol=IDX:{ticker_clean})"
                )
                send(msg_loss)
                
                history_updates.append({"ticker": ticker_clean, "result": "loss", "pnl": pnl_loss})
                print(f"  ❌ {ticker_clean} — Stop kena!")
                continue  # Hapus dari active
            
            # Masih di zona: update current price di CSV
            setup["current"] = round(current_price, 0)
            pnl_current = ((current_price - entry) / entry) * 100
            
            # Kasih update kalo sudah bergerak +2% (mendekati TP)
            if pnl_current >= 2.0:
                msg_update = (
                    f"📊 **#{ticker_clean}** mendekati TP\n"
                    f"Harga: Rp {current_price:,.0f} ({pnl_current:+.2f}%)\n"
                    f"TP: Rp {tp:,.0f} | Gap: {gap_pct}% dari TP: {round((current_price-entry)/(tp-entry)*100, 0)}%\n"
                    f"Waktu: {current_time} WIB"
                )
                send(msg_update)
            
            active_setups.append(setup)
            
        except Exception as e:
            log_error(f"exit_check {ticker} | {e}")
            active_setups.append(setup)
            continue
    
    # Update history
    if history_updates:
        history = load_history()
        for h in history_updates:
            history["total_trades"] += 1
            if h["result"] == "win":
                history["wins"] += 1
            else:
                history["losses"] += 1
            
            history["daily_pnl"].append({
                "date": today,
                "ticker": h["ticker"],
                "result": h["result"],
                "pnl": h["pnl"],
                "time": current_time
            })
        save_history(history)
        
        # Kirim summary kalo ada yang exit
        winrate = round((history["wins"] / history["total_trades"]) * 100, 1) if history["total_trades"] > 0 else 0
        msg_summary = (
            f"📈 **SCALPING SUMMARY**\n"
            f"Hari ini: {len(history_updates)} exit ({sum(1 for h in history_updates if h['result']=='win')}W / {sum(1 for h in history_updates if h['result']=='loss')}L)\n"
            f"Total: {history['total_trades']} trades | Winrate: {winrate}%\n"
        )
        send(msg_summary)
    
    # Tulis ulang CSV dengan active setups
    if active_setups:
        with open(SCALP_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ticker", "entry", "stop", "tp", "gap_pct",
                "vol_ratio", "risk_pct", "rr", "score",
                "entry_time", "entry_date", "first_open", "current"
            ])
            for s in active_setups:
                w.writerow([
                    s.get("ticker", ""), s.get("entry", ""), s.get("stop", ""),
                    s.get("tp", ""), s.get("gap_pct", ""), s.get("vol_ratio", ""),
                    s.get("risk_pct", ""), s.get("rr", ""), s.get("score", ""),
                    s.get("entry_time", ""), s.get("entry_date", ""),
                    s.get("first_open", ""), s.get("current", "")
                ])
    else:
        # Semua setup udah exit, kosongkan file
        if os.path.exists(SCALP_CSV):
            os.remove(SCALP_CSV)
        print("  ✅ Semua setup sudah exit")
    
    print(f"  ✅ Exit check selesai — {len(active_setups)} masih aktif")

# ============================================================
# REPORT HARIAN
# ============================================================
def daily_report():
    """Kirim report akhir sesi scalping jam 11:00"""
    history = load_history()
    
    today = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in history.get("daily_pnl", []) if t["date"] == today]
    
    if not today_trades:
        send(f"📊 **SCALPING REPORT {today}**\nTidak ada trade hari ini.")
        return
    
    wins = sum(1 for t in today_trades if t["result"] == "win")
    losses = sum(1 for t in today_trades if t["result"] == "loss")
    total_pnl = sum(t["pnl"] for t in today_trades if isinstance(t["pnl"], (int, float)))
    
    msg = (
        f"📊 **SCALPING REPORT — {today}**\n\n"
        f"Trades: {len(today_trades)} ({wins}W / {losses}L)\n"
        f"Winrate: {round(wins/len(today_trades)*100, 1) if today_trades else 0}%\n"
        f"Total PnL: {total_pnl:+.2f}%\n\n"
        f"Detail:\n"
    )
    
    for t in today_trades[-10:]:  # Last 10
        emoji = "✅" if t["result"] == "win" else "❌"
        msg += f"  {emoji} {t['ticker']}: {t['pnl']:+.2f}%\n"
    
    if len(today_trades) > 10:
        msg += f"  ...dan {len(today_trades) - 10} lainnya\n"
    
    msg += f"\n_Next session besok pagi_\n"
    send(msg)
    
    print(f"\n=== REPORT HARIAN {today} ===")
    print(f"  Trades: {len(today_trades)} | Winrate: {round(wins/len(today_trades)*100, 1) if today_trades else 0}%")
    print(f"  PnL: {total_pnl:+.2f}%")

# ============================================================
# SCHEDULER
# ============================================================
def schedule_scalp():
    schedule.clear()
    
    # Scan pagi
    schedule.every().day.at("09:05").do(scalp_scan)
    schedule.every().day.at("09:10").do(scalp_scan)
    
    # Exit check tiap 5 menit dari 09:05 sampai 11:00
    for hour in range(9, 11):
        for minute in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(check_scalp_exit)
    
    # Juga jam 10:05 - 10:55 (masuk 10)
    for minute in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
        schedule.every().day.at(f"10:{minute:02d}").do(check_scalp_exit)
    
    # Report jam 11:00
    schedule.every().day.at("11:00").do(daily_report)
    
    # Juga kasih scan 09:08 sebagai fallback kalo yg pertama missed
    schedule.every().day.at("09:08").do(check_scalp_exit)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print("NEUROBRO SCALPING SCANNER - IDX")
    print("Target: +3% per market open session")
    print(f"Mulai: {datetime.now().strftime('%H:%M')} WIB")
    print("Jadwal:")
    print("  - Scan setup: 09:05 & 09:10 WIB")
    print("  - Exit check: tiap 5 menit (09:05 - 11:00 WIB)")
    print("  - Report: 11:00 WIB")
    print(f"Filter: Gap {MIN_GAP}-{MAX_GAP}% | Vol {MIN_VOL_RATIO}x | Risk max {MAX_RISK_PCT}%")
    print("=" * 55)
    
    schedule_scalp()
    
    # Run scan pertama langsung kalo masih di jam market
    now = datetime.now()
    market_open_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close_check = now.replace(hour=11, minute=0, second=0, microsecond=0)
    
    if market_open_time <= now <= market_close_check:
        print("\n🟢 Market sedang buka — menjalankan scan awal...")
        scalp_scan()
    
    print("\nMenunggu jadwal...")
    
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    main()
