# MAIN_BOT.py
# OKX PUMB/DUMB Signal Bot (Render-friendly, config via ENV)
# - Scan Top movers (OKX public tickers)
# - Main timeframe setup + confirm TF1/TF2
# - Star scoring 1..5
# - Telegram notify with Entry / Exit safe / Exit optimal
# - Cooldown via Redis (recommended) or memory fallback

import os
import json
import time
import math
from typing import List, Dict, Optional, Literal, Tuple

import requests

# =========================
# 1) CONFIG (ENV first)
# =========================

DEFAULT_CONFIG = {
    # Universe
    "inst_type": "SWAP",             # "SWAP" or "SPOT"
    "quote_ccy": "USDT",
    "top_n": 200,
    "min_24h_usd_vol": 50000,    # filter low liquidity

    # Timeframes (OKX bar: 1m,3m,5m,15m,30m,1H,2H,4H,1D...)
    "tf_main": "5m",
    "tf_confirm_1": "15m",
    "tf_confirm_2": "30m",
    "lookback_main_bars": 200,
    "lookback_confirm_bars": 200,

    # Messaging / throttle
    "timeframe_min": 30,             # shown in Telegram
    "cooldown_min": 30,
    "min_stars_to_send": 4,
    "mode": "BOTH",                  # "PUMB_ONLY" | "DUMB_ONLY" | "BOTH"

    # Setup rules
    "breakout_lookback_bars": 12,     # N bars (main TF) range breakout
    "vol_spike_mult": 3.0,           # vol spike threshold
    "vol_median_window": 48,         # median window for vol

    # Confirm rules
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_len": 14,
    "rsi_pumb": 55.0,
    "rsi_dumb": 45.0,

    # Exits (by % from entry)
    "exit_safe_pct": 1.22,
    "exit_opt_pct": 2.04,

    # OKX base
    "okx_base": "https://www.okx.com",
}

def load_config() -> Dict:
    raw = os.getenv("BOT_CONFIG_JSON", "").strip()
    cfg = dict(DEFAULT_CONFIG)
    if raw:
        try:
            cfg.update(json.loads(raw))
        except Exception as e:
            print("[CONFIG] BOT_CONFIG_JSON parse error:", e)
    # OKX_BASE env can override
    if os.getenv("OKX_BASE"):
        cfg["okx_base"] = os.getenv("OKX_BASE").strip()
    return cfg

CFG = load_config()

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "").strip()

# =========================
# 2) STATE STORE (Redis preferred)
# =========================

class StateStore:
    def __init__(self, redis_url: str):
        self.redis = None
        self.mem = {}
        if redis_url:
            try:
                import redis  # type: ignore
                self.redis = redis.from_url(redis_url, decode_responses=True)
                self.redis.ping()
                print("[STATE] Redis connected")
            except Exception as e:
                print("[STATE] Redis not available, fallback memory:", e)
                self.redis = None

    def get(self, key: str) -> Optional[str]:
        if self.redis:
            return self.redis.get(key)
        return self.mem.get(key)

    def set(self, key: str, value: str):
        if self.redis:
            self.redis.set(key, value)
        else:
            self.mem[key] = value

STORE = StateStore(REDIS_URL)

def cooldown_ok(symbol: str, cooldown_min: int) -> bool:
    now = int(time.time())
    key = f"cooldown:{symbol}"
    last = STORE.get(key)
    if not last:
        return True
    try:
        last_ts = int(last)
    except:
        return True
    return (now - last_ts) >= cooldown_min * 60

def mark_sent(symbol: str):
    now = int(time.time())
    STORE.set(f"cooldown:{symbol}", str(now))

# =========================
# 3) OKX CLIENT (public REST)
# =========================

def okx_get(path: str, params: dict | None = None) -> dict:
    url = CFG["okx_base"] + path
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def okx_tickers(inst_type: str) -> List[Dict]:
    j = okx_get("/api/v5/market/tickers", {"instType": inst_type})
    return j.get("data", [])

def okx_candles(inst_id: str, bar: str, limit: int) -> List[List[str]]:
    j = okx_get("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": str(limit)})
    rows = j.get("data", [])
    # OKX returns newest-first => reverse to oldest-first
    return list(reversed(rows))

# =========================
# 4) INDICATORS (EMA / RSI / median)
# =========================

def median(xs: List[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.0
    mid = n // 2
    return ys[mid] if (n % 2 == 1) else (ys[mid - 1] + ys[mid]) / 2

def ema(values: List[float], length: int) -> List[float]:
    if len(values) < length:
        return [math.nan] * len(values)
    k = 2 / (length + 1)
    out = [math.nan] * (length - 1)
    prev = sum(values[:length]) / length
    out.append(prev)
    for v in values[length:]:
        prev = v * k + prev * (1 - k)
        out.append(prev)
    return out

def rsi(values: List[float], length: int = 14) -> List[float]:
    if len(values) < length + 1:
        return [math.nan] * len(values)

    gains = []
    losses = []
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length

    out = [math.nan] * length
    rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    out.append(100 - (100 / (1 + rs)))

    for i in range(length, len(gains)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        out.append(100 - (100 / (1 + rs)))
    return out

# =========================
# 5) SCORING (PUMB/DUMB + stars)
# =========================

def parse_ohlcv(rows: List[List[str]]) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    # OKX candle: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    o = [float(x[1]) for x in rows]
    h = [float(x[2]) for x in rows]
    l = [float(x[3]) for x in rows]
    c = [float(x[4]) for x in rows]
    v = [float(x[5]) for x in rows]
    return o, h, l, c, v

def confirm_tf(rows: List[List[str]]) -> Tuple[float, float, float]:
    _o, _h, _l, _c, _v = parse_ohlcv(rows)
    ef = ema(_c, int(CFG["ema_fast"]))[-1]
    es = ema(_c, int(CFG["ema_slow"]))[-1]
    rr = rsi(_c, int(CFG["rsi_len"]))[-1]
    return ef, es, rr

def score_symbol(main_rows, c1_rows, c2_rows) -> Dict:
    o, h, l, c, v = parse_ohlcv(main_rows)

    need = max(int(CFG["breakout_lookback_bars"]) + 2, int(CFG["vol_median_window"]) + 2, 60)
    if len(c) < need:
        return {"side": None, "stars": 0}

    close = c[-1]
    open_ = o[-1]

    N = int(CFG["breakout_lookback_bars"])
    highN = max(h[-(N + 1):-1])
    lowN  = min(l[-(N + 1):-1])

    is_pumb = close > highN
    is_dumb = close < lowN
    if not is_pumb and not is_dumb:
        return {"side": None, "stars": 0}

    side = "PUMB" if is_pumb else "DUMB"
    stars = 1

    # Volume spike
    vmw = int(CFG["vol_median_window"])
    vwin = v[-vmw:]
    vmed = median(vwin)
    vspike = (v[-1] / vmed) if vmed > 0 else 0.0
    if vspike >= float(CFG["vol_spike_mult"]):
        stars += 1

    # Clean candle (body/range)
    rng = max(h[-1] - l[-1], 1e-9)
    body = abs(close - open_)
    body_ratio = body / rng
    if body_ratio >= 0.6:
        stars += 1

    # Confirm TFs
    ef1, es1, r1 = confirm_tf(c1_rows)
    ef2, es2, r2 = confirm_tf(c2_rows)

    if side == "PUMB":
        if ef1 > es1 and r1 >= float(CFG["rsi_pumb"]):
            stars += 1
        if ef2 > es2 and r2 >= float(CFG["rsi_pumb"]):
            stars += 1
    else:
        if ef1 < es1 and r1 <= float(CFG["rsi_dumb"]):
            stars += 1
        if ef2 < es2 and r2 <= float(CFG["rsi_dumb"]):
            stars += 1

    stars = min(stars, 5)

    # Exit levels by %
    safe_pct = float(CFG["exit_safe_pct"]) / 100.0
    opt_pct  = float(CFG["exit_opt_pct"]) / 100.0

    if side == "PUMB":
        exit_safe = close * (1 + safe_pct)
        exit_opt  = close * (1 + opt_pct)
    else:
        exit_safe = close * (1 - safe_pct)
        exit_opt  = close * (1 - opt_pct)

    return {
        "side": side,
        "stars": stars,
        "entry": close,
        "exit_safe": exit_safe,
        "exit_opt": exit_opt,
        "vspike": vspike,
        "body_ratio": body_ratio,
    }

# =========================
# 6) TELEGRAM
# =========================

def send_telegram(text: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        raise RuntimeError("Missing TG_TOKEN or TG_CHAT_ID env var")
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    r = requests.post(
        url,
        json={"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True},
        timeout=20
    )
    r.raise_for_status()

def fmt_msg(symbol: str, side: str, stars: int, entry: float, exit_safe: float, exit_opt: float, timeframe_min: int) -> str:
    safe_pct = (exit_safe / entry - 1) * 100
    opt_pct  = (exit_opt  / entry - 1) * 100

    if side == "PUMB":
        return (
            f"🟢 PUMB {stars}⭐ | {symbol}\n"
            f"Entry: {entry:,.0f}\n"
            f"✅ Exit an toàn: {exit_safe:,.0f}  ({safe_pct:+.2f}%)\n"
            f"🎯 Exit tối ưu: {exit_opt:,.0f} ({opt_pct:+.2f}%)\n"
            f"⏱ Timeframe: {timeframe_min} phút"
        )
    else:
        return (
            f"🔴 DUMB {stars}⭐ | {symbol}\n"
            f"Entry: {entry:,.0f}\n"
            f"✅ Exit an toàn: {exit_safe:,.0f}  ({safe_pct:+.2f}%)\n"
            f"🎯 Exit tối ưu: {exit_opt:,.0f} ({opt_pct:+.2f}%)\n"
            f"⏱ Timeframe: {timeframe_min} phút"
        )

# =========================
# 7) PICK TOP MOVERS
# =========================


def pick_top_symbols() -> List[str]:
    inst_type = str(CFG["inst_type"])
    quote_ccy = str(CFG["quote_ccy"])
    top_n = int(CFG["top_n"])
    min_vol = float(CFG["min_24h_usd_vol"])

    tks = okx_tickers(inst_type)

    picked = []
    for t in tks:
        inst_id = t.get("instId", "")
        if not inst_id:
            continue

        # Filter quote currency
        if quote_ccy not in inst_id:
            continue

        # ===== FIX: Volume fallback for SWAP =====
        # OKX SWAP tickers thường có volCcy24h (base volume) + last (price)
        last = float(t.get("last", "0") or "0")

        vol_quote = float(t.get("volCcyQuote", "0") or "0")  # nếu có thì dùng
        if vol_quote <= 0:
            vol_base = float(t.get("volCcy24h", "0") or "0")
            if vol_base <= 0:
                vol_base = float(t.get("volCcy", "0") or "0")  # fallback cuối
            vol_quote = vol_base * last  # ước tính quote volume

        if vol_quote < min_vol:
            continue

        chg = float(t.get("chg24h", "0") or "0")
        picked.append((abs(chg), inst_id))

    picked.sort(reverse=True, key=lambda x: x[0])
    return [x[1] for x in picked[:top_n]]


# =========================
# 8) MAIN RUN
# =========================

def run_once():
    mode = str(CFG["mode"]).upper()
    min_stars = int(CFG["min_stars_to_send"])
    cooldown_min = int(CFG["cooldown_min"])
    tf_main = str(CFG["tf_main"])
    tf1 = str(CFG["tf_confirm_1"])
    tf2 = str(CFG["tf_confirm_2"])
    lb_main = int(CFG["lookback_main_bars"])
    lb_conf = int(CFG["lookback_confirm_bars"])
    timeframe_min = int(CFG["timeframe_min"])

    symbols = pick_top_symbols()
    print(f"[RUN] scan={len(symbols)} inst_type={CFG['inst_type']} main={tf_main} conf={tf1},{tf2}")

    sent = 0

    for inst_id in symbols:
        # cooldown
        if not cooldown_ok(inst_id, cooldown_min):
            continue

        # fetch candles
        try:
            main_rows = okx_candles(inst_id, tf_main, lb_main)
            c1_rows = okx_candles(inst_id, tf1, lb_conf)
            c2_rows = okx_candles(inst_id, tf2, lb_conf)
        except Exception as e:
            print("[DATA] candles error", inst_id, e)
            continue

        # score
        res = score_symbol(main_rows, c1_rows, c2_rows)
        side = res.get("side")
        stars = int(res.get("stars", 0))
        if not side or stars < min_stars:
            continue

        if mode == "PUMB_ONLY" and side != "PUMB":
            continue
        if mode == "DUMB_ONLY" and side != "DUMB":
            continue

        # format symbol display (remove -SWAP suffix to look cleaner)
        symbol_display = inst_id.replace("-SWAP", "")

        msg = fmt_msg(
            symbol=symbol_display,
            side=side,
            stars=stars,
            entry=float(res["entry"]),
            exit_safe=float(res["exit_safe"]),
            exit_opt=float(res["exit_opt"]),
            timeframe_min=timeframe_min
        )

        try:
            send_telegram(msg)
            mark_sent(inst_id)
            sent += 1
            print("[SEND]", inst_id, side, stars)
        except Exception as e:
            print("[TG] send error", inst_id, e)

    print(f"[DONE] sent={sent}")

if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        print("[FATAL]", e)
        raise
