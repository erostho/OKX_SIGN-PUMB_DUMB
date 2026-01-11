# main.py
import os, time, math, requests

OKX_BASE = "https://www.okx.com"

# ===== ENV (simple) =====
TG_TOKEN    = os.getenv("TG_TOKEN")
TG_CHAT_ID  = os.getenv("TG_CHAT_ID")

MAIN_TF     = os.getenv("MAIN_TF", "5m")
CONFIRM_1   = os.getenv("CONFIRM_TF_1", "15m")
CONFIRM_2   = os.getenv("CONFIRM_TF_2", "30m")

TOP_N       = int(os.getenv("TOP_N", "100"))
MIN_STARS   = int(os.getenv("MIN_STARS", "4"))
MODE        = os.getenv("MODE", "BOTH").upper()  # PUMB_ONLY / DUMB_ONLY / BOTH

EXIT_SAFE_PCT = float(os.getenv("EXIT_SAFE_PCT", "1.22"))
EXIT_OPT_PCT  = float(os.getenv("EXIT_OPT_PCT", "2.04"))

SL_PCT = float(os.getenv("SL_PCT", "0.8"))
ATR_LEN  = int(os.getenv("ATR_LEN", "14"))
ATR_MULT = float(os.getenv("ATR_MULT", "1.2"))
ATR_TF   = os.getenv("ATR_TF", MAIN_TF)  # mặc định dùng main TF


MIN_24H_USD_VOL   = float(os.getenv("MIN_24H_USD_VOL", "10000000"))  # 10M default
MIN_5M_USD_VOL    = float(os.getenv("MIN_5M_USD_VOL", "200000"))     # 200k default
MAX_EMA_DIST_PCT  = float(os.getenv("MAX_EMA_DIST_PCT", "1.2"))      # filter chase (đu đỉnh/đu đáy)

CANDLES_LIMIT     = int(os.getenv("CANDLES_LIMIT", "210"))
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "12"))
MOVERS_BARS       = int(os.getenv("MOVERS_BARS", "3"))               # 3 bars 5m => 15m movers
VOL_SPIKE_MULT    = float(os.getenv("VOL_SPIKE_MULT", "3.0"))

# ===== HTTP =====
def get(url, params=None):
    last_err = None
    for attempt in (1, 2):  # retry 1 lần
        try:
            r = requests.get(url, params=params, timeout=12)
            r.raise_for_status()
            j = r.json()
            return j.get("data", [])
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise last_err
    
DEBUG = os.getenv("DEBUG", "1") == "1"
def dlog(*args):
    if DEBUG:
        print("[DEBUG]", *args, flush=True)

# OKX candles: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
def candles(inst, tf, limit=CANDLES_LIMIT):
    rows = get(f"{OKX_BASE}/api/v5/market/candles", {"instId": inst, "bar": tf, "limit": str(limit)})
    return list(reversed(rows))  # oldest -> newest

# ===== Indicators =====
def ema(vals, n):
    if len(vals) < n:
        return [None] * len(vals)
    k = 2/(n+1)
    out = [sum(vals[:n])/n]
    for v in vals[n:]:
        out.append(v*k + out[-1]*(1-k))
    return [None]*(n-1) + out

def rsi(vals, n=14):
    if len(vals) < n+1:
        return [None] * len(vals)
    gains, losses = [], []
    for i in range(1, len(vals)):
        d = vals[i]-vals[i-1]
        gains.append(max(d,0))
        losses.append(max(-d,0))
    avg_g, avg_l = sum(gains[:n])/n, sum(losses[:n])/n
    out = [None]*n
    rs = avg_g/avg_l if avg_l else 99
    out.append(100-100/(1+rs))
    for i in range(n, len(gains)):
        avg_g=(avg_g*(n-1)+gains[i])/n
        avg_l=(avg_l*(n-1)+losses[i])/n
        rs=avg_g/avg_l if avg_l else 99
        out.append(100-100/(1+rs))
    return out
    
def atr(highs, lows, closes, n=14):
    if len(closes) < n + 1:
        return None

    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        trs.append(tr)

    # Wilder smoothing
    atr_val = sum(trs[:n]) / n
    for tr in trs[n:]:
        atr_val = (atr_val * (n - 1) + tr) / n

    return atr_val

def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0: return 0.0
    mid = n//2
    return ys[mid] if n%2==1 else (ys[mid-1]+ys[mid])/2

def vwap(tp_list, vol_list):
    s = 0.0
    sv = 0.0
    for tp, v in zip(tp_list, vol_list):
        s += tp * v
        sv += v
    return (s/sv) if sv > 0 else None

def fmt_price(x: float) -> str:
    return f"{x:,.5f}"

# ===== Telegram =====
def send(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        raise RuntimeError("Missing TG_TOKEN or TG_CHAT_ID")
    requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True},
        timeout=15
    )

def send_long(text: str):
    # Telegram limit ~4096 chars
    max_len = 3900
    for i in range(0, len(text), max_len):
        send(text[i:i+max_len])

# ===== Core scoring =====
def score_signal(inst: str):
    # main candles
    m = candles(inst, MAIN_TF)
    if len(m) < max(60, BREAKOUT_LOOKBACK + 30):
        return None

    o = [float(x[1]) for x in m]
    h = [float(x[2]) for x in m]
    l = [float(x[3]) for x in m]
    c = [float(x[4]) for x in m]
    v = [float(x[5]) for x in m]
    vq = [float(x[7]) for x in m]  # quote volume (USDT), rất hữu ích để lọc rác

    close = c[-1]
    open_ = o[-1]

    # ---- Liquidity filter (5m quote vol) ----
    if vq[-1] < MIN_5M_USD_VOL:
        return None

    # ---- Breakout/Breakdown (N bars) ----
    N = BREAKOUT_LOOKBACK
    highN = max(h[-(N+1):-1])
    lowN  = min(l[-(N+1):-1])

    is_pumb = close > highN
    is_dumb = close < lowN
    if not is_pumb and not is_dumb:
        return None

    side = "PUMB" if is_pumb else "DUMB"

    # ---- Overextended filter (lọc đu đỉnh/đu đáy) ----
    # Reject if price too far from EMA20 (mean-revert risk)
    e20 = ema(c, 20)
    ema20_now = e20[-1]
    if ema20_now is None:
        return None

    dist_pct = abs(close - ema20_now) / close * 100
    if dist_pct > MAX_EMA_DIST_PCT:
        return None

    # ---- VWAP filter (tránh break giả) ----
    # Use last 40 bars main TF
    w = 40 if len(c) >= 40 else len(c)
    tp = [ (h[-w+i] + l[-w+i] + c[-w+i]) / 3 for i in range(w) ]
    vw = vwap(tp, v[-w:])
    if vw is None:
        return None
    if side == "PUMB" and close < vw:
        return None
    if side == "DUMB" and close > vw:
        return None

    # ---- Stars base ----
    stars = 1

    # ---- Volume spike (step 1) ----
    vmw = 48 if len(vq) >= 48 else len(vq)
    vmed = median(vq[-vmw:])
    vspike = (vq[-1] / vmed) if vmed > 0 else 0
    if vspike >= VOL_SPIKE_MULT:
        stars += 1

    # ---- Clean candle (body / range) ----
    rng = max(h[-1]-l[-1], 1e-9)
    body_ratio = abs(close-open_) / rng
    if body_ratio >= 0.6:
        stars += 1

    # ---- Confirm 15m & 30m (EMA + RSI + slope) + Volume step 2 ----
    for tf in (CONFIRM_1, CONFIRM_2):
        cc_rows = candles(inst, tf, limit=200)
        if len(cc_rows) < 60:
            continue

        cc = [float(x[4]) for x in cc_rows]
        ee9  = ema(cc, 9)
        ee21 = ema(cc, 21)
        rr   = rsi(cc, 14)

        e9 = ee9[-1]; e21 = ee21[-1]; r = rr[-1]
        if e9 is None or e21 is None or r is None:
            continue

        # slope filter (trend strength)
        # require ema20 slope same direction
        e20c = ema(cc, 20)
        if len(e20c) < 2 or e20c[-1] is None or e20c[-2] is None:
            continue
        slope = e20c[-1] - e20c[-2]

        ok_trend = False
        if side == "PUMB":
            ok_trend = (e9 > e21) and (r >= 55) and (slope > 0)
        else:
            ok_trend = (e9 < e21) and (r <= 45) and (slope < 0)

        if ok_trend:
            stars += 1

        # Volume spike step 2 (confirm TF) using quote vol if available
        # cc_rows candle has volCcyQuote at index 7 too
        vq2 = [float(x[7]) for x in cc_rows]
        if len(vq2) >= 30:
            med2 = median(vq2[-30:])
            if med2 > 0 and (vq2[-1] / med2) >= 2.0:
                stars += 1  # extra boost if confirm TF also has vol pop

    stars = min(stars, 5)
    
    # ---- exits + SL by ATR (per coin) ----
    atr_rows = m if ATR_TF == MAIN_TF else candles(inst, ATR_TF, limit=200)
    ah = [float(x[2]) for x in atr_rows]
    al = [float(x[3]) for x in atr_rows]
    ac = [float(x[4]) for x in atr_rows]
    
    atr_val = atr(ah, al, ac, ATR_LEN)
    if atr_val is None:
        return None
    
    sl_dist = atr_val * ATR_MULT
    
    if side == "PUMB":
        sl = close - sl_dist
        exit_safe = close + sl_dist * (EXIT_SAFE_PCT / ATR_MULT)
        exit_opt  = close + sl_dist * (EXIT_OPT_PCT  / ATR_MULT)
    else:
        sl = close + sl_dist
        exit_safe = close - sl_dist * (EXIT_SAFE_PCT / ATR_MULT)
        exit_opt  = close - sl_dist * (EXIT_OPT_PCT  / ATR_MULT)
    
    sl_pct   = (sl - close) / close * 100
    safe_pct = (exit_safe - close) / close * 100
    opt_pct  = (exit_opt  - close) / close * 100

    return {
        "inst": inst,
        "side": side,
        "stars": stars,
        "entry": close,
        "sl": sl,
        "exit_safe": exit_safe,
        "exit_opt": exit_opt,
        "safe_pct": safe_pct,
        "opt_pct": opt_pct,
        "sl_pct": sl_pct,
    }


# ===== Pick TOP movers (short-term) =====
def pick_top_movers():
    t0 = time.time()
    tks = get(f"{OKX_BASE}/api/v5/market/tickers", {"instType": "SWAP"})
    dlog("tickers=", len(tks))

    # 1) filter USDT + vol24h
    candidates = []
    bad_usdt = 0
    bad_vol24 = 0

    for t in tks:
        inst = t.get("instId", "")
        if not inst or "USDT" not in inst:
            bad_usdt += 1
            continue

        vol24 = float(t.get("volCcyQuote", "0") or "0")
        if vol24 < MIN_24H_USD_VOL:
            bad_vol24 += 1
            continue

        candidates.append((vol24, inst))

    print(f"[FILTER] usdt_ok={len(tks)-bad_usdt} vol24_ok={len(candidates)} vol24_cut={bad_vol24} MIN_24H={MIN_24H_USD_VOL}",
          flush=True)

    if not candidates:
        # debug: in thử 5 tickers vol lớn nhất để xem volCcyQuote có đang =0 không
        top_vol = sorted(
            [(float(x.get("volCcyQuote","0") or "0"), x.get("instId","")) for x in tks if x.get("instId")],
            reverse=True
        )[:5]
        print("[HINT] candidates=0. Top volCcyQuote samples:", top_vol, flush=True)
        return []

    # 2) cap candidates (giảm số lần gọi candles)
    candidates.sort(reverse=True, key=lambda x: x[0])
    cap = min(250, len(candidates))
    candidates = candidates[:cap]
    dlog("candidates_capped=", len(candidates))

    # 3) short-term movers
    movers = []
    fail_candles = 0
    too_short = 0

    for _, inst in candidates:
        try:
            rows = candles(inst, MAIN_TF, limit=60)
            if len(rows) < (MOVERS_BARS + 2):
                too_short += 1
                continue
            closes = [float(x[4]) for x in rows]
            chg = abs(closes[-1] / closes[-(MOVERS_BARS+1)] - 1.0)
            movers.append((chg, inst))
        except Exception as e:
            fail_candles += 1
            dlog("candles_fail", inst, MAIN_TF, str(e))
            continue

    print(f"[MOVERS] cap={cap} movers_ok={len(movers)} candles_fail={fail_candles} too_short={too_short} TF={MAIN_TF} bars={MOVERS_BARS}",
          flush=True)

    movers.sort(reverse=True, key=lambda x: x[0])
    out = [x[1] for x in movers[:TOP_N]]
    print(f"[INFO] movers selected={len(out)} elapsed={time.time()-t0:.2f}s", flush=True)

    # debug: in thử 5 movers đầu để xem có hợp lý
    if out:
        dlog("top5_movers=", movers[:5])

    return out


def build_msg(sig):
    inst = sig["inst"].replace("-SWAP", "")
    side_text = "🟢 PUMB" if sig["side"] == "PUMB" else "🔴 DUMB"
    return (
        f"{side_text} {sig['stars']}⭐ | {inst}\n"
        f"Entry: {fmt_price(sig['entry'])}\n"
        f"✅ TP an toàn: {fmt_price(sig['exit_safe'])}  ({sig['safe_pct']:+.2f}%)\n"
        f"🎯 TP tối ưu: {fmt_price(sig['exit_opt'])}  ({sig['opt_pct']:+.2f}%)\n"
        f"🔴 SL: {fmt_price(sig['sl'])}  ({sig['sl_pct']:+.2f}%)\n"
        f"⏱ Timeframe: {MAIN_TF}"
    )

def run():
    t0 = time.time()
    now_str = time.strftime("%H:%M")

    print(f"[RUN] scan={TOP_N} inst_type=SWAP main={MAIN_TF} conf={CONFIRM_1},{CONFIRM_2}", flush=True)

    symbols = pick_top_movers()
    print(f"[INFO] movers_selected={len(symbols)}", flush=True)

    msgs_pumb = []
    msgs_dumb = []

    checked = 0
    passed = 0

    for inst in symbols:
        checked += 1
        sig = None
        try:
            sig = score_signal(inst)
        except Exception:
            continue

        if not sig:
            continue

        if sig["stars"] < MIN_STARS:
            continue

        if MODE == "PUMB_ONLY" and sig["side"] != "PUMB":
            continue
        if MODE == "DUMB_ONLY" and sig["side"] != "DUMB":
            continue
        passed += 1
        if sig["side"] == "PUMB":
            msgs_pumb.append(build_msg(sig))
        else:
            msgs_dumb.append(build_msg(sig))

    if not msgs_pumb and not msgs_dumb:
        print(f"[DONE] sent=0 elapsed={time.time()-t0:.2f}s", flush=True)
        return


    header = f"📡 Scan OKX | {now_str} | TF={MAIN_TF} | MIN⭐={MIN_STARS}\n"
    blocks = [header]

    if msgs_pumb:
        blocks.append("🟢 PUMB\n" + "\n\n".join(msgs_pumb))
    if msgs_dumb:
        blocks.append("🔴 DUMB\n" + "\n\n".join(msgs_dumb))

    big_msg = "\n\n".join(blocks)
    print(f"[FILTER] checked={checked} passed={passed} pumb={len(msgs_pumb)} dumb={len(msgs_dumb)}", flush=True)
    send_long(big_msg)

if __name__ == "__main__":
    t0 = time.time()
    sent = 0
    try:
        # nếu trong run() bạn không trả sent, cứ để run() tự print,
        # còn muốn chắc chắn, bạn có thể tăng sent trong run() và return sent.
        run()
    except Exception as e:
        print("[FATAL]", e, flush=True)
        raise
    finally:
        print(f"[DONE] elapsed={time.time()-t0:.2f}s", flush=True)


