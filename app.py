# app.py
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

TD_BASE = "https://api.twelvedata.com"
TD_KEY = os.getenv("TWELVE_KEY")
FH_BASE = "https://finnhub.io/api/v1"
FH_KEY = os.getenv("FINNHUB_KEY")

REQUEST_TIMEOUT_SECS = 30
MAX_RETRIES = 3
INITIAL_BACKOFF = 0.75  # seconds
CACHE_TTL_SECONDS = 10  # small TTL to ease rate limits

app = FastAPI(title="TwelveData Proxy", version="1.5.0")

# Keep CORS closed (no browser origins) unless you truly need them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=[],
)

# ---------- tiny in-memory TTL cache ----------
_cache: Dict[str, Dict[str, Any]] = {}
def _cache_key(path: str, params: Dict[str, Any]) -> str:
    items = sorted((k, str(v)) for k, v in params.items() if v is not None)
    return f"{path}|{tuple(items)}"

def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return entry["data"]

def _cache_set(key: str, data: Any):
    _cache[key] = {"ts": time.time(), "data": data}

# ---------- helpers ----------
def ensure_key():
    if not TD_KEY:
        raise HTTPException(status_code=500, detail="TWELVE_KEY env var not configured on server")

async def td_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Raw Twelve Data GET with retries + timeouts + error bubbling."""
    ensure_key()
    q = {k: v for k, v in params.items() if v is not None}
    q["apikey"] = TD_KEY

    ck = _cache_key(path, q)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    backoff = INITIAL_BACKOFF
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECS) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = await client.get(f"{TD_BASE}/{path}", params=q)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    time.sleep(backoff); backoff *= 2; continue
                if r.status_code >= 400:
                    try:
                        raise HTTPException(status_code=r.status_code, detail=r.json())
                    except Exception:
                        raise HTTPException(status_code=r.status_code, detail=r.text)
                data = r.json()
                _cache_set(ck, data)
                return data
            except httpx.RequestError as e:
                if attempt < MAX_RETRIES:
                    time.sleep(backoff); backoff *= 2; continue
                raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")

def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return 100.0 * (a / b)

def stddev(xs: List[float]) -> float:
    n = len(xs)
    if n < 2: return 0.0
    m = sum(xs)/n
    var = sum((x-m)*(x-m) for x in xs) / (n-1)
    return var ** 0.5

def macd_hist_dir(values: List[Dict[str, Any]]) -> str:
    """Return 'rising', 'falling', or 'flat' based on last two macd_hist readings."""
    if not values or len(values) < 2: return "n/a"
    last = float(values[0].get("macd_hist", 0))
    prev = float(values[1].get("macd_hist", 0))
    if last > prev: return "rising"
    if last < prev: return "falling"
    return "flat"

def last_two_numeric(values: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (last, prev) for a numeric key in Twelve Data 'values' arrays (descending)."""
    if not values: return (None, None)
    x0 = values[0].get(key)
    x1 = values[1].get(key) if len(values) > 1 else None
    return (float(x0) if x0 is not None else None, float(x1) if x1 is not None else None)

async def compute_vwap_from_1m(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Session VWAP from today's 1m bars:
    vwap = sum(typical_price*volume)/sum(volume), typical = (H+L+C)/3
    Returns (vwap, last_price). If not enough data, returns (None, last_price).
    """
    data = await td_get("time_series", {
        "symbol": symbol, "interval": "1min", "outputsize": 500, "order": "desc"
    })
    values = data.get("values", [])  # TD returns latest first (desc) when order=desc
    # Filter to today's bars (by date string prefix of 'datetime')
    if not values:
        return (None, None)
    today = values[0]["datetime"][:10]
    acc_tp_vol = 0.0
    acc_vol = 0.0
    last_close = None
    for row in reversed(values):  # iterate oldest -> newest for stable accumulation
        if not row["datetime"].startswith(today):
            continue
        h = float(row["high"]); l = float(row["low"]); c = float(row["close"]); v = float(row.get("volume", 0))
        tp = (h + l + c) / 3.0
        acc_tp_vol += tp * v
        acc_vol += v
        last_close = c
    if acc_vol <= 0:
        return (None, last_close)
    return (acc_tp_vol / acc_vol, last_close)

async def width_series_from_bbands(symbol: str, interval: str, outputsize: int = 240) -> List[float]:
    """
    Returns BB width% time series for (upper-lower)/close * 100
    """
    bb = await td_get("bbands", {
        "symbol": symbol, "interval": interval, "time_period": 20, "stddev": 2.0, "outputsize": outputsize, "order": "desc"
    })
    vals = bb.get("values", [])
    series: List[float] = []
    for v in reversed(vals):  # oldest -> newest so index 0 is session open
        upper = v.get("upper_band"); lower = v.get("lower_band"); close = v.get("close")
        if upper is None or lower is None or close is None: continue
        try:
            w = (float(upper) - float(lower)) / float(close) * 100.0
            series.append(w)
        except Exception:
            continue
    return series

async def atr_pct_series(symbol: str, interval: str, outputsize: int = 240) -> Tuple[List[float], Optional[float]]:
    """
    Returns (ATR% series, last_close). ATR% = 100 * ATR / close
    """
    atr = await td_get("atr", {
        "symbol": symbol, "interval": interval, "time_period": 14, "outputsize": outputsize, "order": "desc"
    })
    ts = await td_get("time_series", {
        "symbol": symbol, "interval": interval, "outputsize": outputsize, "order": "desc"
    })
    atr_vals = [float(v["atr"]) for v in reversed(atr.get("values", [])) if "atr" in v]
    closes = [float(v["close"]) for v in reversed(ts.get("values", [])) if "close" in v]
    n = min(len(atr_vals), len(closes))
    atrpcts = [100.0 * atr_vals[i] / closes[i] for i in range(n)]
    last_close = closes[n-1] if n > 0 else None
    return atrpcts, last_close

def classify_regime_spy(spy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply rules:
    Risk Alert if any TWO hold:
      - ADX rising >25 with RSI<45
      - BB width% expanding > +1σ from session open
      - ATR% > +1.5σ (vs intraday series baseline)
      - MFI<30 with price < VWAP
    High if ONE of the above
    Elevated if ATR% > median OR MFI<40 with falling MACD
    Else Normal
    """
    triggers = 0
    notes: List[str] = []

    adx = spy["ADX14"]; adx_delta = spy["ADX14_delta"]
    rsi = spy["RSI14"]
    bb_width = spy["BBWidthPct_last"]; bb_delta = spy["BBWidthPct_delta"]; bb_sigma = spy.get("BBWidthPct_sigma", 0.0)
    atrp = spy["ATRpct_last"]; atr_sigma = spy.get("ATRpct_sigma", 0.0); atr_mean = spy.get("ATRpct_mean", 0.0)
    mfi = spy["MFI14"]; vwap_status = spy["VWAP_status"]
    macd_dir = spy["MACD_hist_dir"]

    # T1: ADX rising >25 with RSI<45
    if adx is not None and adx_delta is not None and rsi is not None:
        if adx > 25 and adx_delta > 0 and rsi < 45:
            triggers += 1; notes.append("ADX>25 rising & RSI<45")

    # T2: BB width expanding > +1σ from session open
    if bb_delta is not None and bb_sigma is not None:
        if bb_delta > bb_sigma:
            triggers += 1; notes.append("BB width expanding > +1σ")

    # T3: ATR% > +1.5σ
    if atrp is not None and atr_sigma is not None and atr_mean is not None:
        if atrp > atr_mean + 1.5 * atr_sigma:
            triggers += 1; notes.append("ATR% > +1.5σ")

    # T4: MFI<30 & price < VWAP
    if mfi is not None and vwap_status is not None:
        if mfi < 30 and vwap_status == "below":
            triggers += 1; notes.append("MFI<30 & below VWAP")

    if triggers >= 2:
        level = "Risk Alert"
    elif triggers == 1:
        level = "High"
    else:
        # Elevated check
        elevated = False
        if atrp is not None and atr_mean is not None:
            if atrp > atr_mean:
                elevated = True
        if mfi is not None and mfi < 40 and macd_dir == "falling":
            elevated = True
        level = "Elevated" if elevated else "Normal"

    return {"regime": level, "notes": notes}

# ---------- health ----------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "twelve-proxy",
        "paths": [
            "/quote", "/time_series", "/rsi", "/macd", "/ema",
            "/bbands", "/atr", "/adx", "/mfi", "/stoch", "/market_health"
        ]
    }

@app.get("/healthz")
def healthz():
    return {"ok": True, "td_key_configured": bool(TD_KEY)}

# ---------- core passthrough routes ----------
@app.get("/quote")
async def quote(symbol: str = Query(..., min_length=1, max_length=20)):
    return JSONResponse(await td_get("quote", {"symbol": symbol}))

@app.get("/time_series")
async def time_series(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    outputsize: int = Query(100, ge=1, le=5000),
    order: str = Query("desc")
):
    return JSONResponse(await td_get("time_series", {
        "symbol": symbol, "interval": interval, "outputsize": outputsize, "order": order
    }))

@app.get("/rsi")
async def rsi(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    time_period: int = Query(14, ge=1, le=200),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return JSONResponse(await td_get("rsi", {
        "symbol": symbol, "interval": interval, "time_period": time_period, "outputsize": outputsize
    }))

@app.get("/macd")
async def macd(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    fastperiod: int = Query(12, ge=1, le=200),
    slowperiod: int = Query(26, ge=1, le=300),
    signalperiod: int = Query(9, ge=1, le=200),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return JSONResponse(await td_get("macd", {
        "symbol": symbol, "interval": interval,
        "fastperiod": fastperiod, "slowperiod": slowperiod, "signalperiod": signalperiod,
        "outputsize": outputsize
    }))

@app.get("/ema")
async def ema(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    time_period: int = Query(20, ge=1, le=300),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return JSONResponse(await td_get("ema", {
        "symbol": symbol, "interval": interval, "time_period": time_period, "outputsize": outputsize
    }))

@app.get("/bbands")
async def bbands(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    time_period: int = Query(20, ge=1, le=300),
    stddev: float = Query(2.0, ge=0.1, le=10.0),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return JSONResponse(await td_get("bbands", {
        "symbol": symbol, "interval": interval, "time_period": time_period, "stddev": stddev, "outputsize": outputsize
    }))

@app.get("/atr")
async def atr(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    time_period: int = Query(14, ge=1, le=300),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return JSONResponse(await td_get("atr", {
        "symbol": symbol, "interval": interval, "time_period": time_period, "outputsize": outputsize
    }))

@app.get("/adx")
async def adx(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    time_period: int = Query(14, ge=1, le=300),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return JSONResponse(await td_get("adx", {
        "symbol": symbol, "interval": interval, "time_period": time_period, "outputsize": outputsize
    }))

@app.get("/mfi")
async def mfi(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    time_period: int = Query(14, ge=1, le=300),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return JSONResponse(await td_get("mfi", {
        "symbol": symbol, "interval": interval, "time_period": time_period, "outputsize": outputsize
    }))

@app.get("/stoch")
async def stoch(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    fastkperiod: int = Query(14, ge=1, le=300),
    slowkperiod: int = Query(3, ge=1, le=100),
    slowdperiod: int = Query(3, ge=1, le=100),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return JSONResponse(await td_get("stoch", {
        "symbol": symbol, "interval": interval,
        "fastkperiod": fastkperiod, "slowkperiod": slowkperiod, "slowdperiod": slowdperiod,
        "outputsize": outputsize
    }))

# ---------- aggregated: Market Health / Crash-Risk ----------
@app.get("/market_health")
async def market_health(
    symbols: str = Query("SPY,QQQ,IWM", description="Comma-separated list"),
    interval: str = Query("1min", description="Intraday interval for snapshot, e.g., 1min, 5min")
):
    """
    Returns:
    {
      "timestamp": "...",
      "interval": "1min",
      "symbols": {
        "SPY": { ADX14, ADX14_delta, RSI14, MACD_hist_dir, ATRpct_last, ATRpct_mean, ATRpct_sigma,
                 BBWidthPct_last, BBWidthPct_open, BBWidthPct_delta, BBWidthPct_sigma,
                 MFI14, VWAP, Last, VWAP_status },
        ...
      },
      "regime": { "label": "Risk Alert|High|Elevated|Normal", "notes": [...] }
    }
    """
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    out: Dict[str, Any] = {}
    for sym in syms:
        # Indicators (intraday)
        adx = await td_get("adx", {"symbol": sym, "interval": interval, "time_period": 14, "outputsize": 3, "order": "desc"})
        rsi = await td_get("rsi", {"symbol": sym, "interval": interval, "time_period": 14, "outputsize": 2, "order": "desc"})
        macd = await td_get("macd", {"symbol": sym, "interval": interval, "fastperiod": 12, "slowperiod": 26, "signalperiod": 9, "outputsize": 3, "order": "desc"})
        mfi = await td_get("mfi", {"symbol": sym, "interval": interval, "time_period": 14, "outputsize": 2, "order": "desc"})

        adx_last, adx_prev = last_two_numeric(adx.get("values", []), "adx")
        rsi_last, _ = last_two_numeric(rsi.get("values", []), "rsi")
        macd_dir = macd_hist_dir(macd.get("values", []))
        mfi_last, _ = last_two_numeric(mfi.get("values", []), "mfi")

        # ATR% series
        atrpcts, last_close_from_atr = await atr_pct_series(sym, interval, outputsize=240)
        atr_mean = sum(atrpcts)/len(atrpcts) if atrpcts else None
        atr_sig = stddev(atrpcts) if atrpcts else None
        atr_last = atrpcts[-1] if atrpcts else None

        # BB width% series + delta from session open
        bb_widths = await width_series_from_bbands(sym, interval, outputsize=240)
        bb_open = bb_widths[0] if bb_widths else None
        bb_last = bb_widths[-1] if bb_widths else None
        bb_delta = (bb_last - bb_open) if (bb_last is not None and bb_open is not None) else None
        bb_sig = stddev(bb_widths) if bb_widths else None

        # VWAP from today's 1m bars
        vwap, last_px = await compute_vwap_from_1m(sym)
        vwap_status = None
        if vwap is not None and last_px is not None:
            vwap_status = "above" if last_px >= vwap else "below"

        out[sym] = {
            "ADX14": adx_last, "ADX14_delta": (adx_last - adx_prev) if (adx_last is not None and adx_prev is not None) else None,
            "RSI14": rsi_last,
            "MACD_hist_dir": macd_dir,
            "ATRpct_last": atr_last, "ATRpct_mean": atr_mean, "ATRpct_sigma": atr_sig,
            "BBWidthPct_last": bb_last, "BBWidthPct_open": bb_open, "BBWidthPct_delta": bb_delta, "BBWidthPct_sigma": bb_sig,
            "MFI14": mfi_last,
            "VWAP": vwap, "Last": last_px, "VWAP_status": vwap_status
        }

    # Classify regime using SPY
    spy_metrics = out.get("SPY")
    regime = {"regime": "Unknown", "notes": []}
    if spy_metrics:
        regime = classify_regime_spy(spy_metrics)

    return JSONResponse({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "interval": interval,
        "symbols": out,
        "regime": {"label": regime["regime"], "notes": regime["notes"]}
    })

