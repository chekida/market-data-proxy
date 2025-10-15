# app_fh.py
import os, time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

# ===== Settings =====
FH_BASE = "https://finnhub.io/api/v1"
FH_KEY = os.getenv("FINNHUB_KEY")
REQUEST_TIMEOUT_SECS = 30
MAX_RETRIES = 3
INITIAL_BACKOFF = 0.75
CACHE_TTL_SECONDS = 10  # tiny cache to ease rate limits

# ===== App =====
app = FastAPI(title="Finnhub Proxy", version="1.1.1")

# keep CORS closed unless you truly need browser calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=[],
)

# ===== Tiny in-memory cache =====
_cache: Dict[str, Dict[str, Any]] = {}
def _ck(path: str, params: Dict[str, Any]) -> str:
    items = sorted((k, str(v)) for k, v in params.items() if v is not None)
    return f"{path}|{tuple(items)}"

def _get(k: str) -> Optional[Any]:
    e = _cache.get(k)
    if not e:
        return None
    if time.time() - e["ts"] > CACHE_TTL_SECONDS:
        _cache.pop(k, None)
        return None
    return e["data"]

def _set(k: str, data: Any):
    _cache[k] = {"ts": time.time(), "data": data}

# ===== Upstream caller with retries =====
async def fh_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not FH_KEY:
        raise HTTPException(status_code=500, detail="FINNHUB_KEY env var not configured on server")
    q = {k: v for k, v in params.items() if v is not None}
    q["token"] = FH_KEY  # Finnhub auth

    key = _ck(path, q)
    cached = _get(key)
    if cached is not None:
        return cached

    backoff = INITIAL_BACKOFF
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECS) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = await client.get(f"{FH_BASE}/{path}", params=q)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    time.sleep(backoff); backoff *= 2; continue
                if r.status_code >= 400:
                    try:
                        raise HTTPException(status_code=r.status_code, detail=r.json())
                    except Exception:
                        raise HTTPException(status_code=r.status_code, detail=r.text)
                data = r.json()
                _set(key, data)
                return data
            except httpx.RequestError as e:
                if attempt < MAX_RETRIES:
                    time.sleep(backoff); backoff *= 2; continue
                raise HTTPException(status_code=502, detail=f"Finnhub upstream error: {str(e)}")

# ===== Health =====
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "finnhub-proxy",
        "paths": ["/fh/quote","/fh/candles","/fh/indicator","/fh/profile","/fh/news"]
    }

@app.get("/healthz")
def healthz():
    return {"ok": True, "finnhub_key_configured": bool(FH_KEY)}

# ===== Finnhub routes =====

@app.get("/fh/quote")
async def fh_quote(symbol: str = Query(..., min_length=1, max_length=20)):
    """
    Real-time quote (Finnhub /quote)
    """
    return JSONResponse(await fh_get("quote", {"symbol": symbol}))

@app.get("/fh/candles")
async def fh_candles(
    symbol: str = Query(..., min_length=1, max_length=20),
    resolution: str = Query("D", description="Valid: 1,5,15,30,60,D,W,M"),
    _from: int = Query(..., alias="from", description="UNIX seconds"),
    to: int = Query(..., description="UNIX seconds"),
):
    """
    Historical candles (Finnhub /stock/candle)
    """
    return JSONResponse(await fh_get("stock/candle", {
        "symbol": symbol, "resolution": resolution, "from": _from, "to": to
    }))

@app.get("/fh/indicator")
async def fh_indicator(
    symbol: str = Query(..., min_length=1, max_length=20),
    resolution: str = Query("D"),
    indicator: str = Query(..., description="rsi, macd, ema, sma, adx, mfi, stoch, bbands, atr, ..."),
    timeperiod: int = Query(14),
    **extras
):
    """
    Technical indicator (Finnhub /indicator)
    - We accept arbitrary query params and forward them to Finnhub,
      but we ignore the placeholder param 'extras' and any empty/defaults.
    - Example: fastperiod=12&slowperiod=26&signalperiod=9 for MACD
    """
    # Drop placeholder/empty params some tools send
    sanitized = {}
    for k, v in extras.items():
        if k == "extras":
            continue
        if v in (None, "", "default"):
            continue
        sanitized[k] = v

    params = {
        "symbol": symbol,
        "resolution": resolution,
        "indicator": indicator,
        "timeperiod": timeperiod,
    }
    params.update(sanitized)

    return JSONResponse(await fh_get("indicator", params))

@app.get("/fh/profile")
async def fh_profile(symbol: str = Query(..., min_length=1, max_length=20)):
    """
    Company profile (Finnhub /stock/profile2)
    """
    return JSONResponse(await fh_get("stock/profile2", {"symbol": symbol}))

@app.get("/fh/news")
async def fh_news(
    symbol: str = Query(..., min_length=1, max_length=20),
    _from: str = Query(..., alias="from", description="YYYY-MM-DD"),
    to: str = Query(..., description="YYYY-MM-DD"),
    limit: int = Query(50, ge=1, le=500, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Items to skip before returning"),
    compact: bool = Query(True, description="If true, proxy trims to essential fields"),
    fields: str = Query("", description="CSV of fields to keep; overrides 'compact'")
):
    """
    Company news (Finnhub /company-news) with server-side pagination/compaction
    to avoid oversized responses to the Action caller.
    """
    data = await fh_get("company-news", {"symbol": symbol, "from": _from, "to": to})
    items = data if isinstance(data, list) else []

    # slice
    start = min(offset, max(0, len(items)))
    end = min(start + limit, len(items))
    page = items[start:end]

    # field filter
    if fields:
        keep = {f.strip() for f in fields.split(",") if f.strip()}
        def pick(d): return {k: v for k, v in d.items() if k in keep}
        page = [pick(x) for x in page]
    elif compact:
        keep = {"datetime","headline","source","url","summary","image","category","id","related"}
        def pick(d): return {k: v for k, v in d.items() if k in keep}
        page = [pick(x) for x in page]

    return JSONResponse({
        "symbol": symbol,
        "from": _from,
        "to": to,
        "offset": start,
        "limit": limit,
        "returned": len(page),
        "total_estimate": len(items),
        "items": page
    })
