# app_fh.py
import os, time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

FH_BASE = "https://finnhub.io/api/v1"
FH_KEY = os.getenv("FINNHUB_KEY")
REQUEST_TIMEOUT_SECS = 30
MAX_RETRIES = 3
INITIAL_BACKOFF = 0.75
CACHE_TTL_SECONDS = 10

app = FastAPI(title="Finnhub Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=[],
)

_cache: Dict[str, Dict[str, Any]] = {}
def _ck(path: str, params: Dict[str, Any]) -> str:
    items = sorted((k, str(v)) for k, v in params.items() if v is not None)
    return f"{path}|{tuple(items)}"
def _get(k: str) -> Optional[Any]:
    e = _cache.get(k)
    if not e: return None
    if time.time() - e["ts"] > CACHE_TTL_SECONDS:
        _cache.pop(k, None); return None
    return e["data"]
def _set(k: str, data: Any):
    _cache[k] = {"ts": time.time(), "data": data}

async def fh_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not FH_KEY:
        raise HTTPException(status_code=500, detail="FINNHUB_KEY env var not configured on server")
    q = {k: v for k, v in params.items() if v is not None}
    q["token"] = FH_KEY
    key = _ck(path, q)
    cached = _get(key)
    if cached is not None: return cached

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

@app.get("/")
def root(): return {"ok": True, "service": "finnhub-proxy", "paths": ["/fh/quote","/fh/candles","/fh/indicator","/fh/profile","/fh/news"]}

@app.get("/healthz")
def healthz(): return {"ok": True, "finnhub_key_configured": bool(FH_KEY)}

# -------- Finnhub routes --------
@app.get("/fh/quote")
async def fh_quote(symbol: str = Query(..., min_length=1, max_length=20)):
    return JSONResponse(await fh_get("quote", {"symbol": symbol}))

@app.get("/fh/candles")
async def fh_candles(
    symbol: str = Query(..., min_length=1, max_length=20),
    resolution: str = Query("D", description="1,5,15,30,60,D,W,M"),
    _from: int = Query(..., description="UNIX seconds"),
    to: int = Query(..., description="UNIX seconds"),
):
    return JSONResponse(await fh_get("stock/candle", {
        "symbol": symbol, "resolution": resolution, "from": _from, "to": to
    }))

@app.get("/fh/indicator")
async def fh_indicator(
    symbol: str = Query(..., min_length=1, max_length=20),
    resolution: str = Query("D"),
    indicator: str = Query(..., description="rsi, macd, ema, sma, adx, mfi, stoch, bbands, atr"),
    timeperiod: int = Query(14),
    **extras
):
    params = {"symbol": symbol, "resolution": resolution, "indicator": indicator, "timeperiod": timeperiod}
    params.update({k: v for k, v in extras.items() if v is not None})
    return JSONResponse(await fh_get("indicator", params))

@app.get("/fh/profile")
async def fh_profile(symbol: str = Query(..., min_length=1, max_length=20)):
    return JSONResponse(await fh_get("stock/profile2", {"symbol": symbol}))

@app.get("/fh/news")
async def fh_news(
    symbol: str = Query(..., min_length=1, max_length=20),
    _from: str = Query(..., description="YYYY-MM-DD"),
    to: str = Query(..., description="YYYY-MM-DD")
):
    return JSONResponse(await fh_get("company-news", {"symbol": symbol, "from": _from, "to": to}))
