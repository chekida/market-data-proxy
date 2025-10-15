# app.py
import os
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

TD_BASE = "https://api.twelvedata.com"
TD_KEY = os.getenv("TWELVE_KEY")

REQUEST_TIMEOUT_SECS = 30
MAX_RETRIES = 3
INITIAL_BACKOFF = 0.75  # seconds
CACHE_TTL_SECONDS = 10  # small TTL to ease rate limits

app = FastAPI(title="TwelveData Proxy", version="1.2.0")

# Optional: lock down CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # keep empty (no browser access needed). Add origins if you must.
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=[],
)

# ----------- simple in-memory TTL cache -----------
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

# ----------- helpers -----------
def ensure_key():
    if not TD_KEY:
        raise HTTPException(status_code=500, detail="TWELVE_KEY env var not configured on server")

async def forward(path: str, params: Dict[str, Any]) -> JSONResponse:
    """Forward to Twelve Data with retries, timeouts, and caching."""
    ensure_key()
    clean_params = {k: v for k, v in params.items() if v is not None}
    clean_params["apikey"] = TD_KEY

    ck = _cache_key(path, clean_params)
    cached = _cache_get(ck)
    if cached is not None:
        return JSONResponse(cached)

    backoff = INITIAL_BACKOFF
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECS) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = await client.get(f"{TD_BASE}/{path}", params=clean_params)
                # retry on rate-limit and server errors
                if r.status_code in (429, 500, 502, 503, 504):
                    if attempt < MAX_RETRIES:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                # bubble up error body if any error remains
                if r.status_code >= 400:
                    try:
                        return JSONResponse(status_code=r.status_code, content=r.json())
                    except Exception:
                        raise HTTPException(status_code=r.status_code, detail=r.text)
                data = r.json()
                _cache_set(ck, data)
                return JSONResponse(data)
            except httpx.RequestError as e:
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")

# ----------- health / root -----------
@app.get("/")
def root():
    return {"ok": True, "service": "twelve-proxy", "paths": ["/quote", "/time_series", "/rsi", "/macd", "/ema"]}

@app.get("/healthz")
def healthz():
    return {"ok": True, "td_key_configured": bool(TD_KEY)}

# ----------- routes -----------
@app.get("/quote")
async def quote(symbol: str = Query(..., min_length=1, max_length=20)):
    return await forward("quote", {"symbol": symbol})

@app.get("/time_series")
async def time_series(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    outputsize: int = Query(100, ge=1, le=5000),
    order: str = Query("desc")
):
    return await forward("time_series", {
        "symbol": symbol, "interval": interval, "outputsize": outputsize, "order": order
    })

@app.get("/rsi")
async def rsi(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    time_period: int = Query(14, ge=1, le=200),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return await forward("rsi", {
        "symbol": symbol, "interval": interval, "time_period": time_period, "outputsize": outputsize
    })

@app.get("/macd")
async def macd(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    fastperiod: int = Query(12, ge=1, le=200),
    slowperiod: int = Query(26, ge=1, le=300),
    signalperiod: int = Query(9, ge=1, le=200),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return await forward("macd", {
        "symbol": symbol,
        "interval": interval,
        "fastperiod": fastperiod,
        "slowperiod": slowperiod,
        "signalperiod": signalperiod,
        "outputsize": outputsize
    })

@app.get("/ema")
async def ema(
    symbol: str = Query(..., min_length=1, max_length=20),
    interval: str = Query("1day"),
    time_period: int = Query(20, ge=1, le=300),
    outputsize: int = Query(120, ge=1, le=5000)
):
    return await forward("ema", {
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "outputsize": outputsize
    })
