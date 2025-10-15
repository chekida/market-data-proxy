# app.py — Market Data Proxy (Twelve Data only)
# Endpoints:
#   GET /health
#   GET /quote/{symbol}
#   GET /sma?symbol=...&interval=1day&time_period=50[&outputsize=300]
#   GET /time_series?symbol=...&interval=1day&outputsize=300&order=desc
#   GET /oas.min.json  (OpenAPI 3.1.1 for easy Actions import)

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import httpx

app = FastAPI(title="Market Data Proxy", version="1.0.0")

# Root + health (safe, minimal)
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def root():
    return {"ok": True, "service": "market-data-proxy"}

@app.get("/health", include_in_schema=False)
def health():
    return {"status":"ok"}

TD_BASE = "https://api.twelvedata.com"
TD_KEY = os.getenv("TWELVEDATA_KEY")


# ---------- helpers ----------
def require_key():
    if not TD_KEY:
        raise HTTPException(status_code=500, detail="Missing TWELVEDATA_KEY environment variable")


def as_float(x):
    try:
        return float(x)
    except Exception:
        return None


# ---------- health ----------
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


# ---------- quote ----------
@app.get("/quote/{symbol}")
async def quote_by_symbol(symbol: str):
    """Return a normalized Twelve Data quote."""
    require_key()
    params = {"symbol": symbol.upper(), "apikey": TD_KEY}
    async with httpx.AsyncClient(timeout=15) as s:
        r = await s.get(f"{TD_BASE}/quote", params=params)

    # prefer JSON; if not JSON, bubble first 200 chars
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=r.text[:200])

    # Twelve Data may return a dict with status=error, or a list
    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {data.get('message')}")
    if isinstance(data, list) and data:
        data = data[0]

    price = as_float(data.get("price") or data.get("close"))
    return {
        "symbol": (data.get("symbol") or symbol).upper(),
        "name": data.get("name"),
        "price": price,
        "currency": data.get("currency"),
        "change": as_float(data.get("change")),
        "percent_change": as_float(data.get("percent_change")),
        "volume": as_float(data.get("volume")),
        "timestamp": data.get("datetime"),
        "raw": data,  # keep raw for debugging/client flexibility
    }


# ---------- SMA passthrough ----------
@app.get("/sma")
async def sma(
    symbol: str = Query(..., description="Ticker, e.g., AAPL"),
    interval: str = Query("1day"),
    time_period: int = Query(50, ge=1),
    outputsize: int | None = Query(None, ge=1),
):
    """Thin proxy to Twelve Data SMA; returns their JSON unmodified."""
    require_key()
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "time_period": time_period,
        "apikey": TD_KEY,
    }
    if outputsize:
        params["outputsize"] = outputsize

    async with httpx.AsyncClient(timeout=20) as s:
        r = await s.get(f"{TD_BASE}/sma", params=params)
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=r.text[:200])

    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {data.get('message')}")
    return JSONResponse(data)


# ---------- time series passthrough ----------
@app.get("/time_series")
async def time_series(
    symbol: str = Query(..., description="Ticker, e.g., AAPL"),
    interval: str = Query("1day"),
    outputsize: int = Query(300, ge=1),
    order: str = Query("desc"),
):
    """Thin proxy to Twelve Data time_series; returns their JSON unmodified."""
    require_key()
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "outputsize": outputsize,
        "order": order,
        "apikey": TD_KEY,
    }
    async with httpx.AsyncClient(timeout=30) as s:
        r = await s.get(f"{TD_BASE}/time_series", params=params)
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=r.text[:200])

    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {data.get('message')}")
    return JSONResponse(data)


# ---------- tiny OpenAPI (3.1.1) for Actions import ----------
@app.get("/oas.min.json", include_in_schema=False)
def oas_min_json():
    return JSONResponse({
        "openapi": "3.1.1",
        "info": {"title": "Market Data Proxy", "version": "1.0.0"},
        "servers": [{"url": "https://market-data-proxy.onrender.com/"}],
        "paths": {
            "/health": {
                "get": {
                    "operationId": "getHealth",
                    "summary": "Health check",
                    "responses": {"200": {"description": "OK"}}
                }
            },
            "/quote/{symbol}": {
                "get": {
                    "operationId": "quoteBySymbol",
                    "summary": "Get live quote by symbol",
                    "parameters": [
                        {"name": "symbol", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {"200": {"description": "OK"}}
                }
            },
            "/sma": {
                "get": {
                    "operationId": "getSMA",
                    "summary": "Simple Moving Average",
                    "parameters": [
                        {"name": "symbol", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "interval", "in": "query", "schema": {"type": "string", "default": "1day"}},
                        {"name": "time_period", "in": "query", "required": True, "schema": {"type": "integer", "default": 50}},
                        {"name": "outputsize", "in": "query", "schema": {"type": "integer"}}
                    ],
                    "responses": {"200": {"description": "OK"}}
                }
            },
            "/time_series": {
                "get": {
                    "operationId": "getTimeSeries",
                    "summary": "OHLCV Time Series",
                    "parameters": [
                        {"name": "symbol", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "interval", "in": "query", "schema": {"type": "string", "default": "1day"}},
                        {"name": "outputsize", "in": "query", "schema": {"type": "integer", "default": 300}},
                        {"name": "order", "in": "query", "schema": {"type": "string", "default": "desc"}}
                    ],
                    "responses": {"200": {"description": "OK"}}
                }
            }
        },
        "components": {"schemas": {}}
    })


# (Optional) local dev entrypoint — Render uses the Start Command instead
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","10000")))




