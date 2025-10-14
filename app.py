# C:\td_proxy\app.py
import os
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

# --- load env key ---
load_dotenv(find_dotenv(), override=False)
TD_KEY = os.environ.get("TWELVEDATA_KEY")
BASE = "https://api.twelvedata.com"

# --- fastapi app ---
app = FastAPI(title="My Twelve Data Proxy", version="1.0.0")

@app.on_event("startup")
async def check_key():
    if TD_KEY:
        print("✔ TWELVEDATA_KEY loaded")
    else:
        print("✖ TWELVEDATA_KEY missing (set it in C:\\td_proxy\\.env or via setx)")

# --- models ---
class Quote(BaseModel):
    symbol: str
    price: float
    currency: str | None = None
    change: float | None = None
    percent_change: float | None = None
    timestamp: str

# --- helpers ---
def first_num_from(data: dict, *keys):
    for k in keys:
        v = data.get(k)
        if v not in (None, "", "null"):
            try:
                return float(v)
            except Exception:
                pass
    return None

# --- routes ---
@app.get("/quote/{symbol}", response_model=Quote)
async def quote(symbol: str):
    if not TD_KEY:
        raise HTTPException(status_code=500, detail="Server missing TWELVEDATA_KEY. Set it in .env and restart.")

    params = {"symbol": symbol, "apikey": TD_KEY}
    async with httpx.AsyncClient(timeout=15) as s:
        r = await s.get(f"{BASE}/quote", params=params)

    raw_text = r.text
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Upstream non-JSON: {raw_text[:200]}")

    # Bubble up Twelve Data error payloads
    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {data.get('message')}")

    # Normalize price fields
    price = first_num_from(data, "price", "close", "last", "previous_close")
    change = first_num_from(data, "change")
    percent_change = first_num_from(data, "percent_change")

    # Compute percent_change if missing and we can
    if percent_change is None and change is not None:
        prev = first_num_from(data, "previous_close")
        if prev:
            try:
                percent_change = (change / prev) * 100.0
            except Exception:
                percent_change = None

    if price is None:
        raise HTTPException(
            status_code=404,
            detail=f"No price-like field for {symbol}. Keys: {list(data.keys())[:20]}"
        )

    return Quote(
        symbol=symbol.upper(),
        price=price,
        currency=data.get("currency"),
        change=change,
        percent_change=percent_change,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

# Optional raw passthrough for debugging
@app.get("/raw/{symbol}")
async def raw(symbol: str):
    if not TD_KEY:
        raise HTTPException(status_code=500, detail="Server missing TWELVEDATA_KEY.")
    params = {"symbol": symbol, "apikey": TD_KEY}
    async with httpx.AsyncClient(timeout=15) as s:
        r = await s.get(f"{BASE}/quote", params=params)
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        return {"status_code": r.status_code, "json": r.json()}
    return {"status_code": r.status_code, "text": r.text[:500]}

from fastapi.responses import FileResponse

@app.get("/openapi.yaml")
async def openapi_yaml():
    return FileResponse("openapi.yaml", media_type="text/yaml")
# app.py
import os, httpx
from datetime import datetime, timezone, date, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="Market Data + News Proxy", version="1.3.0")

TD_BASE = "https://api.twelvedata.com"
TD_KEY = os.getenv("TWELVEDATA_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
FINNHUB_COMPANY_NEWS = "https://finnhub.io/api/v1/company-news"

def need_key(env, name):
    if not env:
        raise HTTPException(status_code=500, detail=f"Missing {name} environment variable")

@app.get("/")           # optional
def root(): return {"ok": True, "docs": "/openapi.yaml"}

@app.get("/openapi.yaml")
def serve_openapi():
    path = os.path.join(os.path.dirname(_file_), "openapi.yaml")
    if not os.path.exists(path): raise HTTPException(404, "openapi.yaml not found")
    return FileResponse(path, media_type="text/yaml", filename="openapi.yaml")

# -------- QUOTE
@app.get("/quote/{symbol}")
async def quote_by_symbol(symbol: str):
    need_key(TD_KEY, "TWELVEDATA_KEY")
    async with httpx.AsyncClient(timeout=15) as s:
        r = await s.get(f"{TD_BASE}/quote", params={"symbol": symbol, "apikey": TD_KEY})
    try:
        data = r.json()
    except Exception:
        raise HTTPException(502, r.text[:200])
    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(502, f"Twelve Data error: {data.get('message')}")
    if isinstance(data, list) and data:
        data = data[0]
    price = data.get("price") or data.get("close")
    try: price = float(price) if price is not None else None
    except: price = None
    return {
        "symbol": data.get("symbol") or symbol.upper(),
        "name": data.get("name"),
        "price": price,
        "currency": data.get("currency"),
        "change": float(data.get("change")) if data.get("change") else None,
        "percent_change": float(data.get("percent_change")) if data.get("percent_change") else None,
        "volume": float(data.get("volume")) if data.get("volume") else None,
        "timestamp": data.get("datetime") or datetime.now(timezone.utc).isoformat(),
    }

# -------- SMA passthrough
@app.get("/sma")
async def sma(symbol: str, interval: str = "1day", time_period: int = 50, outputsize: int | None = None):
    need_key(TD_KEY, "TWELVEDATA_KEY")
    params = {"symbol": symbol, "interval": interval, "time_period": time_period, "apikey": TD_KEY}
    if outputsize: params["outputsize"] = outputsize
    async with httpx.AsyncClient(timeout=20) as s:
        r = await s.get(f"{TD_BASE}/sma", params=params)
    try: data = r.json()
    except Exception: raise HTTPException(502, r.text[:200])
    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(502, f"Twelve Data error: {data.get('message')}")
    return JSONResponse(data)

# -------- TIME SERIES passthrough
@app.get("/time_series")
async def time_series(symbol: str, interval: str = "1day", outputsize: int = 300, order: str = "desc"):
    need_key(TD_KEY, "TWELVEDATA_KEY")
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "order": order, "apikey": TD_KEY}
    async with httpx.AsyncClient(timeout=30) as s:
        r = await s.get(f"{TD_BASE}/time_series", params=params)
    try: data = r.json()
    except Exception: raise HTTPException(502, r.text[:200])
    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(502, f"Twelve Data error: {data.get('message')}")
    return JSONResponse(data)

# -------- FINNHUB company news (normalized)
@app.get("/news/company")
async def news_company(symbol: str = Query(..., description="e.g., AAPL"),
                       days: int = Query(3, ge=1, le=14)):
    need_key(FINNHUB_KEY, "FINNHUB_KEY")
    to_d, from_d = date.today(), date.today() - timedelta(days=days)
    params = {"symbol": symbol.upper(), "from": from_d.isoformat(), "to": to_d.isoformat(), "token": FINNHUB_KEY}
    async with httpx.AsyncClient(timeout=20) as s:
        r = await s.get(FINNHUB_COMPANY_NEWS, params=params)
    try: data = r.json()
    except Exception: raise HTTPException(502, r.text[:200])
    if not isinstance(data, list):
        msg = (isinstance(data, dict) and data.get("error")) or str(data)[:200]
        raise HTTPException(502, f"Finnhub error: {msg}")
    out = []
    for it in data:
        out.append({
            "ticker": symbol.upper(),
            "headline": it.get("headline"),
            "source": it.get("source"),
            "url": it.get("url"),
            "published_at": it.get("datetime"),  # epoch seconds
            "summary": it.get("summary"),
        })
    return out

