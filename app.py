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
