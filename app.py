# app.py
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import httpx

TD_BASE = "https://api.twelvedata.com"
TD_KEY = os.getenv("TWELVE_KEY")  # set in your host env (Render dashboard)

app = FastAPI(title="TwelveData Proxy")

@app.get("/")
def root():
    return {"ok": True, "service": "twelve-proxy", "paths": ["/quote", "/time_series", "/rsi", "/macd"]}

def ensure_key():
    if not TD_KEY:
        raise HTTPException(status_code=500, detail="TWELVE_KEY not configured on server")

async def forward(path: str, params: dict):
    ensure_key()
    params = {k: v for k, v in params.items() if v is not None}
    params["apikey"] = TD_KEY
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{TD_BASE}/{path}", params=params)
    if r.status_code >= 400:
        # Bubble up TD error payload to help debugging
        try:
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception:
            raise HTTPException(status_code=r.status_code, detail=r.text)
    return JSONResponse(r.json())

@app.get("/quote")
async def quote(symbol: str = Query(...)):
    return await forward("quote", {"symbol": symbol})

@app.get("/time_series")
async def time_series(
    symbol: str = Query(...),
    interval: str = Query("1day"),
    outputsize: int = Query(100),
    order: str = Query("desc")
):
    return await forward("time_series", {
        "symbol": symbol, "interval": interval, "outputsize": outputsize, "order": order
    })

@app.get("/rsi")
async def rsi(
    symbol: str = Query(...),
    interval: str = Query("1day"),
    time_period: int = Query(14),
    outputsize: int = Query(120)
):
    return await forward("rsi", {
        "symbol": symbol, "interval": interval, "time_period": time_period, "outputsize": outputsize
    })

@app.get("/macd")
async def macd(
    symbol: str = Query(...),
    interval: str = Query("1day"),
    fastperiod: int = Query(12),
    slowperiod: int = Query(26),
    signalperiod: int = Query(9),
    outputsize: int = Query(120)
):
    return await forward("macd", {
        "symbol": symbol,
        "interval": interval,
        "fastperiod": fastperiod,
        "slowperiod": slowperiod,
        "signalperiod": signalperiod,
        "outputsize": outputsize
    })
