# app.py — Market Data + News Proxy (cleaned)
# FastAPI app providing:
# - Health checks (/health and /getHealth)
# - Twelve Data passthrough: /quote/{symbol}, /sma, /time_series
# - Finnhub company news: /news/company
# - Combined summary: /combined/summary
# - Minimal OpenAPI JSON for GPT Actions import: /oas.min.json
# - (Optional) Serve a static openapi.yaml from your repo root

import os
from datetime import datetime, timezone, date, timedelta
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse

# -------------------- Config & App --------------------

app = FastAPI(title="Market Data + News Proxy", version="1.3.2")

TD_BASE = "https://api.twelvedata.com"
TD_KEY = os.getenv("TWELVEDATA_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
FINNHUB_COMPANY_NEWS = "https://finnhub.io/api/v1/company-news"
SPY_SYMBOL = "SPY"  # benchmark for relative strength

def need_key(env_value: str | None, name: str) -> None:
    if not env_value:
        raise HTTPException(status_code=500, detail=f"Missing {name} environment variable")

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

# -------------------- Health & Root --------------------

@app.get("/health", include_in_schema=False)
@app.get("/getHealth", include_in_schema=False)  # alias for picky clients
def health_check():
    return JSONResponse({"status": "ok"})

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    """Simple liveness endpoint."""
    return {"ok": True, "docs": "/openapi.yaml", "oas": "/oas.min.json"}

# -------------------- Minimal OpenAPI (for GPT Actions import) --------------------

@app.get("/oas.min.json", include_in_schema=False)
def oas_min_json():
    # Ultra-minimal OpenAPI 3.1.1 with a valid servers URL and one path
    return JSONResponse({
        "openapi": "3.1.1",
        "info": {"title": "MD Proxy MIN", "version": "0.0.3"},
        "servers": [{"url": "https://market-data-proxy.onrender.com/"}],
        "paths": {
            "/health": {
                "get": {
                    "operationId": "getHealth",
                    "summary": "Health check",
                    "responses": {
                        "200": {
                            "description": "OK",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"status": {"type": "string"}},
                                        "required": ["status"]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {"schemas": {}}
    })

# -------------------- (Optional) Serve a local openapi.yaml --------------------

@app.get("/openapi.yaml")
def serve_openapi():
    path = os.path.join(os.path.dirname(_file_), "openapi.yaml")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="openapi.yaml not found")
    return FileResponse(path, media_type="text/yaml", filename="openapi.yaml")

# -------------------- Twelve Data: Quote --------------------

@app.get("/quote/{symbol}")
async def quote_by_symbol(symbol: str):
    need_key(TD_KEY, "TWELVEDATA_KEY")
    async with httpx.AsyncClient(timeout=15) as s:
        r = await s.get(f"{TD_BASE}/quote", params={"symbol": symbol, "apikey": TD_KEY})
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=r.text[:200])

    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {data.get('message')}")
    if isinstance(data, list) and data:
        data = data[0]

    price = data.get("price") or data.get("close")
    try:
        price = float(price) if price is not None else None
    except Exception:
        price = None

    return {
        "symbol": (data.get("symbol") or symbol).upper(),
        "name": data.get("name"),
        "price": price,
        "currency": data.get("currency"),
        "change": float(data.get("change")) if data.get("change") else None,
        "percent_change": float(data.get("percent_change")) if data.get("percent_change") else None,
        "volume": float(data.get("volume")) if data.get("volume") else None,
        "timestamp": data.get("datetime") or datetime.now(timezone.utc).isoformat(),
    }

# -------------------- Twelve Data: SMA passthrough --------------------

@app.get("/sma")
async def sma(symbol: str, interval: str = "1day", time_period: int = 50, outputsize: int | None = None):
    need_key(TD_KEY, "TWELVEDATA_KEY")
    params = {"symbol": symbol, "interval": interval, "time_period": time_period, "apikey": TD_KEY}
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

# -------------------- Twelve Data: Time Series passthrough --------------------

@app.get("/time_series")
async def time_series(symbol: str, interval: str = "1day", outputsize: int = 300, order: str = "desc"):
    need_key(TD_KEY, "TWELVEDATA_KEY")
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "order": order, "apikey": TD_KEY}
    async with httpx.AsyncClient(timeout=30) as s:
        r = await s.get(f"{TD_BASE}/time_series", params=params)
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=r.text[:200])
    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {data.get('message')}")
    return JSONResponse(data)

# -------------------- Finnhub: Company News --------------------

def _news_item(headline, source, url, published_at, summary=None, ticker=None):
    return {
        "ticker": ticker,
        "headline": headline,
        "source": source,
        "url": url,
        "published_at": published_at,  # epoch seconds from Finnhub
        "summary": summary,
    }

@app.get("/news/company")
async def get_company_news(
    symbol: str = Query(..., description="Ticker, e.g., AAPL"),
    days: int = Query(3, ge=1, le=14, description="Lookback (1–14 days)")
):
    """Returns recent company headlines from Finnhub in a normalized format."""
    need_key(FINNHUB_KEY, "FINNHUB_KEY")
    to_d = date.today()
    from_d = to_d - timedelta(days=days)
    params = {"symbol": symbol.upper(), "from": from_d.isoformat(), "to": to_d.isoformat(), "token": FINNHUB_KEY}

    async with httpx.AsyncClient(timeout=20) as s:
        r = await s.get(FINNHUB_COMPANY_NEWS, params=params)

    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=r.text[:200])
    if not isinstance(data, list):
        msg = (isinstance(data, dict) and data.get("error")) or "Unexpected Finnhub response"
        raise HTTPException(status_code=502, detail=f"Finnhub error: {msg}")

    out = []
    for it in data[:25]:
        out.append(_news_item(
            headline=it.get("headline"),
            source=it.get("source"),
            url=it.get("url"),
            published_at=it.get("datetime"),
            summary=it.get("summary"),
            ticker=symbol.upper(),
        ))
    return out

# -------------------- Helpers for Combined Summary --------------------

def _rolling_sma(vals, window):
    if len(vals) < window:
        return None
    return sum(vals[-window:]) / window

def _atr14_from_ohlc(values):
    """
    values: list of dicts from Twelve Data time_series (can be asc/desc).
    Returns (atr, atrpct). Needs at least 15 rows.
    """
    if not values or len(values) < 15:
        return (None, None)
    try:
        ordered = sorted(values, key=lambda v: v["datetime"])  # oldest -> newest
    except Exception:
        ordered = values[:]
    rows = []
    for v in ordered:
        o = _to_float(v.get("open"))
        h = _to_float(v.get("high"))
        l = _to_float(v.get("low"))
        c = _to_float(v.get("close"))
        if None in (o, h, l, c):
            continue
        rows.append((o, h, l, c))
    if len(rows) < 15:
        return (None, None)
    trs = []
    prev_close = rows[0][3]
    for (_, h, l, c) in rows[1:]:
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if len(trs) < 14:
        return (None, None)
    atr = sum(trs[-14:]) / 14.0
    last_close = rows[-1][3]
    atrpct = 100.0 * atr / last_close if last_close else None
    return (atr, atrpct)

def _fifty_two_week_stats(values):
    """Return (hi, lo, dist_to_hi_pct) using last ~252 closes if available."""
    closes = []
    for v in values:
        c = _to_float(v.get("close"))
        if c is not None:
            closes.append(c)
    if not closes:
        return (None, None, None)
    # Twelve Data returns most recent first
    window = closes[:252] if len(closes) >= 252 else closes
    hi = max(window)
    lo = min(window)
    last = closes[0]
    dist = None if not hi or not last else 100.0 * (hi - last) / hi
    return (hi, lo, dist)

def _pct_change(a, b):
    if a is None or b is None or b == 0:
        return None
    return 100.0 * (a - b) / b

def _rs_scores_vs_spy(values_sym, values_spy):
    """Approx RS_1M & RS_3M using close-to-close % change vs SPY over ~21/63 trading days."""
    def closes(lst):
        out = []
        for v in lst:
            c = _to_float(v.get("close"))
            if c is not None:
                out.append(c)
        return out

    ct = closes(values_sym)
    cs = closes(values_spy)
    if len(ct) < 64 or len(cs) < 64:
        return (None, None)

    def ret_n(lst, n):
        try:
            return _pct_change(lst[0], lst[n])
        except Exception:
            return None

    r1_t, r3_t = ret_n(ct, 21), ret_n(ct, 63)
    r1_s, r3_s = ret_n(cs, 21), ret_n(cs, 63)
    rs1 = (r1_t - r1_s) if (r1_t is not None and r1_s is not None) else None
    rs3 = (r3_t - r3_s) if (r3_t is not None and r3_s is not None) else None
    return (rs1, rs3)

# -------------------- Combined Summary --------------------

@app.get("/combined/summary")
async def combined_summary(symbol: str, interval: str = "1day", outputsize: int = 300):
    need_key(TD_KEY, "TWELVEDATA_KEY")

    # 1) Quote
    async with httpx.AsyncClient(timeout=15) as s:
        qr = await s.get(f"{TD_BASE}/quote", params={"symbol": symbol, "apikey": TD_KEY})
    try:
        q = qr.json()
    except Exception:
        raise HTTPException(status_code=502, detail=qr.text[:200])
    if isinstance(q, dict) and q.get("status") == "error":
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {q.get('message')}")
    if isinstance(q, list) and q:
        q = q[0]
    last_price = _to_float(q.get("price") or q.get("close"))
    ts_iso = q.get("datetime")

    # 2) Time series for symbol and SPY
    params = {"interval": interval, "outputsize": outputsize, "order": "desc", "apikey": TD_KEY}
    async with httpx.AsyncClient(timeout=30) as s:
        rs_sym = await s.get(f"{TD_BASE}/time_series", params={"symbol": symbol, **params})
        rs_spy = await s.get(f"{TD_BASE}/time_series", params={"symbol": SPY_SYMBOL, **params})
    try:
        ts_sym = rs_sym.json()
        ts_spy = rs_spy.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Bad time_series payload")
    if isinstance(ts_sym, dict) and ts_sym.get("status") == "error":
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {ts_sym.get('message')}")
    if isinstance(ts_spy, dict) and ts_spy.get("status") == "error":
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {ts_spy.get('message')}")

    values_sym = (ts_sym or {}).get("values") or []
    values_spy = (ts_spy or {}).get("values") or []

    # 3) SMA50/200
    closes = []
    for v in values_sym:
        c = _to_float(v.get("close"))
        if c is not None:
            closes.append(c)
    sma50 = _rolling_sma(closes, 50)
    sma200 = _rolling_sma(closes, 200)

    # 4) ATR14 and 52-week stats
    atr, atrpct = _atr14_from_ohlc(values_sym)
    hi, lo, dist_pct = _fifty_two_week_stats(values_sym)

    # 5) RS vs SPY
    rs1, rs3 = _rs_scores_vs_spy(values_sym, values_spy)

    # 6) Recent company news (best-effort)
    news_out = []
    if FINNHUB_KEY:
        try:
            to_d = date.today()
            from_d = to_d - timedelta(days=3)
            async with httpx.AsyncClient(timeout=20) as s:
                nr = await s.get(
                    FINNHUB_COMPANY_NEWS,
                    params={
                        "symbol": symbol.upper(),
                        "from": from_d.isoformat(),
                        "to": to_d.isoformat(),
                        "token": FINNHUB_KEY,
                    },
                )
            nn = nr.json()
            if isinstance(nn, list):
                for it in nn[:5]:
                    news_out.append({
                        "ticker": symbol.upper(),
                        "headline": it.get("headline"),
                        "source": it.get("source"),
                        "url": it.get("url"),
                        "published_at": it.get("datetime"),
                        "summary": it.get("summary"),
                    })
        except Exception:
            news_out = []

    return {
        "symbol": (q.get("symbol") or symbol).upper(),
        "price": last_price,
        "timestamp": ts_iso,
        "sma50": sma50,
        "sma200": sma200,
        "atr14": atr,
        "atrpct": atrpct,
        "fiftyTwoWeekHigh": hi,
        "fiftyTwoWeekLow": lo,
        "distTo52wHighPct": dist_pct,
        "rs_1m_vs_spy": rs1,
        "rs_3m_vs_spy": rs3,
        "news": news_out,
        "note": "Computed in-proxy. RS uses ~21/63 trading day differentials vs SPY."
    }



