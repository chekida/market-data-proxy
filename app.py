# app.py
from fastapi import FastAPI, Request
import httpx, os

API = "https://api.twelvedata.com"
KEY = os.getenv("TWELVE_KEY", "bc8b8da525794eed9c37f475dafa17d2")
app = FastAPI()

@app.get("/{path:path}")
async def passthrough(path: str, request: Request):
    params = dict(request.query_params)
    params["apikey"] = KEY
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{API}/{path}", params=params, timeout=30)
    return r.json()


