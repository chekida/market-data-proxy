from fastapi import FastAPI, Response
import os, requests

APP = FastAPI()

API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")  # set this in Render → Environment

@APP.get("/")
def root():
    # Render health check will also send HEAD here — FastAPI auto-handles HEAD if GET exists
    return {"status": "ok", "service": "market-data-proxy"}

@APP.head("/")
def root_head():
    # Explicit 200 for HEAD just to be safe
    return Response(status_code=200)

@APP.get("/quote/{symbol}")
def quote(symbol: str):
    if not API_KEY:
        return {"error": "Missing TWELVE_DATA_API_KEY"}
    url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={API_KEY}"
    r = requests.get(url, timeout=8)
    return r.json()

@APP.get("/healthcheck")
def healthcheck():
    # Lightweight probe so you can test in Actions/GPT
    return {"status": "ok", "source": "twelve_data_proxy"}




