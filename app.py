from fastapi import FastAPI, Response
import os, requests

app = FastAPI()  # <= LOWERCASE 'app' to match uvicorn app:app

API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")

@app.get("/")
def root():
    return {"status": "ok", "service": "market-data-proxy"}

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.get("/quote/{symbol}")
def quote(symbol: str):
    if not API_KEY:
        return {"error": "Missing TWELVE_DATA_API_KEY"}
    url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={API_KEY}"
    r = requests.get(url, timeout=8)
    return r.json()

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "source": "twelve_data_proxy"}
