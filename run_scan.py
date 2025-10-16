# run_scan.py
# ------------------------------------------------------------
# Automated market scanner for Render cron jobs.
# Twelve Data for price data, Finnhub for news,
# Discord webhook for alerts, and Google Sheets for logs.

import argparse
import datetime as dt
import requests
import os
import json

# === ENVIRONMENT VARIABLES ===
TWELVE_DATA_API_KEY = os.getenv("TWELVE_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # Discord webhook
SHEETS_WEBAPP_URL = os.getenv("SHEETS_WEBAPP_URL")  # optional Google Apps Script endpoint

TWELVE_URL = "https://api.twelvedata.com/time_series"
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"

DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "TSLA", "META", "AMD",
    "AVGO", "GOOG", "AMZN", "NFLX", "SMCI", "CRM", "INTC", "NXPI", "SNOW", "SHOP", "TTD"
]

# === HELPER FUNCTIONS ===
def fetch_data(symbol, interval="1day", outputsize=200):
    """Fetch OHLCV data from Twelve Data."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": TWELVE_DATA_API_KEY,
        "outputsize": outputsize
    }
    try:
        r = requests.get(TWELVE_URL, params=params, timeout=10)
        data = r.json()
        if "values" not in data:
            print(f"[{symbol}] API error: {data}", flush=True)
            return []
        return data["values"]
    except Exception as e:
        print(f"[{symbol}] Fetch error: {e}", flush=True)
        return []

def compute_sma(data, n):
    closes = [float(p["close"]) for p in data[:n] if "close" in p]
    return sum(closes) / len(closes) if closes else None

def compute_atr(data, n=14):
    trs = []
    for i in range(n):
        try:
            h, l, c_prev = float(data[i]["high"]), float(data[i]["low"]), float(data[i+1]["close"])
            trs.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
        except:
            continue
    return sum(trs) / len(trs) if trs else None

def analyze_signal(symbol):
    data = fetch_data(symbol)
    if len(data) < 200:
        return None

    sma50 = compute_sma(data, 50)
    sma200 = compute_sma(data, 200)
    atr = compute_atr(data)
    close = float(data[0]["close"])

    atrpct = (atr / close) * 100 if atr else 0
    signal = "None"
    if close > sma50 * 1.02:
        signal = "Breakout"
    elif close > sma50 and close < sma200 * 1.05:
        signal = "Inflection"

    return {
        "symbol": symbol,
        "close": round(close, 2),
        "SMA50": round(sma50, 2),
        "SMA200": round(sma200, 2),
        "ATRpct": round(atrpct, 2),
        "signal": signal
    }

# === DISCORD WEBHOOK ===
def post_to_webhook(task_name, payload):
    """Send scan results to Discord."""
    if not WEBHOOK_URL:
        print("[Webhook] No WEBHOOK_URL found — skipping send.", flush=True)
        return
    try:
        short_payload = payload[:5] if isinstance(payload, list) else payload
        msg = f"📊 **{task_name.capitalize()} Scan Results**\n```json\n{json.dumps(short_payload, indent=2)}```"
        r = requests.post(WEBHOOK_URL, json={"content": msg}, timeout=10)
        if r.status_code in (200, 204):
            print(f"[Webhook] ✅ Sent {task_name} results successfully.", flush=True)
        else:
            print(f"[Webhook] ⚠️ Failed ({r.status_code}): {r.text}", flush=True)
    except Exception as e:
        print(f"[Webhook] ❌ Error sending message: {e}", flush=True)

# === GOOGLE SHEETS LOGGER (OPTIONAL) ===
def log_to_google_sheets(task_name, payload):
    """Push JSON payload to a Google Sheets Apps Script endpoint."""
    if not SHEETS_WEBAPP_URL:
        print("[Sheets] No SHEETS_WEBAPP_URL found — skipping log.", flush=True)
        return
    try:
        r = requests.post(SHEETS_WEBAPP_URL, json={"task": task_name, "data": payload}, timeout=10)
        if r.status_code == 200:
            print("[Sheets] ✅ Logged successfully.", flush=True)
        else:
            print(f"[Sheets] ⚠️ Failed to log: {r.status_code} {r.text}", flush=True)
    except Exception as e:
        print(f"[Sheets] ❌ Error logging: {e}", flush=True)

# === TASKS ===
def task_premarket():
    print("[Premarket] Checking market regime...", flush=True)
    result = {s: analyze_signal(s) for s in ["SPY", "QQQ", "IWM"]}
    print(json.dumps(result, indent=2), flush=True)
    post_to_webhook("premarket", result)
    log_to_google_sheets("premarket", result)
    return result

def task_breakout():
    print("[Breakout] Running breakout scan...", flush=True)
    signals = [analyze_signal(s) for s in DEFAULT_SYMBOLS]
    signals = [s for s in signals if s and s["signal"] == "Breakout"]
    print(json.dumps(signals, indent=2), flush=True)
    post_to_webhook("breakout", signals)
    log_to_google_sheets("breakout", signals)
    return signals

def task_inflection():
    print("[Inflection] Identifying trend inflections...", flush=True)
    signals = [analyze_signal(s) for s in DEFAULT_SYMBOLS]
    signals = [s for s in signals if s and s["signal"] == "Inflection"]
    print(json.dumps(signals, indent=2), flush=True)
    post_to_webhook("inflection", signals)
    log_to_google_sheets("inflection", signals)
    return signals

def task_midday():
    print("[Midday] Running intraday refresh...", flush=True)
    signals = []
    for s in DEFAULT_SYMBOLS:
        result = analyze_signal(s)
        if not result:
            continue
        if result["signal"] in ("Breakout", "Inflection"):
            signals.append(result)
    print(json.dumps(signals, indent=2), flush=True)
    post_to_webhook("midday", signals)
    log_to_google_sheets("midday", signals)
    return signals

def task_closing():
    print("[Closing] Generating RS leaders...", flush=True)
    signals = [analyze_signal(s) for s in DEFAULT_SYMBOLS]
    leaders = sorted([s for s in signals if s], key=lambda x: -x["close"])[:3]
    print(json.dumps(leaders, indent=2), flush=True)
    post_to_webhook("closing", leaders)
    log_to_google_sheets("closing", leaders)
    return leaders

def task_news(symbols=DEFAULT_SYMBOLS[:5]):
    print("[News] Fetching headlines via Finnhub...", flush=True)
    today = dt.date.today()
    from_date = today - dt.timedelta(days=1)
    news_report = {}
    for s in symbols:
        try:
            url = f"{FINNHUB_URL}?symbol={s}&from={from_date}&to={today}&token={FINNHUB_API_KEY}"
            r = requests.get(url, timeout=10)
            stories = r.json()[:3]
            news_report[s] = [{"headline": n.get("headline"), "source": n.get("source")} for n in stories]
        except Exception as e:
            news_report[s] = f"Error fetching news: {e}"
    print(json.dumps(news_report, indent=2), flush=True)
    post_to_webhook("news", news_report)
    log_to_google_sheets("news", news_report)
    return news_report

# === MAIN ROUTER ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Twelve Data + Finnhub scan task.")
    parser.add_argument("--task", required=True, help="Task name: premarket, breakout, inflection, midday, closing, news")
    args = parser.parse_args()

    task = args.task.lower()
    print(f"\n=== Executing Task: {task} | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n", flush=True)

    if task in ["premarket", "regime"]:
        task_premarket()
    elif task in ["breakout", "top25", "refresh"]:
        task_breakout()
    elif task == "midday":
        task_midday()
    elif task == "inflection":
        task_inflection()
    elif task in ["closing", "atr", "overnight"]:
        task_closing()
    elif task == "news":
        task_news()
    else:
        print(f"Unknown task: {task}", flush=True)
