# run_scan.py
# ------------------------------------------------------------
# Automated market scanner for Render cron jobs.
# Uses Twelve Data for price data and Finnhub for news headlines.

import argparse
import datetime as dt
import requests
import os
import json

# === CONFIGURATION ===
TWELVE_KEY = os.getenv("TWELVE_DATA_API_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

TWELVE_URL = "https://api.twelvedata.com/time_series"
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"

DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "TSLA", "META", "AMD",
    "AVGO", "GOOG", "AMZN", "NFLX", "SMCI", "CRM", "INTC", "NXPI", "SNOW",
    "SHOP", "TTD"
]

# === HELPER FUNCTIONS ===
def fetch_data(symbol, interval="1day", outputsize=200):
    """Fetch OHLCV from Twelve Data."""
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

# === SIGNAL ANALYSIS ===
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

# === TASK HANDLERS ===
def task_premarket():
    print("[Premarket] Checking market regime...", flush=True)
    result = {s: analyze_signal(s) for s in ["SPY", "QQQ", "IWM"]}
    print(json.dumps(result, indent=2), flush=True)
    return result

def task_breakout():
    print("[Breakout] Running breakout scan...", flush=True)
    signals = [analyze_signal(s) for s in DEFAULT_SYMBOLS]
    signals = [s for s in signals if s and s["signal"] == "Breakout"]
    print(json.dumps(signals, indent=2), flush=True)
    with open("breakout_results.json", "w") as f:
        json.dump(signals, f, indent=2)
    print("[Breakout] Results saved to breakout_results.json", flush=True)
    return signals

def task_inflection():
    print("[Inflection] Identifying trend inflections...", flush=True)
    signals = [analyze_signal(s) for s in DEFAULT_SYMBOLS]
    signals = [s for s in signals if s and s["signal"] == "Inflection"]
    print(json.dumps(signals, indent=2), flush=True)
    with open("inflection_results.json", "w") as f:
        json.dump(signals, f, indent=2)
    print("[Inflection] Results saved to inflection_results.json", flush=True)
    return signals

def task_midday():
    """Midday recheck — confirms ongoing breakouts/inflections."""
    print("[Midday] Running intraday refresh on Breakout & Inflection signals...", flush=True)
    signals = []
    for s in DEFAULT_SYMBOLS:
        result = analyze_signal(s)
        if not result:
            continue
        if result["signal"] in ("Breakout", "Inflection"):
            signals.append(result)
    signals = sorted(signals, key=lambda x: -x["close"])[:10]
    print(json.dumps(signals, indent=2), flush=True)
    with open("midday_results.json", "w") as f:
        json.dump(signals, f, indent=2)
    print("[Midday] Results saved to midday_results.json", flush=True)
    return signals

def task_closing():
    print("[Closing] Generating RS leaders...", flush=True)
    signals = [analyze_signal(s) for s in DEFAULT_SYMBOLS]
    leaders = sorted([s for s in signals if s], key=lambda x: -x["close"])[:3]
    print(json.dumps(leaders, indent=2), flush=True)
    with open("closing_results.json", "w") as f:
        json.dump(leaders, f, indent=2)
    print("[Closing] Results saved to closing_results.json", flush=True)
    return leaders

# === FINNHUB NEWS ===
def task_news(symbols=DEFAULT_SYMBOLS[:5]):
    print("[News] Fetching latest headlines via Finnhub...", flush=True)
    today = dt.date.today()
    from_date = today - dt.timedelta(days=1)
    news_report = {}
    for s in symbols:
        try:
            url = f"{FINNHUB_URL}?symbol={s}&from={from_date}&to={today}&token={FINNHUB_API_KEY}"
            r = requests.get(url, timeout=10)
            stories = r.json()[:3]
            news_report[s] = [
                {"headline": n.get("headline"), "source": n.get("source"), "datetime": n.get("datetime")}
                for n in stories if "headline" in n
            ]
        except Exception as e:
            news_report[s] = f"Error fetching news: {e}"
    print(json.dumps(news_report, indent=2), flush=True)
    with open("news_results.json", "w") as f:
        json.dump(news_report, f, indent=2)
    print("[News] Results saved to news_results.json", flush=True)
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
