# run_scan.py
# ------------------------------------------------------------
# Automated market scanner for Render cron jobs.
# - Fetches dynamic S&P 500 + Nasdaq 100 tickers
# - Uses Twelve Data for price data and Finnhub for news
# - Sends formatted ChatGPT-style alerts to Discord
# ------------------------------------------------------------

import argparse
import datetime as dt
import requests
import os
import json
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

# === ENVIRONMENT VARIABLES ===
TWELVE_DATA_API_KEY = os.getenv("TWELVE_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# Optional rate limit config (defaults to Grow plan)
PLAN_LIMIT = int(os.getenv("TD_PLAN_LIMIT", 377))   # credits/min
SAFE_THREADS = max(1, min(10, int(PLAN_LIMIT / 60)))  # auto-scale threads

TWELVE_URL = "https://api.twelvedata.com/time_series"
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"

# === BUILD SYMBOL UNIVERSE ===
def get_tradable_symbols(limit=300):
    """Combine S&P 500 and Nasdaq 100 symbols (top liquid U.S. names)."""
    try:
        sp500 = pd.read_csv(
            "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        )["Symbol"].dropna().tolist()
    except Exception:
        sp500 = []

    try:
        nasdaq100 = pd.read_csv(
            "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.csv"
        )["Symbol"].dropna().tolist()
    except Exception:
        nasdaq100 = []

    universe = list(dict.fromkeys(sp500 + nasdaq100))
    if not universe:
        universe = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"]
    return universe[:limit]

DEFAULT_SYMBOLS = get_tradable_symbols()

# === DATA HELPERS ===
def fetch_data(symbol, interval="1day", outputsize=200):
    """Fetch OHLCV data from Twelve Data."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": TWELVE_DATA_API_KEY,
        "outputsize": outputsize,
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
        "signal": signal,
    }

# === FORMATTING HELPERS ===
def format_signal_table(signals):
    """Format results into a ChatGPT-style Markdown table."""
    if not signals:
        return "⚠️ No active signals."
    header = "| Ticker | Close | Signal | ATR% | Trend |\n|:--|--:|:--|:--:|:--|\n"
    rows = []
    for s in signals[:5]:
        trend = "🚀 Uptrend" if s["signal"] == "Breakout" else "↗️ Inflection"
        rows.append(f"| **{s['symbol']}** | {s['close']:.2f} | {s['signal']} | {s['ATRpct']:.2f}% | {trend} |")
    return header + "\n".join(rows)

# === DISCORD WEBHOOK ===
def post_to_webhook(task_name, payload):
    """Send scan results to Discord."""
    if not WEBHOOK_URL:
        print("[Webhook] No WEBHOOK_URL — skipping send.", flush=True)
        return
    try:
        msg = f"📊 **{task_name.capitalize()} Scan Results**\n\n{format_signal_table(payload)}"
        r = requests.post(WEBHOOK_URL, json={"content": msg}, timeout=10)
        if r.status_code in (200, 204):
            print(f"[Webhook] ✅ Sent {task_name} results.", flush=True)
        else:
            print(f"[Webhook] ⚠️ Failed ({r.status_code}): {r.text}", flush=True)
    except Exception as e:
        print(f"[Webhook] ❌ Error sending: {e}", flush=True)

# === TASKS ===
def run_parallel(func, symbols):
    with ThreadPoolExecutor(max_workers=SAFE_THREADS) as ex:
        results = list(ex.map(func, symbols))
    return results

def task_premarket():
    print("[Premarket] Checking regime...", flush=True)
    result = {s: analyze_signal(s) for s in ["SPY", "QQQ", "IWM"]}
    print(json.dumps(result, indent=2), flush=True)
    post_to_webhook("premarket", [v for v in result.values() if v])
    print("[Task] ✅ Premarket complete.", flush=True)
    return result

def task_breakout():
    print("[Breakout] Scanning breakout signals...", flush=True)
    results = run_parallel(analyze_signal, DEFAULT_SYMBOLS)
    signals = [s for s in results if s and s["signal"] == "Breakout"]
    print(json.dumps(signals, indent=2), flush=True)
    post_to_webhook("breakout", signals)
    print("[Task] ✅ Breakout complete.", flush=True)
    return signals

def task_inflection():
    print("[Inflection] Scanning trend inflections...", flush=True)
    results = run_parallel(analyze_signal, DEFAULT_SYMBOLS)
    signals = [s for s in results if s and s["signal"] == "Inflection"]
    print(json.dumps(signals, indent=2), flush=True)
    post_to_webhook("inflection", signals)
    print("[Task] ✅ Inflection complete.", flush=True)
    return signals

def task_midday():
    print("[Midday] Refreshing active signals...", flush=True)
    results = run_parallel(analyze_signal, DEFAULT_SYMBOLS)
    signals = [s for s in results if s and s["signal"] in ("Breakout", "Inflection")]
    print(json.dumps(signals, indent=2), flush=True)
    post_to_webhook("midday", signals)
    print("[Task] ✅ Midday complete.", flush=True)
    return signals

def task_closing():
    print("[Closing] Ranking RS leaders...", flush=True)
    results = run_parallel(analyze_signal, DEFAULT_SYMBOLS)
    leaders = sorted([s for s in results if s], key=lambda x: -x["close"])[:3]
    print(json.dumps(leaders, indent=2), flush=True)
    post_to_webhook("closing", leaders)
    print("[Task] ✅ Closing complete.", flush=True)
    return leaders

def task_news(symbols=DEFAULT_SYMBOLS[:5]):
    print("[News] Fetching Finnhub headlines...", flush=True)
    today = dt.date.today()
    from_date = today - dt.timedelta(days=1)
    news_report = {}
    for s in symbols:
        try:
            url = f"{FINNHUB_URL}?symbol={s}&from={from_date}&to={today}&token={FINNHUB_API_KEY}"
            r = requests.get(url, timeout=10)
            stories = r.json()[:3]
            news_report[s] = [
                {"headline": n.get("headline"), "source": n.get("source")}
                for n in stories if "headline" in n
            ]
        except Exception as e:
            news_report[s] = f"Error fetching: {e}"
    print(json.dumps(news_report, indent=2), flush=True)
    post_to_webhook("news", [{"symbol": s, "signal": "News", "ATRpct": 0, "close": 0} for s in news_report.keys()])
    print("[Task] ✅ News complete.", flush=True)
    return news_report

# === MAIN ROUTER ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Twelve Data + Finnhub scanner.")
    parser.add_argument("--task", required=True,
                        help="premarket, breakout, inflection, midday, closing, news")
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
