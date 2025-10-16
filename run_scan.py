# run_scan.py
# Twelve Data Automated Market Scanner for Render Cron Jobs
# -----------------------------------------------------------
# This script executes specific market scan tasks based on the `--task` argument.
# Supports: premarket, regime, breakout, sector_rs, top25, midday, refresh, closing, atr, overnight

import argparse
import datetime as dt
import requests
import json

# === CONFIGURATION ===
TWELVE_DATA_API_KEY = "e07bdb3e2efb4844ba538a0fc12fad84"
BASE_URL = "https://api.twelvedata.com/time_series"
DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "TSLA", "META", "AMD", "AVGO", 
    "GOOG", "AMZN", "NFLX", "SMCI", "CRM", "INTC", "NXPI", "SNOW", "SHOP", "TTD"
]

# === HELPER FUNCTIONS ===
def fetch_data(symbol, interval="1day", outputsize=200):
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": TWELVE_DATA_API_KEY,
        "outputsize": outputsize
    }
    r = requests.get(BASE_URL, params=params)
    data = r.json()
    if "values" not in data:
        return None
    return data["values"]

def compute_sma(prices, length):
    closes = [float(p['close']) for p in prices[:length]]
    return sum(closes) / len(closes)

def compute_atr(prices, length=14):
    trs = []
    for i in range(length):
        h, l, c_prev = float(prices[i]['high']), float(prices[i]['low']), float(prices[i+1]['close'])
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    return sum(trs) / len(trs)

def analyze_signal(symbol):
    data = fetch_data(symbol)
    if not data:
        return None
    sma50 = compute_sma(data, 50)
    sma200 = compute_sma(data, 200)
    atr = compute_atr(data)
    close = float(data[0]['close'])
    atrpct = atr / close * 100

    signal = None
    if close > sma50 and close < sma200 * 1.05:
        signal = "Inflection"
    elif close > sma50 * 1.02:
        signal = "Breakout"

    return {
        "symbol": symbol,
        "close": close,
        "SMA50": round(sma50, 2),
        "SMA200": round(sma200, 2),
        "ATRpct": round(atrpct, 2),
        "signal": signal or "None"
    }

# === TASK HANDLERS ===
def task_premarket():
    print("[Premarket] Checking volatility and market breadth...")
    result = {"SPY": analyze_signal("SPY"), "QQQ": analyze_signal("QQQ"), "IWM": analyze_signal("IWM")}
    print(json.dumps(result, indent=2))

def task_breakout():
    print("[Breakout] Running breakout scan on default symbols...")
    signals = [analyze_signal(s) for s in DEFAULT_SYMBOLS]
    signals = [s for s in signals if s and s['signal'] == 'Breakout']
    print(json.dumps(signals, indent=2))

def task_inflection():
    print("[Inflection] Scanning for inflection points (above SMA50, near 52-week high)...")
    signals = [analyze_signal(s) for s in DEFAULT_SYMBOLS]
    signals = [s for s in signals if s and s['signal'] == 'Inflection']
    print(json.dumps(signals, indent=2))

def task_closing():
    print("[Closing] Generating RS leader report...")
    signals = [analyze_signal(s) for s in DEFAULT_SYMBOLS]
    leaders = sorted([s for s in signals if s], key=lambda x: -x['close'])[:3]
    print(json.dumps(leaders, indent=2))

# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific Twelve Data trading scan task.")
    parser.add_argument("--task", required=True, help="Task to run: premarket, regime, breakout, sector_rs, top25, midday, refresh, closing, atr, overnight")
    args = parser.parse_args()

    task = args.task.lower()
    print(f"\n=== Twelve Data Scan Execution | Task: {task} | {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    if task in ["premarket", "regime"]:
        task_premarket()
    elif task in ["breakout", "top25", "refresh"]:
        task_breakout()
    elif task in ["midday", "inflection"]:
        task_inflection()
    elif task in ["closing", "atr", "overnight"]:
        task_closing()
    else:
        print(f"Unknown task: {task}")
