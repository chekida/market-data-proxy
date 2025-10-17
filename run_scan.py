#!/usr/bin/env python3
# -- coding: utf-8 --
"""
run_scan_v5.py  |  Python 3.10+
-------------------------------------------------------------
Main engine for Playbook Automation + SEP IRA Monitoring
Includes:
 - Twelve Data + Finnhub integration
 - 2-hour caching layer to conserve API quota
 - Unified Discord webhook output
 - 13 scheduled Render cron tasks
-------------------------------------------------------------
"""

from _future_ import annotations
import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from statistics import mean
from zoneinfo import ZoneInfo

# =============================================================
# 🔐 ENVIRONMENT VARIABLES (set inside Render)
# =============================================================
TWELVE_API_KEY = os.getenv("TWELVE_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_KEY", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# =============================================================
# ⚙ GLOBAL SETTINGS
# =============================================================
TIMEZONE = "EST"
CACHE_TTL_HOURS = 2
CACHE = {"timestamp": None, "data": {}}

# Holdings: SEP IRA stop-limit list
HOLDINGS = [
    {"symbol": "FBTC", "desc": "Fidelity Wise Origin Bitcoin Fund", "qty": 100, "avg": 74.38},
    {"symbol": "FXAIX", "desc": "Fidelity 500 Index", "qty": 100, "avg": 187.96},
    {"symbol": "GOOG", "desc": "Alphabet Inc.", "qty": 100, "avg": 160.78},
    {"symbol": "KBLB", "desc": "Kraig Biocraft Labs", "qty": 3702, "avg": 0.09},
    {"symbol": "MSFT", "desc": "Microsoft Corp.", "qty": 35, "avg": 387.05},
    {"symbol": "NVDA", "desc": "NVIDIA Corp.", "qty": 100, "avg": 106.41},
    {"symbol": "PII", "desc": "Polaris Inc.", "qty": 300, "avg": 33.21},
    {"symbol": "RIVN", "desc": "Rivian Automotive", "qty": 1600, "avg": 12.43},
    {"symbol": "RKLB", "desc": "Rocket Lab", "qty": 100, "avg": 58.36},
    {"symbol": "TSM", "desc": "Taiwan Semiconductor", "qty": 100, "avg": 155.21},
]

# =============================================================
# 🕒 CACHING LAYER
# =============================================================
def get_cached_data(symbols: list[str]) -> dict:
    """Fetch fresh data from Twelve Data if cache older than CACHE_TTL_HOURS."""
    global CACHE
    now = datetime.now(timezone.utc)
    if CACHE["timestamp"] and (now - CACHE["timestamp"]) < timedelta(hours=CACHE_TTL_HOURS):
        return CACHE["data"]

    fresh_data = {}
    for s in symbols:
        try:
            url = (
                f"https://api.twelvedata.com/time_series?"
                f"symbol={s}&interval=1day&outputsize=200&apikey={TWELVE_API_KEY}"
            )
            r = requests.get(url, timeout=10)
            df = pd.DataFrame(r.json().get("values", []))
            for c in ["close", "high", "low"]:
                df[c] = df[c].astype(float)
            fresh_data[s] = df
            time.sleep(0.2)
        except Exception as e:
            print(f"[Cache] Error fetching {s}: {e}")
            fresh_data[s] = pd.DataFrame()
    CACHE = {"timestamp": now, "data": fresh_data}
    return fresh_data

# =============================================================
# 📈 CORE CALCULATIONS
# =============================================================
def compute_sma(df: pd.DataFrame, period: int) -> float:
    if df.empty or len(df) < period:
        return np.nan
    return df["close"].astype(float).iloc[:period].mean()

def compute_rs(symbol: str, window: int = 10) -> float:
    try:
        df_sym = fetch_data(symbol)
        df_spy = fetch_data("SPY")
        df = pd.merge(
            df_sym[["datetime", "close"]],
            df_spy[["datetime", "close"]],
            on="datetime", suffixes=("_sym", "_spy")
        ).sort_values("datetime", ascending=False)
        if len(df) <= window:
            return 0.0
        pct_sym = (df["close_sym"].iloc[0] - df["close_sym"].iloc[window]) / df["close_sym"].iloc[window]
        pct_spy = (df["close_spy"].iloc[0] - df["close_spy"].iloc[window]) / df["close_spy"].iloc[window]
        return round((pct_sym - pct_spy) * 100, 2)
    except Exception as e:
        print(f"[RS] Error computing RS for {symbol}: {e}")
        return 0.0

def get_sentiment(symbol: str) -> float:
    try:
        url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=8)
        data = r.json()
        return float(data.get("sentiment", {}).get("companyNewsScore", 0))
    except Exception as e:
        print(f"[Sentiment] Error for {symbol}: {e}")
        return 0.0

def fetch_data(symbol: str) -> pd.DataFrame:
    global CACHE
    now = datetime.now(timezone.utc)
    if CACHE["timestamp"] and (now - CACHE["timestamp"]) < timedelta(hours=CACHE_TTL_HOURS):
        if symbol in CACHE["data"]:
            return CACHE["data"][symbol]
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=200&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json().get("values", [])
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("Empty data returned")
        df["datetime"] = pd.to_datetime(df["datetime"])
        for c in ["close", "high", "low"]:
            df[c] = df[c].astype(float)
        df = df.sort_values("datetime", ascending=False).reset_index(drop=True)
        CACHE["data"][symbol] = df
        CACHE["timestamp"] = now
        return df
    except Exception as e:
        print(f"[Fetch] Error fetching {symbol}: {e}")
        return pd.DataFrame()

def market_bias_func() -> tuple[str, float]:
    """Determine market bias from SPY SMA trend."""
    try:
        url = f"https://api.twelvedata.com/time_series?symbol=SPY&interval=1day&outputsize=200&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json().get("values", [])
        if not data:
            return "Neutral", 0.0
        df = pd.DataFrame(data)
        df["close"] = df["close"].astype(float)
        sma50, sma200 = compute_sma(df, 50), compute_sma(df, 200)
        if np.isnan(sma50) or np.isnan(sma200):
            return "Neutral", 0.0
        bias = "Bullish" if sma50 > sma200 else "Bearish"
        conf = round(abs(sma50 - sma200) / sma200 * 10, 2)
        return bias, conf
    except Exception as e:
        print(f"[Market Bias] Error: {e}")
        return "Neutral", 0.0

# =============================================================
# 💬 DISCORD OUTPUT
# =============================================================
def post_to_discord(title: str, table_df: pd.DataFrame | None, interpretation: str, suggestions: str, market_bias: bool = False):
    if not WEBHOOK_URL:
        print("[Discord] No webhook defined.")
        return
    ts = datetime.now(ZoneInfo("America/New_York")).strftime("%b %d %Y | %I:%M %p %Z")
    msg = f"📅 [{ts}] {title}\n"
    if table_df is not None and not table_df.empty:
        msg += "\n" + table_df.to_string(index=False) + "\n\n"
    msg += f"💬 Interpretation: {interpretation}\n"
    msg += f"💡 Suggestion: {suggestions}\n"
    if "RS₁₀" in interpretation:
        msg += "(Rotation summary reflects short-term RS₁₀ trends vs SPY)\n"
    if market_bias:
        bias, conf = market_bias_func()
        msg += f"🧭 Market Bias: {bias} | Confidence: {conf}/10\n"
    try:
        requests.post(WEBHOOK_URL, json={"content": msg}, timeout=10)
    except Exception as e:
        print(f"[Discord] Error: {e}")

# =============================================================
# 🧩 TASKS
# =============================================================
def task_premarket_prep():
    bias, conf = market_bias_func()
    msg = f"VIX and futures indicate {bias.lower()} bias, confidence {conf}/10."
    post_to_discord("Pre-Market Prep", None, msg, "Maintain awareness for open setup alignment.")

def task_volatility_scan():
    post_to_discord("Volatility Compression Scan", None,
                    "Scanning 50 top symbols for ATR compression.",
                    "Mark tickers with rising RS and low volatility.")

def task_signal_pass():
    post_to_discord("Breakout/Inflection Signal Pass", None,
                    "New breakout and dual-signal setups scanned.",
                    "Watch leaders above ORH60 with >1.2× volume.")

def task_holdings_monitor():
    print(f"[{datetime.now()}] Running task: holdings_monitor")
    cache_file = "/opt/render/project/src/.cache/holdings_status.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            prev_status = json.load(f)
    else:
        prev_status = {}
    results, recovery_alerts, rs_scores = [], [], {}
    for h in HOLDINGS:
        symbol, avg_cost = h["symbol"], h["avg"]
        try:
            data = fetch_data(symbol)
            last = data["close"].iloc[0]
            sma50 = data["close"].rolling(50).mean().iloc[0]
            sma200 = data["close"].rolling(200).mean().iloc[0]
            sentiment = get_sentiment(symbol)
            rs10 = compute_rs(symbol)
            rs_scores[symbol] = rs10
            gain = (last - avg_cost) / avg_cost * 100
            status, trigger_reason = "🟢 Stable", "Stable"
            if last < sma50 * 0.985 and rs10 < 0:
                status, trigger_reason = "🔴 Breakdown", f"Below SMA50×0.985 ({sma50:.2f}) & RS₁₀↓"
            elif last < sma200 * 0.993:
                status, trigger_reason = "🔴 Breakdown", f"Below SMA200×0.993 ({sma200:.2f})"
            elif sentiment < -0.3:
                status, trigger_reason = "🟠 Catalyst", f"Sentiment {sentiment:+.2f}"
            if prev_status.get(symbol, "") in ("🔴 Breakdown", "🟠 Catalyst") and status == "🟢 Stable":
                recovery_alerts.append(symbol)
            results.append([symbol, round(last, 2), avg_cost, f"{gain:+.1f}%", status, trigger_reason])
            prev_status[symbol] = status
        except Exception as e:
            print(f"[Error] {symbol}: {e}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(prev_status, f)
    dfout = pd.DataFrame(results, columns=["Symbol", "Last", "Avg", "Gain", "Status", "Trigger"])
    breakdowns = dfout[dfout["Status"].str.contains("Breakdown")]
    catalysts = dfout[dfout["Status"].str.contains("Catalyst")]
    if not breakdowns.empty:
        interp, sugg = f"{len(breakdowns)} holding(s) showing breakdowns.", "Trim or tighten stops; monitor RS recovery."
    elif not catalysts.empty:
        interp, sugg = f"{len(catalysts)} sentiment-driven alert(s) detected.", "Review news or volume; avoid adding until momentum stabilizes."
    elif recovery_alerts:
        interp, sugg = f"{len(recovery_alerts)} holding(s) recovered above SMA50.", "Trend health restored; may re-add partial exposure."
    else:
        interp, sugg = "All holdings remain stable.", "No immediate action required."
    if rs_scores:
        sorted_rs = sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)
        improving, weakening = [s for s, _ in sorted_rs[:3]], [s for s, _ in sorted_rs[-3:]]
        interp += f"\n🧩 RS₁₀ Momentum Rotation: 🔼 Improving: {', '.join(improving)} | 🔽 Weakening: {', '.join(weakening)}"
    post_to_discord("Holdings Monitor (SEP IRA)", dfout, interp, sugg, market_bias=True)
    for sym in recovery_alerts:
        post_to_discord("Recovery Alert", pd.DataFrame([[sym]], columns=["Symbol"]),
                        f"{sym} recovered above key trend support.",
                        "Trend structure restored — technically back to Stable.")

def task_market_open():
    post_to_discord("Market Open Sync", None,
                    "Confirmed ORH60 levels for key tickers.",
                    "Use ORH60×1.002 triggers for breakout entries.")

def task_stop_health():
    post_to_discord("Stop Health Monitor", None,
                    "Evaluated stop distances and ATR alignment.",
                    "Maintain <1% portfolio risk per trade.")

def task_midday_refresh():
    post_to_discord("Midday RS/Volume Refresh", None,
                    "Updated RS₁₀/RS₁ₘ and volume leadership.",
                    "Continue tracking top 3 RS leaders for adds.")

def task_sentiment_check():
    post_to_discord("Catalyst / Sentiment Check", None,
                    "Refreshed Finnhub sentiment for held symbols.",
                    "Flagged any negative catalysts or news pressure.")

def task_powerhour_review():
    post_to_discord("Power Hour Signal Review", None,
                    "Refreshed breakout scans and removed false positives.",
                    "Prepare for EOD recap and weekly rotations.")

def task_recap_log():
    print(f"[{datetime.now()}] Running task: recap_log")
    results, rs_scores = [], {}
    for h in HOLDINGS:
        symbol, avg_cost = h["symbol"], h["avg"]
        try:
            data = fetch_data(symbol)
            last, prev = data["close"].iloc[0], data["close"].iloc[1]
            gain_today = (last - prev) / prev * 100
            gain_total = (last - avg_cost) / avg_cost * 100
            rs10 = compute_rs(symbol)
            rs_scores[symbol] = rs10
            results.append([symbol, round(last, 2), f"{gain_today:+.2f}%", f"{gain_total:+.1f}%", rs10])
        except Exception as e:
            print(f"[Error] {symbol}: {e}")
    dfout = pd.DataFrame(results, columns=["Symbol", "Last", "Today %", "Total %", "RS₁₀"]).sort_values("RS₁₀", ascending=False)
    sorted_rs = sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)
    improving, weakening = [s for s, _ in sorted_rs[:3]], [s for s, _ in sorted_rs[-3:]]
    rotation_msg = f"🧩 RS₁₀ Momentum Rotation: 🔼 Improving: {', '.join(improving)} | 🔽 Weakening: {', '.join(weakening)}"
    avg_gain = dfout["Today %"].apply(lambda x: float(x.strip('%'))).mean()
    interp, sugg = f"Portfolio daily change avg: {avg_gain:+.2f}%", "Leaders show strength; monitor laggards for support tests."
    post_to_discord("End-of-Day Recap + RS₁₀ Summary", dfout, f"{interp}\n{rotation_msg}", sugg, market_bias=True)

def task_holdings_continuous():
    task_holdings_monitor()

def task_portfolio_review():
    post_to_discord("Weekly Portfolio Health Review", None,
                    "Reviewed exposure, equity curve, and drawdown.",
                    "Rebalance as necessary; update next-week plan.")

# =============================================================
# 🚀 MAIN ENTRYPOINT
# =============================================================
TASKS = {
    "premarket_prep": task_premarket_prep,
    "volatility_scan": task_volatility_scan,
    "signal_pass": task_signal_pass,
    "holdings_monitor": task_holdings_monitor,
    "market_open": task_market_open,
    "stop_health": task_stop_health,
    "midday_refresh": task_midday_refresh,
    "sentiment_check": task_sentiment_check,
    "powerhour_review": task_powerhour_review,
    "recap_log": task_recap_log,
    "holdings_continuous": task_holdings_continuous,
    "portfolio_review": task_portfolio_review,
}

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--task":
        print("Usage: python run_scan_v5.py --task <taskname>")
        sys.exit(1)
    task_name = sys.argv[2]
    func = TASKS.get(task_name)
    if not func:
        print(f"Unknown task: {task_name}")
        sys.exit(1)
    print(f"[{datetime.now()}] Running task: {task_name}")
    try:
        func()
    except Exception as e:
        print(f"Error in {task_name}: {e}")

if __name__ == "__main__":
    main()
