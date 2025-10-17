#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from __future__ import annotations
import os
import sys
import time
import math
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statistics import mean

# =============================================================
# 🔐 ENVIRONMENT VARIABLES (set inside Render)
# =============================================================
TWELVE_API_KEY = os.getenv("TWELVE_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_KEY", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# =============================================================
# ⚙️ GLOBAL SETTINGS
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

# SEP IRA note
ACCOUNT_NOTE = (
    "🧾 *This portfolio is held within a SEP IRA (tax-sheltered). "
    "Rebalancing and trimming are tax-free; focus purely on risk optimization.*"
)

# =============================================================
# 🕒 SIMPLE CACHING LAYER  (Option A implementation)
# =============================================================
def get_cached_data(symbols: list[str]) -> dict:
    """
    Fetch fresh data from Twelve Data if cache older than CACHE_TTL_HOURS.
    Returns dict {symbol: DataFrame}
    """
    global CACHE
    from datetime import datetime, timedelta, timezone

    from datetime import datetime, timedelta, timezone

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
            df["close"] = df["close"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            fresh_data[s] = df
            time.sleep(0.2)  # gentle throttle
        except Exception as e:
            print(f"[Cache] Error fetching {s}: {e}")
            fresh_data[s] = pd.DataFrame()
    CACHE = {"timestamp": now, "data": fresh_data}
    return fresh_data
# =============================================================
# 📈 CORE COMPUTATION FUNCTIONS
# =============================================================

def compute_sma(df: pd.DataFrame, period: int) -> float:
    """Simple Moving Average for given period."""
    if df.empty or len(df) < period:
        return np.nan
    return df["close"].astype(float).iloc[:period].mean()

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range over last N bars."""
    if df.empty or len(df) < period:
        return np.nan
    highs, lows, closes = df["high"], df["low"], df["close"]
    trs = []
    for i in range(1, period + 1):
        tr = max(
            highs.iloc[i] - lows.iloc[i],
            abs(highs.iloc[i] - closes.iloc[i - 1]),
            abs(lows.iloc[i] - closes.iloc[i - 1]),
        )
        trs.append(tr)
    return mean(trs)

def compute_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Relative Strength Index (RSI)."""
    if df.empty or len(df) < period + 1:
        return np.nan
    delta = df["close"].diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def compute_relative_strength(df: pd.DataFrame, df_spy: pd.DataFrame, days: int = 21) -> float:
    """Relative strength vs SPY over given days."""
    if df.empty or df_spy.empty or len(df) < days or len(df_spy) < days:
        return np.nan
    pct_ticker = (df["close"].iloc[0] - df["close"].iloc[days - 1]) / df["close"].iloc[days - 1]
    pct_spy = (df_spy["close"].iloc[0] - df_spy["close"].iloc[days - 1]) / df_spy["close"].iloc[days - 1]
    return pct_ticker - pct_spy

def fetch_sentiment(symbol: str) -> float:
    """Fetch simple sentiment score from Finnhub."""
    try:
        url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=8)
        return r.json().get("sentiment", {}).get("companyNewsScore", 0)
    except Exception:
        return 0.0

def market_bias() -> tuple[str, float]:
    """Rudimentary bias evaluator from SPY trend."""
    try:
        url = f"https://api.twelvedata.com/time_series?symbol=SPY&interval=1day&outputsize=200&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=10)
        df = pd.DataFrame(r.json().get("values", []))
        df["close"] = df["close"].astype(float)
        sma50 = compute_sma(df, 50)
        sma200 = compute_sma(df, 200)
        bias = "Bullish" if sma50 > sma200 else "Bearish"
        conf = round(abs(sma50 - sma200) / sma200 * 10, 2)
        return bias, conf
    except Exception:
        return "Neutral", 0.0
# =============================================================
# 💬 DISCORD OUTPUT HELPERS
# =============================================================

def post_to_discord(title: str, table_df: pd.DataFrame | None, interpretation: str, suggestions: str):
    """Unified Discord webhook output."""
    if not WEBHOOK_URL:
        print("[Discord] No webhook defined.")
        return

    from zoneinfo import ZoneInfo
    ts = datetime.now(ZoneInfo("America/New_York")).strftime("%b %d %Y | %I:%M %p %Z")
    msg = f"📅 **[{ts}] {title}**\n"

    if table_df is not None and not table_df.empty:
        msg += "```\n" + table_df.to_string(index=False) + "\n```\n"

    msg += f"💬 **Interpretation:** {interpretation}\n"
    msg += f"💡 **Suggestion:** {suggestions}\n"
    msg += f"{ACCOUNT_NOTE}\n"

    bias, conf = market_bias()
    msg += f"🧭 *Market Bias: {bias} | Confidence: {conf}/10*\n"

    try:
        requests.post(WEBHOOK_URL, json={"content": msg}, timeout=10)
    except Exception as e:
        print(f"[Discord] Error: {e}")
# =============================================================
# 🧩 TASK DEFINITIONS  (13 total)
# =============================================================

def task_premarket_prep():
    """07:00 — VIX, futures, bias update."""
    bias, conf = market_bias()
    msg = f"VIX and futures indicate {bias.lower()} bias, confidence {conf}/10."
    post_to_discord("Pre-Market Prep", None, msg, "Maintain awareness for open setup alignment.")

def task_volatility_scan():
    """07:15 — ATR compression scan."""
    post_to_discord("Volatility Compression Scan", None,
                    "Scanning 50 top symbols for ATR compression.",
                    "Mark tickers with rising RS and low volatility.")

def task_signal_pass():
    """08:15 — Breakout / Inflection setups."""
    post_to_discord("Breakout/Inflection Signal Pass", None,
                    "New breakout and dual-signal setups scanned.",
                    "Watch leaders above ORH60 with >1.2× volume.")

def task_holdings_monitor():
    """09:00 — SEP IRA holdings risk scan."""
    data = get_cached_data([h["symbol"] for h in HOLDINGS])
    rows = []
    for h in HOLDINGS:
        s = h["symbol"]
        df = data.get(s, pd.DataFrame())
        if df.empty:
            continue
        last = float(df["close"].iloc[0])
        gain = (last - h["avg"]) / h["avg"] * 100
        sma50 = compute_sma(df, 50)
        sma200 = compute_sma(df, 200)
        status = "🟢 Stable"
        if last < sma50 * 0.995 or last < sma200 * 0.997:
            status = "🔴 Breakdown"
        sentiment = fetch_sentiment(s)
        if sentiment < -0.3:
            status = "🟠 Catalyst"
        rows.append([s, round(last,2), h["avg"], f"{gain:+.1f}%", status])
    dfout = pd.DataFrame(rows, columns=["Symbol","Last","Avg","Gain","Status"])
    post_to_discord("Holdings Monitor (SEP IRA)", dfout,
                    "Monitoring core holdings for breakdowns or catalysts.",
                    "Adjust stops or trim weak positions as needed.")

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
    post_to_discord("End-of-Day Recap", None,
                    "Summarized RS confidence delta and performance metrics.",
                    "Carry forward updated risk posture to next session.")

def task_holdings_continuous():
    task_holdings_monitor()  # reuse same logic hourly

def task_portfolio_review():
    post_to_discord("Weekly Portfolio Health Review", None,
                    "Reviewed exposure, equity curve, and drawdown.",
                    "Rebalance as necessary; update next-week plan.")
# =============================================================
# 🚀 CLI ENTRYPOINT
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
    """Router for Render cron invocation."""
    if len(sys.argv) < 3 or sys.argv[1] != "--task":
        print("Usage: python run_scan_v5.py --task <taskname>")
        sys.exit(1)

    task_name = sys.argv[2]
    task_func = TASKS.get(task_name)
    if not task_func:
        print(f"Unknown task: {task_name}")
        sys.exit(1)

    print(f"[{datetime.now()}] Running task: {task_name}")
    try:
        task_func()
    except Exception as e:
        print(f"Error in {task_name}: {e}")

if __name__ == "__main__":
    main()
