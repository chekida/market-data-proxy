#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_scan.py | Python 3.10+

Main engine for Playbook Automation + SEP IRA Monitoring
Includes:
- Twelve Data + Finnhub integration
- 2-hour caching layer to conserve API quota
- Unified Discord webhook output
- 13 scheduled Render cron tasks
"""

from __future__ import annotations
import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from statistics import mean
import logging

# Configure logging for better diagnostics
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# ENVIRONMENT VARIABLES (set inside Render)
TWELVE_API_KEY = os.getenv("TWELVE_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_KEY", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# GLOBAL SETTINGS
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

ACCOUNT_NOTE = (
    "🧾 *This portfolio is held within a SEP IRA (tax-sheltered). "
    "Rebalancing and trimming are tax-free; focus purely on risk optimization.*"
)

def get_cached_data(symbols: list[str]) -> dict:
    """
    Fetch fresh data from Twelve Data if cache older than CACHE_TTL_HOURS.
    Returns dict {symbol: DataFrame}
    """
    global CACHE
    now = datetime.now(timezone.utc)
    
    if CACHE["timestamp"] is not None and (now - CACHE["timestamp"]) < timedelta(hours=CACHE_TTL_HOURS):
        logging.info("Using cached data")
        return CACHE["data"]
    
    fresh_data = {}
    for s in symbols:
        try:
            url = (
                f"https://api.twelvedata.com/time_series?"
                f"symbol={s}&interval=1day&outputsize=200&apikey={TWELVE_API_KEY}"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            df = pd.DataFrame(r.json().get("values", []))
            if not df.empty:
                df["close"] = df["close"].astype(float)
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
            fresh_data[s] = df
            time.sleep(0.2)  # gentle throttle
        except Exception as e:
            logging.error(f"[Cache] Error fetching {s}: {e}")
            fresh_data[s] = pd.DataFrame()
    CACHE = {"timestamp": now, "data": fresh_data}
    return fresh_data

# Other financial calculation functions (compute_sma, compute_atr, compute_rsi, etc.) are unchanged
# but should also include proper logging on exceptional cases

def post_to_discord(title: str, table_df: pd.DataFrame | None, interpretation: str, suggestions: str):
    """
    Unified Discord webhook output.
    """
    if not WEBHOOK_URL:
        logging.warning("[Discord] No webhook defined.")
        return

    ts = datetime.now(timezone.utc).strftime("%b %d %Y | %I:%M %p UTC")
    msg = f"📅 **[{ts}] {title}**\n"
    if table_df is not None and not table_df.empty:
        msg += "``````\n"
    msg += f"💬 **Interpretation:** {interpretation}\n"
    msg += f"💡 **Suggestion:** {suggestions}\n"
    msg += f"{ACCOUNT_NOTE}\n"

    bias, conf = market_bias()
    msg += f"🧭 *Market Bias: {bias} | Confidence: {conf}/10*\n"

    try:
        response = requests.post(WEBHOOK_URL, json={"content": msg}, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"[Discord] Error posting message: {e}")

# CLI Entrypoint and task functions remain largely unchanged 
# Just ensure all error handling uses logging and that all functions follow consistent exception management

# Example CLI main:
def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--task":
        logging.error("Usage: python run_scan.py --task <task_name>")
        sys.exit(1)

    task_name = sys.argv[2]
    task_func = TASKS.get(task_name)

    if not task_func:
        logging.error(f"Unknown task: {task_name}")
        sys.exit(1)

    logging.info(f"Running task: {task_name}")

    try:
        task_func()
    except Exception as e:
        logging.error(f"Error in task {task_name}: {e}")

if __name__ == "__main__":
    main()
