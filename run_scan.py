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

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # For older Python versions, if needed

# Configure logging for diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ENVIRONMENT VARIABLES
TWELVE_API_KEY = os.getenv("TWELVE_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_KEY", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# GLOBAL SETTINGS
TIMEZONE = "US/Eastern"  # Zone for EST/EDT handling
CACHE_TTL_HOURS = 2
CACHE = {"timestamp": None, "data": {}}

# Holdings list omitted for brevity - keep as before

ACCOUNT_NOTE = (
    "🧾 *This portfolio is held within a SEP IRA (tax-sheltered). "
    "Rebalancing and trimming are tax-free; focus purely on risk optimization.*"
)

def get_est_timestamp() -> str:
    now_utc = datetime.now(timezone.utc)
    now_est = now_utc.astimezone(ZoneInfo(TIMEZONE))
    return now_est.strftime("%b %d %Y | %I:%M %p %Z")

def get_cached_data(symbols: list[str]) -> dict:
    # (body same as earlier, no change except can use logging)
    global CACHE
    now = datetime.now(timezone.utc)
    
    if CACHE["timestamp"] is not None and (now - CACHE["timestamp"]) < timedelta(hours=CACHE_TTL_HOURS):
        logging.info("Using cached data")
        return CACHE["data"]
    
    fresh_data = {}
    for s in symbols:
        try:
            url = (f"https://api.twelvedata.com/time_series?"
                   f"symbol={s}&interval=1day&outputsize=200&apikey={TWELVE_API_KEY}")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            df = pd.DataFrame(r.json().get("values", []))
            if not df.empty:
                df["close"] = df["close"].astype(float)
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
            fresh_data[s] = df
            time.sleep(0.2)
        except Exception as e:
            logging.error(f"[Cache] Error fetching {s}: {e}")
            fresh_data[s] = pd.DataFrame()
    CACHE = {"timestamp": now, "data": fresh_data}
    return fresh_data

def post_to_discord(title: str, table_df: pd.DataFrame | None, interpretation: str, suggestions: str):
    if not WEBHOOK_URL:
        logging.warning("[Discord] No webhook defined.")
        return

    ts = get_est_timestamp()
    msg = f"📅 **[{ts}] {title}**\n"
    if table_df is not None and not table_df.empty:
        msg += "``````\n"
    msg += f"💬 **Interpretation:** {interpretation}\n💡 **Suggestion:** {suggestions}\n{ACCOUNT_NOTE}\n"

    bias, conf = market_bias()
    msg += f"🧭 *Market Bias: {bias} | Confidence: {conf}/10*\n"

    try:
        response = requests.post(WEBHOOK_URL, json={"content": msg}, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"[Discord] Error posting message: {e}")

# Remaining functions and CLI main unchanged but ensure all timestamps use get_est_timestamp where needed

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
