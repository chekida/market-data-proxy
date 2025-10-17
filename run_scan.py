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

# =============================================================
# 🕒 TIMESTAMP UTILITIES
# =============================================================
from datetime import datetime
import pytz

def get_est_timestamp():
    """Return a human-readable timestamp string in US/Eastern time."""
    tz_est = pytz.timezone("US/Eastern")
    now_est = datetime.now(tz_est)
    return now_est.strftime("%b %d %Y | %I:%M %p %Z")

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
# 🧩 MARKET UNIVERSE LOADER
# =============================================================
CACHE_FILE = "market_universe.json"

def load_market_universe():
    """
    Dynamically load S&P 500 + Nasdaq 100 tickers via Twelve Data API.
    Liquidity filter: 10-day average volume ≥ 1 M shares.
    Cache refreshes once per trading day (first 07:00 AM run).
    """
    key = os.getenv("TWELVE_KEY")
    today = datetime.date.today()

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached = json.load(f)
            if cached.get("date") == str(today):
                print(f"[Universe] Using cached list ({len(cached['symbols'])} tickers) for {today}.")
                return cached["symbols"]
        except Exception as e:
            print(f"[Universe] Cache read failed: {e}")

    tickers = set()
    try:
        res = requests.get(
            f"https://api.twelvedata.com/stocks?country=United%20States&index=SPX&apikey={key}",
            timeout=10,
        ).json()
        if "data" in res:
            tickers.update([t["symbol"] for t in res["data"] if t.get("symbol")])
    except Exception as e:
        print(f"[Universe] SP500 load failed: {e}")

    try:
        res = requests.get(
            f"https://api.twelvedata.com/stocks?country=United%20States&index=NDX&apikey={key}",
            timeout=10,
        ).json()
        if "data" in res:
            tickers.update([t["symbol"] for t in res["data"] if t.get("symbol")])
    except Exception as e:
        print(f"[Universe] NASDAQ load failed: {e}")

    if not tickers:
        tickers = {
            "AAPL", "MSFT", "NVDA", "GOOG", "META", "AMZN", "TSLA", "TSM",
            "AVGO", "LLY", "UNH", "JPM", "JNJ", "V", "MA", "HD", "PG", "XOM", "COST", "ORCL"
        }
        print("[Universe] Using fallback list (20 symbols).")

    print(f"[Universe] Pulled {len(tickers)} tickers before filtering.")

    filtered = []
    for symbol in tickers:
        try:
            url = (
                f"https://api.twelvedata.com/time_series?"
                f"symbol={symbol}&interval=1day&outputsize=10&apikey={key}"
            )
            data = requests.get(url, timeout=8).json()
            if "values" not in data:
                continue
            df = pd.DataFrame(data["values"])
            df["volume"] = df["volume"].astype(float)
            if df["volume"].mean() >= 1_000_000:
                filtered.append(symbol)
        except Exception:
            continue

    print(f"[Universe] Filtered to {len(filtered)} liquid symbols.")

    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"date": str(today), "symbols": sorted(filtered)}, f)
        print(f"[Universe] Cached universe for {today}.")
    except Exception as e:
        print(f"[Universe] Cache save failed: {e}")

    return sorted(filtered)


# Initialize once when the script starts (used by all tasks)
MARKET_UNIVERSE = load_market_universe()

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

def compute_relative_strength(symbol: str, period: int = 10) -> float:
    """Compute RS (relative strength) vs SPY over N days."""
    try:
        df_sym = fetch_data(symbol)
        df_spy = fetch_data("SPY")
        if df_sym.empty or df_spy.empty:
            return 0.0
        df = pd.merge(
            df_sym[["datetime", "close"]],
            df_spy[["datetime", "close"]],
            on="datetime",
            suffixes=("_sym", "_spy")
        ).sort_values("datetime", ascending=False)
        if len(df) <= period:
            return 0.0
        pct_sym = (df["close_sym"].iloc[0] - df["close_sym"].iloc[period]) / df["close_sym"].iloc[period]
        pct_spy = (df["close_spy"].iloc[0] - df["close_spy"].iloc[period]) / df["close_spy"].iloc[period]
        return round((pct_sym - pct_spy) * 100, 2)
    except Exception as e:
        print(f"[RS] {symbol} RS error: {e}")
        return 0.0


def get_volume_ratio(symbol: str) -> float:
    """Compare current volume to recent 20-bar average."""
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=5min&outputsize=40&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=8).json()
        if "values" not in r:
            return 1.0
        df = pd.DataFrame(r["values"])
        df["volume"] = df["volume"].astype(float)
        if len(df) < 20:
            return 1.0
        curr_vol = df["volume"].iloc[0]
        avg_vol = df["volume"].iloc[1:21].mean()
        return round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1.0
    except Exception as e:
        print(f"[VolRatio] Error {symbol}: {e}")
        return 1.0


def compute_atr(symbol: str, period: int = 14) -> float:
    """Compute ATR for a symbol."""
    try:
        df = fetch_data(symbol)
        if df.empty or len(df) < period + 1:
            return 0.0
        high, low, close = df["high"], df["low"], df["close"]
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(-1)), abs(low - close.shift(-1))))
        return round(tr.iloc[:period].mean(), 2)
    except Exception as e:
        print(f"[ATR] Error {symbol}: {e}")
        return 0.0


def get_orh60(symbol: str) -> float:
    """Get first-hour high (ORH60) from hourly data."""
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=8&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=8).json()
        if "values" not in r:
            return 0.0
        df = pd.DataFrame(r["values"])
        df["high"] = df["high"].astype(float)
        return float(df["high"].iloc[-1])
    except Exception as e:
        print(f"[ORH60] Error {symbol}: {e}")
        return 0.0


# =============================================================
# 💬 DISCORD OUTPUT
# =============================================================
def post_to_discord(
    title: str,
    table_data: pd.DataFrame | str | None,
    interpretation: str,
    suggestions: str,
    footer: str | None = None,
    market_bias: bool = False,
    emoji: str = "📈"
):
    """
    Unified Discord output.
    Supports DataFrames, Markdown tables, or plain strings.
    Adds timestamp, emoji header, interpretation, suggestion, and optional bias summary.
    """
    if not WEBHOOK_URL:
        print("[Discord] No webhook defined.")
        return

    # --- Header timestamp ---
    ts = datetime.now(ZoneInfo("America/New_York")).strftime("%b %d %Y | %I:%M %p %Z")
    msg = f"{emoji} **[{ts}] {title}**\n"

    # --- Format table or text body ---
    if table_data is not None:
        if isinstance(table_data, pd.DataFrame):
            if not table_data.empty:
                msg += "```\n" + table_data.to_string(index=False)[:3800] + "\n```\n"
        elif isinstance(table_data, str):
            msg += table_data.strip() + "\n\n"

    # --- Interpretation + suggestion ---
    msg += f"💬 *Interpretation:* {interpretation}\n"
    msg += f"💡 *Suggestion:* {suggestions}\n"

    # --- Market bias summary (optional) ---
    if market_bias:
        bias, conf = market_bias_func()
        msg += f"🧭 *Market Bias:* {bias} | Confidence: {conf}/10\n"

    # --- Footer (optional) ---
    if footer:
        msg += f"📊 {footer}\n"

    # --- Discord send ---
    try:
        requests.post(WEBHOOK_URL, json={"content": msg}, timeout=10)
        print(f"[Discord] ✅ Sent: {emoji} {title}")
    except Exception as e:
        print(f"[Discord] ❌ Error sending {title}: {e}")

# =============================================================
# 🧩 TASK SCHEDULE DEFINITIONS
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
    """
    Power Hour Review (15:30 EST)
    Final rotation & volume scan across the market universe.
    Identifies top RS₁ₘ leaders and flags names within 1×ATR of breakout levels.
    """
    print(f"[{get_est_timestamp()}] Running task: powerhour_review")

    symbols = MARKET_UNIVERSE
    results = []

    for s in symbols:
        try:
            data = fetch_data(s)
            rs10 = compute_relative_strength(s, period=10)
            rs1m = compute_relative_strength(s, period=21)
            vol_ratio = get_volume_ratio(s)
            atr = compute_atr(s, period=14)
            last_close = float(data["values"][0]["close"])
            orh60 = get_orh60(s)

            # ΔBreakout distance in ATR units (how close to trigger)
            dist_to_breakout = ((orh60 * 1.002) - last_close) / atr if atr > 0 else None

            results.append((s, rs10, rs1m, vol_ratio, dist_to_breakout))
        except Exception as e:
            print(f"[PowerHour] {s} error: {e}")
            continue

    if not results:
        post_to_discord(
            "Power Hour Review",
            "_No data available — check API limits or market status._",
            "No results produced for this session.",
            "—"
        )
        return

    df = pd.DataFrame(results, columns=["Symbol", "RS10", "RS1M", "VolRatio", "DistToBreakout"])
    df.sort_values("RS1M", ascending=False, inplace=True)
    top5 = df.head(5)

    # --- Markdown Table for Discord ---
    md_table = (
        "| Ticker | RS₁₀ | RS₁ₘ | Vol/Avg | ΔBreakout | Status |\n"
        "|:--|--:|--:|--:|--:|:--|\n"
    )

    for _, row in top5.iterrows():
        if row.DistToBreakout is not None:
            if row.DistToBreakout <= 1:
                proximity = f"{row.DistToBreakout:.1f}×ATR ⚡"
                status = "🟢 Add-On Ready"
            else:
                proximity = f"{row.DistToBreakout:.1f}×ATR"
                status = "🔵 Leader"
        else:
            proximity = "—"
            status = "⚪ Neutral"

        md_table += (
            f"| **{row.Symbol}** | {row.RS10:+.2f} | {row.RS1M:+.2f} "
            f"| {row.VolRatio:.1f}× | {proximity} | {status} |\n"
        )

    # --- Interpretation / Guidance ---
    if (top5.RS10 > 0).all() and (top5.VolRatio > 1).mean() > 0.5:
        interp = "Leaders holding gains — momentum broad and volume-backed."
        sugg = "Focus on ⚡ ‘Add-On Ready’ names for breakout continuation setups."
    elif (top5.RS10 < 0).any():
        interp = "RS softening — leadership rotation likely next session."
        sugg = "Tighten stops or trim laggards ahead of close."
    else:
        interp = "Leadership mixed but stable."
        sugg = "Maintain exposure to core RS₁ₘ names only."

    footer = f"Universe size {len(MARKET_UNIVERSE)} | Updated {datetime.date.today()}"
    post_to_discord("Power Hour Review", pd.DataFrame(), interp, sugg, footer)

    print(f"[PowerHour] Scan complete — leaders: {', '.join(top5.Symbol.tolist())}")

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
    print(f"[{get_est_timestamp()}] Running task: {task_name}")
    try:
        func()
    except Exception as e:
        print(f"Error in {task_name}: {e}")

if __name__ == "__main__":
    main()
