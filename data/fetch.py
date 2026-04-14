"""
Commodity price data fetcher with multiple source fallbacks.
Tests: Oil (WTI), Gold, Copper, 10Y Treasury, USD Index, SPX

Sources tried in order:
  1. yfinance  (fast, often unreliable)
  2. FRED API  (reliable, requires free API key for some series)
  3. Stooq    (reliable, no auth needed)

Usage:
    python data/fetch.py

Output:
    data/raw/<ticker>.csv  for each series
"""

import os
import time
import urllib.request
import urllib.parse
import json
import csv
from datetime import datetime, timedelta
from io import StringIO

# ── Config ────────────────────────────────────────────────────────────────────

START_DATE = "2010-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR = "data/raw"

# FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
# Leave empty to skip FRED and fall through to Stooq
FRED_API_KEY = ""

SERIES = {
    "oil":    {"yf": "CL=F",    "fred": "DCOILWTICO",  "stooq": "cl.f"},
    "gold":   {"yf": "GC=F",    "fred": "GOLDAMGBD228NLBM", "stooq": "gc.f"},
    "copper": {"yf": "HG=F",    "fred": "PCOPPUSDM",   "stooq": "hg.f"},
    "tnote":  {"yf": "^TNX",    "fred": "DGS10",       "stooq": "tnx.b"},
    "usd":    {"yf": "DX-Y.NYB","fred": "DTWEXBGS",    "stooq": "usdidx.fx"},
    "spx":    {"yf": "^GSPC",   "fred": None,          "stooq": "^spx"},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_url(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def save_csv(name, rows, header=("date", "close")):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def parse_date(s):
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None

# ── Source 1: yfinance ────────────────────────────────────────────────────────

def fetch_yfinance(ticker, start, end):
    """Download via yfinance library if installed."""
    try:
        import yfinance as yf
        import pandas as pd
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError("Empty dataframe returned")

        # yfinance sometimes returns MultiIndex columns like ("Close", "CL=F")
        # flatten to single level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        if "Close" not in df.columns:
            raise RuntimeError(f"No Close column, got: {list(df.columns)}")

        close = df["Close"]
        rows = []
        for d, v in zip(df.index, close):
            try:
                rows.append((str(d.date()), round(float(v), 4)))
            except (ValueError, TypeError):
                pass  # skip non-numeric rows

        if not rows:
            raise RuntimeError("No numeric rows after parsing")
        return rows
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"yfinance failed: {e}")

# ── Source 2: FRED ────────────────────────────────────────────────────────────

def fetch_fred(series_id, start, end, api_key):
    if not api_key:
        raise RuntimeError("No FRED API key provided")
    if not series_id:
        raise RuntimeError("No FRED series ID for this ticker")
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&observation_start={start}"
        f"&observation_end={end}"
        f"&api_key={api_key}"
        "&file_type=json"
    )
    raw = fetch_url(url)
    data = json.loads(raw)
    if "observations" not in data:
        raise RuntimeError(f"FRED error: {data.get('error_message', 'unknown')}")
    rows = []
    for obs in data["observations"]:
        v = obs["value"]
        if v == ".":          # FRED uses "." for missing
            continue
        rows.append((obs["date"], round(float(v), 4)))
    if not rows:
        raise RuntimeError("FRED returned no data")
    return rows

# ── Source 3: Stooq ───────────────────────────────────────────────────────────

def fetch_stooq(symbol, start, end):
    """
    Stooq offers free historical CSV downloads.
    URL format: https://stooq.com/q/d/l/?s=<symbol>&d1=YYYYMMDD&d2=YYYYMMDD&i=d
    """
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={urllib.parse.quote(symbol)}&d1={d1}&d2={d2}&i=d"
    raw = fetch_url(url)
    if "No data" in raw or len(raw) < 30:
        raise RuntimeError("Stooq returned no data")
    reader = csv.DictReader(StringIO(raw))
    rows = []
    for row in reader:
        date_str = parse_date(row.get("Date", ""))
        close_str = row.get("Close", "")
        if date_str and close_str and close_str != "null":
            try:
                rows.append((date_str, round(float(close_str), 4)))
            except ValueError:
                pass
    if not rows:
        raise RuntimeError("Stooq CSV parsed but no rows extracted")
    return sorted(rows)

# ── Orchestrator ──────────────────────────────────────────────────────────────

def fetch_with_fallback(name, cfg, start, end):
    attempts = [
        ("yfinance", lambda: fetch_yfinance(cfg["yf"],   start, end)),
        ("FRED",     lambda: fetch_fred(cfg["fred"], start, end, FRED_API_KEY)),
        ("Stooq",    lambda: fetch_stooq(cfg["stooq"], start, end)),
    ]
    for source_name, fn in attempts:
        try:
            rows = fn()
            print(f"  ✓ {name:<8} ← {source_name}  ({len(rows)} rows)")
            return rows
        except RuntimeError as e:
            print(f"  ✗ {name:<8}   {source_name}: {e}")
        time.sleep(0.3)   # polite pause between retries
    return None


def main():
    print(f"\nFetching commodity data  {START_DATE} → {END_DATE}\n")
    results = {}
    for name, cfg in SERIES.items():
        rows = fetch_with_fallback(name, cfg, START_DATE, END_DATE)
        if rows:
            path = save_csv(name, rows)
            results[name] = {"rows": len(rows), "path": path}
        else:
            print(f"  !! {name}: all sources failed — skipping")

    print("\n── Summary ──────────────────────────────────────")
    for name, info in results.items():
        print(f"  {name:<8} {info['rows']:>6} rows  →  {info['path']}")
    missing = [n for n in SERIES if n not in results]
    if missing:
        print(f"\n  Missing: {', '.join(missing)}")
        print("  Tip: set FRED_API_KEY at the top of this file for better coverage.")
    print()


if __name__ == "__main__":
    main()
