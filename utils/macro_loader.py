"""
Macro economic feature pipeline — two modes:

1. load_macro_features(processed)  [non-vintage, fast]
   Fetches current-vintage daily/monthly/quarterly FRED series and aligns
   them to the project's trading-day index.  Used for rapid prototyping.

2. build_macro_pit(processed)      [real-time vintage, no-lookahead]
   Fetches ALL historical vintages from FRED (output_type=4) and builds a
   point-in-time correct feature matrix: every value on trading day t was
   publicly available before market open on t.  Used for production training.

Public API
----------
    # Non-vintage (existing)
    load_macro_features(processed) -> pd.DataFrame

    # Vintage / point-in-time (new)
    fetch_realtime_series(fred_key, series_id, start_date, end_date)
                                                   -> pd.DataFrame
    build_point_in_time_series(vintage_df, trading_days)
                                                   -> (values, days_stale, is_revision)
    compute_revision_surprise(vintage_df)          -> pd.Series
    build_macro_pit(processed)                     -> pd.DataFrame
"""

import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ── FRED series catalogue — non-vintage loader ────────────────────────────────

_DAILY_FRED: dict[str, str] = {
    "dgs2":   "DGS2",
    "t10yie": "T10YIE",
}

_MONTHLY_LEVEL_FRED: dict[str, tuple[str, str]] = {
    "cpi_yoy":          ("CPIAUCSL", "yoy"),
    "corecpi_yoy":      ("CPILFESL", "yoy"),
    "ppi_yoy":          ("PPIACO",   "yoy"),
    "retail_sales_mom": ("RSXFS",    "mom"),
}

_MONTHLY_DIRECT_FRED: dict[str, str] = {
    "ism_mfg_proxy": "MANEMP",
    "unrate":        "UNRATE",
    "fedfunds":      "FEDFUNDS",
}

_QUARTERLY_DIRECT_FRED: dict[str, str] = {
    "gdp_qoq": "A191RL1Q225SBEA",
}

_COL_ORDER: list[str] = [
    "vix", "dgs2", "t10yie", "copper_gold", "spread_2s10s",
    "cpi_yoy", "corecpi_yoy", "ppi_yoy", "retail_sales_mom",
    "ism_mfg_proxy", "unrate", "fedfunds",
    "gdp_qoq",
]

# ── FRED series catalogue — vintage / PIT loader ──────────────────────────────

# series_id → human-readable description
_PIT_SERIES: dict[str, str] = {
    "UNRATE":          "Unemployment Rate",
    "CPIAUCSL":        "CPI All Items",
    "CPILFESL":        "Core CPI",
    "PPIACO":          "PPI",
    "RSXFS":           "Retail Sales",
    "FEDFUNDS":        "Federal Funds Rate",
    "A191RL1Q225SBEA": "GDP Growth QoQ",
}

_CACHE_DIR = Path("data/cache")


# ── .env / API key helpers ────────────────────────────────────────────────────

def _load_fred_key() -> str:
    """Read FRED_API_KEY from the .env file at the repository root.

    Tries python-dotenv first; falls back to manual line parsing so the file
    works without the extra dependency.

    Raises:
        FileNotFoundError: .env does not exist.
        ValueError:        Key is absent or still the placeholder value.
    """
    # python-dotenv path (preferred)
    try:
        from dotenv import load_dotenv  # pip install python-dotenv
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
        key = os.environ.get("FRED_API_KEY", "").strip().strip('"').strip("'")
        if key and key != "your_key_here":
            return key
    except ImportError:
        pass

    # Manual fallback
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        raise FileNotFoundError(
            f".env not found at {env_path}\n"
            "Create it and add the line:  FRED_API_KEY=<your_key>\n"
            "Free keys: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    with open(env_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("FRED_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if not key or key == "your_key_here":
                    raise ValueError(
                        "FRED_API_KEY in .env is still the placeholder value.\n"
                        "Replace 'your_key_here' with your real FRED API key."
                    )
                return key
    raise ValueError("FRED_API_KEY line not found in .env.")


# ── Network helpers — non-vintage ─────────────────────────────────────────────

def _fetch_url(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _fetch_fred(series_id: str, start: str, end: str, api_key: str) -> pd.Series:
    """Fetch one FRED series (current vintage) and return a dated pd.Series."""
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&observation_start={start}"
        f"&observation_end={end}"
        f"&api_key={api_key}"
        "&file_type=json"
    )
    raw = _fetch_url(url)
    payload = json.loads(raw)
    if "observations" not in payload:
        raise RuntimeError(
            f"FRED error for {series_id}: "
            f"{payload.get('error_message', 'unknown error')}"
        )
    records: list[tuple[str, float]] = []
    for obs in payload["observations"]:
        v = obs["value"]
        if v == ".":
            continue
        try:
            records.append((obs["date"], float(v)))
        except ValueError:
            pass
    if not records:
        raise RuntimeError(f"FRED returned no usable data for {series_id}")
    dates, values = zip(*records)
    return pd.Series(
        list(values),
        index=pd.to_datetime(list(dates)),
        name=series_id,
        dtype="float64",
    )


def _fetch_vix(start: str, end: str) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance is not installed — cannot fetch VIX")
    df = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("yfinance returned empty DataFrame for ^VIX")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    if "Close" not in df.columns:
        raise RuntimeError(f"No 'Close' column in VIX data; got {list(df.columns)}")
    series = df["Close"].rename("vix").astype("float64")
    series.index = pd.to_datetime(series.index)
    return series


def _align_to_index(
    raw: pd.Series,
    trading_index: pd.DatetimeIndex,
    col_name: str,
) -> tuple[pd.Series, pd.Series]:
    """Reindex raw to trading_index, record imputation flag, then ffill."""
    aligned = raw.reindex(trading_index)
    was_imputed = aligned.isna().astype("int8")
    aligned = aligned.ffill()
    aligned.name = col_name
    was_imputed.name = f"{col_name}_was_imputed"
    return aligned, was_imputed


# ── Non-vintage public function ───────────────────────────────────────────────

def load_macro_features(
    processed: pd.DataFrame,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch and align all macro features to the processed DataFrame's index.

    Uses current-vintage data only (no release-date awareness).
    For point-in-time correct data use build_macro_pit() instead.

    Args:
        processed:  Output of data/preprocessing.run_pipeline().
                    Must contain columns 'copper', 'gold', and 'tnote'.
        start:      Earliest date to request from remote APIs (ISO YYYY-MM-DD).
        end:        Latest date (defaults to today).

    Returns:
        DataFrame with the same DatetimeIndex as processed, containing all
        macro feature columns followed by their *_was_imputed flags.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    trading_index: pd.DatetimeIndex = processed.index
    api_key = _load_fred_key()

    feature_cols: dict[str, pd.Series] = {}
    flag_cols: dict[str, pd.Series] = {}

    def _safe(label: str, fn) -> pd.Series | None:
        try:
            result = fn()
            print(f"      ✓ {label}")
            return result
        except Exception as exc:
            print(f"      ✗ {label}: {exc}")
            return None

    def _register(col_name: str, raw: pd.Series | None) -> None:
        if raw is None:
            feature_cols[col_name] = pd.Series(
                np.nan, index=trading_index, name=col_name, dtype="float64"
            )
            flag_cols[f"{col_name}_was_imputed"] = pd.Series(
                1, index=trading_index, name=f"{col_name}_was_imputed", dtype="int8"
            )
        else:
            s, flag = _align_to_index(raw, trading_index, col_name)
            feature_cols[col_name] = s
            flag_cols[f"{col_name}_was_imputed"] = flag

    print("  [macro] Fetching daily FRED series...")
    for col_name, series_id in _DAILY_FRED.items():
        raw = _safe(
            f"{col_name} ({series_id})",
            lambda sid=series_id: _fetch_fred(sid, start, end, api_key),
        )
        _register(col_name, raw)
        time.sleep(0.3)

    print("  [macro] Fetching VIX (yfinance ^VIX)...")
    vix_raw = _safe("vix (^VIX)", lambda: _fetch_vix(start, end))
    _register("vix", vix_raw)

    print("  [macro] Computing derived series...")
    copper_gold_raw = (processed["copper"] / processed["gold"]).rename("copper_gold")
    _register("copper_gold", copper_gold_raw)

    if feature_cols["dgs2"].notna().any():
        spread_raw = (processed["tnote"] - feature_cols["dgs2"]).rename("spread_2s10s")
        _register("spread_2s10s", spread_raw)
    else:
        print("      ! spread_2s10s skipped (dgs2 unavailable)")
        _register("spread_2s10s", None)

    print("  [macro] Fetching monthly FRED series (with YoY / MoM transforms)...")
    for col_name, (series_id, transform) in _MONTHLY_LEVEL_FRED.items():
        def _fetch_transformed(sid=series_id, tr=transform, cname=col_name) -> pd.Series:
            level = _fetch_fred(sid, start, end, api_key)
            if tr == "yoy":
                return level.pct_change(12).mul(100).rename(cname)
            return level.pct_change(1).mul(100).rename(cname)
        raw = _safe(f"{col_name} ({series_id})", _fetch_transformed)
        _register(col_name, raw)
        time.sleep(0.3)

    print("  [macro] Fetching monthly FRED series (direct levels)...")
    for col_name, series_id in _MONTHLY_DIRECT_FRED.items():
        raw = _safe(
            f"{col_name} ({series_id})",
            lambda sid=series_id, cname=col_name: (
                _fetch_fred(sid, start, end, api_key).rename(cname)
            ),
        )
        _register(col_name, raw)
        time.sleep(0.3)

    print("  [macro] Fetching quarterly FRED series...")
    for col_name, series_id in _QUARTERLY_DIRECT_FRED.items():
        raw = _safe(
            f"{col_name} ({series_id})",
            lambda sid=series_id, cname=col_name: (
                _fetch_fred(sid, start, end, api_key).rename(cname)
            ),
        )
        _register(col_name, raw)
        time.sleep(0.3)

    feature_df = pd.DataFrame(
        {col: feature_cols[col] for col in _COL_ORDER},
        index=trading_index,
    )
    flag_df = pd.DataFrame(
        {f"{col}_was_imputed": flag_cols[f"{col}_was_imputed"] for col in _COL_ORDER},
        index=trading_index,
    )
    result = pd.concat([feature_df, flag_df], axis=1)

    print(
        f"\n  [macro] Feature matrix: "
        f"{result.shape[0]:,} rows × {result.shape[1]} cols"
    )
    nan_counts = feature_df.isna().sum()
    nan_any = nan_counts[nan_counts > 0]
    if not nan_any.empty:
        print("  [macro] NaN counts (data gaps or failed fetches):")
        for col, n in nan_any.items():
            pct = 100.0 * n / len(result)
            print(f"            {col:<22}: {n:,} NaN rows ({pct:.1f}%)")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Real-time vintage fetcher
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_realtime_series(
    fred_key: str,
    series_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch all historical vintages for a FRED series (output_type=4).

    Each row in the returned DataFrame represents one vintage observation:
        realtime_start  — date this value first became public
        realtime_end    — date this value was superseded (or 9999-01-01)
        date            — reference period (e.g. 2006-01-01 for January data)
        value           — the reported number

    Results are cached to data/cache/{series_id}_vintage.parquet.
    Delete the cache file to force a re-fetch.

    Args:
        fred_key:   FRED API key.
        series_id:  FRED series identifier (e.g. "UNRATE").
        start_date: Earliest reference period to include (ISO YYYY-MM-DD).
        end_date:   Latest real-time end date (ISO YYYY-MM-DD).

    Returns:
        DataFrame with columns [realtime_start, realtime_end, date, value].
    """
    cache_path = _CACHE_DIR / f"{series_id}_vintage.parquet"
    if cache_path.exists():
        print(f"    [cache] {series_id}: loading from {cache_path}")
        return pd.read_parquet(cache_path)

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":         series_id,
        "api_key":           fred_key,
        "file_type":         "json",
        "realtime_start":    start_date,
        "realtime_end":      end_date,
        "output_type":       4,          # all vintages
        "observation_start": start_date,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "error_message" in payload:
        raise RuntimeError(
            f"FRED API error for {series_id}: {payload['error_message']}"
        )
    if "observations" not in payload:
        raise RuntimeError(f"FRED returned no 'observations' key for {series_id}")

    df = pd.DataFrame(payload["observations"])

    # Drop missing values ("." sentinel used by FRED)
    df = df[df["value"] != "."].copy()
    df["value"] = df["value"].astype("float64")

    for col in ["realtime_start", "realtime_end", "date"]:
        df[col] = pd.to_datetime(df[col])

    df = df[["realtime_start", "realtime_end", "date", "value"]].sort_values(
        ["realtime_start", "date"]
    ).reset_index(drop=True)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"    [fetch] {series_id}: {len(df):,} vintage rows → cached")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Build point-in-time series
# ═══════════════════════════════════════════════════════════════════════════════

def build_point_in_time_series(
    vintage_df: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Map a vintage DataFrame onto the trading-day index with no lookahead.

    For each trading day t, finds the most recently published value where
    realtime_start <= t.  The "most recent" is determined first by
    realtime_start (publication date) then by date (reference period), so
    on days when multiple periods are released simultaneously, the value for
    the latest reference period is used.

    Args:
        vintage_df:   Output of fetch_realtime_series().
        trading_days: DatetimeIndex of all trading days in the study window.

    Returns:
        (pit_values, days_stale, is_revision)

        pit_values   — pd.Series[float]: point-in-time correct value for each
                       trading day.  NaN before first publication.
        days_stale   — pd.Series[float]: calendar days since the last new
                       publication available on each trading day.  NaN before
                       first publication.
        is_revision  — pd.Series[bool]: True on trading days when the
                       point-in-time value changed from the prior trading day.
    """
    trading_days = pd.DatetimeIndex(trading_days)

    # For each publication date (realtime_start), keep the row with the
    # latest reference period (date) — that is the "headline" value.
    pub_events = (
        vintage_df
        .sort_values(["realtime_start", "date"])
        .groupby("realtime_start", as_index=False)
        .agg({"date": "last", "value": "last"})   # last = latest ref period
        [["realtime_start", "value"]]
        .sort_values("realtime_start")
        .reset_index(drop=True)
    )
    # Keep pub_date as a separate column so merge_asof doesn't overwrite it
    pub_events["pub_date"] = pub_events["realtime_start"]

    # merge_asof: for each trading day, find the last publication event
    # where realtime_start <= t  (backward join)
    left = pd.DataFrame({"realtime_start": trading_days})
    merged = pd.merge_asof(
        left,
        pub_events,
        on="realtime_start",
        direction="backward",
    )
    merged.index = trading_days

    pit_values = merged["value"].copy()
    pit_values.index = trading_days
    pit_values.name = "value"

    last_pub = merged["pub_date"].copy()
    last_pub.index = trading_days

    # days_stale: calendar days between the trading day and last publication
    days_stale = pd.Series(np.nan, index=trading_days, name="days_stale")
    valid = last_pub.notna()
    if valid.any():
        days_stale.loc[valid] = (
            trading_days[valid] - pd.DatetimeIndex(last_pub.loc[valid])
        ).days.astype("float64")

    # is_revision: PIT value changed relative to the prior trading day
    shifted = pit_values.shift(1)
    is_revision = (
        pit_values.notna() & shifted.notna() & (pit_values != shifted)
    )
    is_revision.name = "is_revision"

    return pit_values, days_stale, is_revision


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Revision surprise feature
# ═══════════════════════════════════════════════════════════════════════════════

def compute_revision_surprise(vintage_df: pd.DataFrame) -> pd.Series:
    """Compute the revision surprise for each publication event.

    For each new publication of a series:
        surprise = new_headline_value - previous_headline_value

    First-release events have no prior value; their surprise is set to 0.0.

    Args:
        vintage_df: Output of fetch_realtime_series().

    Returns:
        pd.Series indexed by realtime_start (publication dates) containing
        the surprise value.  Only entries where a new publication appeared
        are included.
    """
    pub_events = (
        vintage_df
        .sort_values(["realtime_start", "date"])
        .groupby("realtime_start", as_index=False)
        .agg({"date": "last", "value": "last"})
        [["realtime_start", "value"]]
        .sort_values("realtime_start")
        .reset_index(drop=True)
    )

    pub_events["prev_value"] = pub_events["value"].shift(1)
    # First release: no previous → surprise = 0.0 (value itself is not a "surprise")
    pub_events["surprise"] = (
        pub_events["value"] - pub_events["prev_value"]
    ).fillna(0.0)

    surprise = pub_events.set_index("realtime_start")["surprise"]
    surprise.name = "revision_surprise"
    return surprise


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 helpers — Release calendar printer
# ═══════════════════════════════════════════════════════════════════════════════

def _print_release_calendar(
    series_id: str,
    vintage_df: pd.DataFrame,
    months: int = 12,
) -> None:
    """Print a human-readable release calendar for the last N months.

    Shows every publication event (first release or revision) so that the
    user can visually verify release dates before using them in the model.

    Each line has the format:
        YYYY-MM-DD: Mon YYYY value = X.XX (first release | revised from Y | confirmed)
    """
    cutoff = pd.Timestamp.today() - pd.DateOffset(months=months)
    recent = vintage_df[vintage_df["realtime_start"] >= cutoff].copy()

    if recent.empty:
        print(f"  {series_id}: no release events in the last {months} months")
        return

    # For each reference period, find its very first publication date globally
    first_pub_per_ref: dict[pd.Timestamp, pd.Timestamp] = (
        vintage_df.groupby("date")["realtime_start"].min().to_dict()
    )

    print(f"\n{series_id} release dates (last {months} months):")

    for pub_date, group in recent.groupby("realtime_start"):
        for _, row in group.sort_values("date").iterrows():
            ref_date: pd.Timestamp = row["date"]
            val: float = row["value"]
            ref_label = ref_date.strftime("%b %Y")

            if first_pub_per_ref.get(ref_date) == pub_date:
                note = "(first release)"
            else:
                # Find the value from the immediately prior publication of this period
                prior = vintage_df[
                    (vintage_df["date"] == ref_date)
                    & (vintage_df["realtime_start"] < pub_date)
                ]
                if prior.empty:
                    note = "(revision)"
                else:
                    prev_val = prior.sort_values("realtime_start").iloc[-1]["value"]
                    if val != prev_val:
                        note = f"(revised from {prev_val:.4g})"
                    else:
                        note = "(confirmed, no revision)"

            print(f"  {pub_date.date()}: {ref_label} value = {val:.4g} {note}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 helpers — Lookahead audit printer
# ═══════════════════════════════════════════════════════════════════════════════

def _print_lookahead_audit(
    pit_results: dict[str, dict],
    trading_days: pd.DatetimeIndex,
    n_dates: int = 3,
    seed: int = 42,
) -> None:
    """Print a lookahead audit for n_dates randomly selected trading days.

    For each audited date, prints the point-in-time value, the publication
    date, and the staleness in days for every series.  Confirms that all
    publication dates are strictly <= the audited trading day.
    """
    rng = np.random.default_rng(seed)
    eligible = trading_days[trading_days.year >= 2020]
    if len(eligible) == 0:
        eligible = trading_days
    chosen_idx = rng.choice(len(eligible), size=min(n_dates, len(eligible)), replace=False)
    audit_dates = sorted(eligible[chosen_idx])

    for audit_date in audit_dates:
        print(f"\n=== LOOKAHEAD AUDIT: {audit_date.date()} ===")
        violations: list[str] = []

        for series_id, res in pit_results.items():
            pit_val   = res["pit_values"].get(audit_date, np.nan)
            stale     = res["days_stale"].get(audit_date, np.nan)
            last_pub  = res["last_pub"].get(audit_date, pd.NaT)

            if pd.isna(pit_val):
                print(f"  {series_id:<20}: N/A (no data published yet)")
                continue

            stale_str = f"{int(stale)} days ago" if pd.notna(stale) else "unknown"
            pub_str   = last_pub.date() if pd.notna(last_pub) else "N/A"
            print(
                f"  {series_id:<20}: {pit_val:<10.4g} "
                f"(published {pub_str}, {stale_str})"
            )

            if pd.notna(last_pub) and last_pub > audit_date:
                violations.append(series_id)

        if violations:
            print(f"  *** LOOKAHEAD DETECTED for: {violations} ***")
        else:
            print(f"  All values confirmed available before {audit_date.date()} ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — Main PIT builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_macro_pit(
    processed: pd.DataFrame,
    start_date: str = "2010-01-01",
    end_date: str | None = None,
    force_refetch: bool = False,
) -> pd.DataFrame:
    """Build a point-in-time correct macro feature matrix.

    Fetches all historical FRED vintages for the series in _PIT_SERIES,
    constructs no-lookahead feature columns, prints release calendars and a
    lookahead audit, then writes data/macro_pit.parquet.

    Args:
        processed:     Output of data/preprocessing.run_pipeline().
                       Provides the trading-day index.
        start_date:    Earliest date for FRED observations (ISO YYYY-MM-DD).
        end_date:      Latest real-time date (defaults to today).
        force_refetch: If True, delete cache files before fetching.

    Output columns per series:
        {series_id}_value      — point-in-time correct value (NaN before first pub)
        {series_id}_days_stale — calendar days since last publication
        {series_id}_revision   — surprise on release/revision days, 0.0 otherwise

    Returns:
        DataFrame indexed by trading days with the above columns for all series.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    fred_key     = _load_fred_key()
    trading_days = pd.DatetimeIndex(processed.index)

    pit_results: dict[str, dict] = {}   # series_id → {pit_values, days_stale, ...}
    surprises:   dict[str, pd.Series] = {}

    print("[macro_pit] Fetching real-time vintage data from FRED...")
    print(f"            Date range : {start_date} → {end_date}")
    print(f"            Series     : {', '.join(_PIT_SERIES)}\n")

    for series_id, description in _PIT_SERIES.items():
        print(f"  ── {series_id}  ({description})")

        cache_path = _CACHE_DIR / f"{series_id}_vintage.parquet"
        if force_refetch and cache_path.exists():
            cache_path.unlink()

        try:
            vintage_df = fetch_realtime_series(
                fred_key, series_id, start_date, end_date
            )

            pit_values, days_stale, is_revision = build_point_in_time_series(
                vintage_df, trading_days
            )
            surprise = compute_revision_surprise(vintage_df)

            # Resolve pub_date per trading day for the audit
            pub_events = (
                vintage_df
                .sort_values(["realtime_start", "date"])
                .groupby("realtime_start", as_index=False)
                .agg({"value": "last"})
                [["realtime_start"]]
            )
            pub_events["pub_date"] = pub_events["realtime_start"]
            left = pd.DataFrame({"realtime_start": trading_days})
            merged_pub = pd.merge_asof(
                left, pub_events, on="realtime_start", direction="backward"
            )
            last_pub_series = merged_pub["pub_date"].set_axis(trading_days)

            pit_results[series_id] = {
                "pit_values": pit_values,
                "days_stale": days_stale,
                "is_revision": is_revision,
                "last_pub":    last_pub_series,
            }
            surprises[series_id] = surprise

            n_vintages = vintage_df["realtime_start"].nunique()
            n_revisions = int(is_revision.sum())
            print(
                f"    ✓  {n_vintages} publication vintages, "
                f"{n_revisions} revision days on trading calendar"
            )

            # ── PART 3: Release calendar for last 12 months ───────────────
            _print_release_calendar(series_id, vintage_df, months=12)

        except Exception as exc:
            print(f"    ✗  {exc}")
            nan_series  = pd.Series(np.nan,  index=trading_days, dtype="float64")
            bool_series = pd.Series(False,   index=trading_days, dtype="bool")
            nat_series  = pd.Series(pd.NaT,  index=trading_days, dtype="datetime64[ns]")
            pit_results[series_id] = {
                "pit_values": nan_series.rename("value"),
                "days_stale": nan_series.rename("days_stale"),
                "is_revision": bool_series.rename("is_revision"),
                "last_pub":    nat_series,
            }
            surprises[series_id] = pd.Series(dtype="float64")

        time.sleep(0.3)

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    frames: list[pd.Series] = []

    for series_id in _PIT_SERIES:
        res      = pit_results[series_id]
        surprise = surprises.get(series_id, pd.Series(dtype="float64"))

        # Map publication-date surprises onto trading days
        # (use the first trading day >= each publication date)
        revision_col = pd.Series(0.0, index=trading_days, name=f"{series_id}_revision")
        if len(surprise) > 0:
            pub_dates_arr = np.array(surprise.index, dtype="datetime64[ns]")
            td_arr        = np.array(trading_days,   dtype="datetime64[ns]")
            positions     = np.searchsorted(td_arr, pub_dates_arr, side="left")
            for pos, surp_val in zip(positions, surprise.values):
                if pos < len(trading_days):
                    revision_col.iloc[pos] += surp_val

        frames.append(res["pit_values"].rename(f"{series_id}_value"))
        frames.append(res["days_stale"].rename(f"{series_id}_days_stale"))
        frames.append(revision_col)

    result = pd.concat(frames, axis=1)

    # ── PART 5: Lookahead audit ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("LOOKAHEAD AUDIT (3 random trading days ≥ 2020)")
    print("─" * 60)
    _print_lookahead_audit(pit_results, trading_days)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path("data/macro_pit.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path)

    print(f"\n[macro_pit] Saved → {out_path}  shape {result.shape}")
    nan_counts = result[[c for c in result.columns if c.endswith("_value")]].isna().sum()
    nan_any = nan_counts[nan_counts > 0]
    if not nan_any.empty:
        print("  NaN counts in *_value columns (pre-first-publication gaps):")
        for col, n in nan_any.items():
            pct = 100.0 * n / len(result)
            print(f"    {col:<35}: {n:,} rows ({pct:.1f}%)")

    return result
