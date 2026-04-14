from pathlib import Path

import pandas as pd


def load_all(raw_dir: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    """Load all CSV files from raw_dir.

    Each CSV must have columns: date, close.
    Returns a dict keyed by asset name (filename stem), where each value is a
    single-column DataFrame with a DatetimeIndex named 'date'.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir.resolve()}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir.resolve()}")

    print(f"[1/3] Loading raw data from {raw_dir}/")

    frames: dict[str, pd.DataFrame] = {}
    for path in csv_files:
        asset = path.stem  # e.g. "oil", "gold"
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df.index.name = "date"
        df = df[["close"]].rename(columns={"close": asset})
        df = df.sort_index()
        frames[asset] = df
        print(
            f"      {asset:<8}: {len(df):,} rows  "
            f"({df.index[0].date()} to {df.index[-1].date()})"
        )

    print(f"      {len(frames)} assets loaded.\n")
    return frames
