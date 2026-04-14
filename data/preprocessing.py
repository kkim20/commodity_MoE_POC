import numpy as np
import pandas as pd


def build_unified_index(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Return the sorted union of all individual asset date indexes."""
    unified = frames[next(iter(frames))].index
    for df in frames.values():
        unified = unified.union(df.index)
    return unified.sort_values()


def align_and_impute(
    frames: dict[str, pd.DataFrame],
    unified_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reindex all assets to unified_index, forward fill, and record imputation flags.

    The was_imputed flag is computed BEFORE forward filling so that filled rows
    are distinguishable from originally observed rows.
    """
    price_cols: dict[str, pd.Series] = {}
    flag_cols: dict[str, pd.Series] = {}

    for asset, df in frames.items():
        series = df[asset].reindex(unified_index)

        # Record which positions are NaN *before* filling — these will be imputed
        was_imputed = series.isna().astype("int8")

        # Forward fill only; no bfill to avoid lookahead bias at series start
        series = series.ffill()

        price_cols[asset] = series
        flag_cols[f"{asset}_was_imputed"] = was_imputed

    prices_df = pd.DataFrame(price_cols, index=unified_index)
    flags_df = pd.DataFrame(flag_cols, index=unified_index)
    return pd.concat([prices_df, flags_df], axis=1)


def add_rolling_zscores(
    df: pd.DataFrame,
    asset_names: list[str],
    window: int = 252,
    min_periods: int = 30,
) -> pd.DataFrame:
    """Add {asset}_zscore columns using a rolling z-score (no lookahead bias).

    Uses right-aligned rolling windows (pandas default). center=True is never
    used as it would incorporate future data.
    """
    zscore_cols: dict[str, pd.Series] = {}

    for asset in asset_names:
        series = df[asset]
        rolling = series.rolling(window=window, min_periods=min_periods)
        roll_mean = rolling.mean()
        roll_std = rolling.std()

        zscore = (series - roll_mean) / roll_std

        # Replace inf (zero-std edge case) with NaN
        n_inf = np.isinf(zscore).sum()
        if n_inf > 0:
            print(
                f"      WARNING: {asset}_zscore has {n_inf} inf value(s) "
                f"(zero rolling std) — replaced with NaN"
            )
            zscore = zscore.replace([np.inf, -np.inf], np.nan)

        zscore_cols[f"{asset}_zscore"] = zscore

    zscore_df = pd.DataFrame(zscore_cols, index=df.index)
    return pd.concat([df, zscore_df], axis=1)


def run_pipeline(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Full preprocessing pipeline: align → impute → rolling z-score."""
    asset_names = list(frames.keys())

    unified_index = build_unified_index(frames)
    print(f"[2/3] Preprocessing...")
    print(f"      Unified index : {len(unified_index):,} trading days")

    df = align_and_impute(frames, unified_index)

    # Imputation summary
    print(f"      Imputation summary:")
    for asset in asset_names:
        flag_col = f"{asset}_was_imputed"
        n_filled = int(df[flag_col].sum())
        pct = 100.0 * n_filled / len(df)
        print(f"        {flag_col:<28}: {n_filled:5,} filled values ({pct:.1f}%)")

    df = add_rolling_zscores(df, asset_names)

    # Z-score NaN summary
    print(f"      Z-score NaN count (first ~{30} rows expected):")
    for asset in asset_names:
        n_nan = int(df[f"{asset}_zscore"].isna().sum())
        print(f"        {asset}_zscore   : {n_nan} NaN rows")

    print()
    return df
