"""Build residual forecasting targets for commodity price prediction.

For each asset and each day t:
    forward_return[t]  = (price[t+126] / price[t]) - 1
    ar_prediction[t]   = AR(5) fitted on forward_return[0..t-1] only
    residual[t]        = forward_return[t] - ar_prediction[t]

The AR(5) uses an expanding window (refitted at every t) so no future
return observations ever enter the training set.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ASSETS = ["oil", "gold", "copper"]


def build_targets(
    df: pd.DataFrame,
    assets: list[str] | None = None,
    horizon: int = 126,
    ar_lags: int = 5,
    min_train: int = 252,
) -> pd.DataFrame:
    """Compute forward returns, AR(5) predictions, and residual targets.

    Args:
        df:        DataFrame with raw price columns (at minimum: oil, gold, copper).
        assets:    Which price columns to process. Defaults to oil, gold, copper.
        horizon:   Forward return horizon in trading days (default 126 ≈ 6 months).
        ar_lags:   Number of autoregressive lags (default 5).
        min_train: Minimum number of clean observations required before the first
                   AR fit. Predictions before this threshold are NaN (default 252).

    Returns:
        DataFrame indexed by date with columns:
            {asset}_fwd_ret   — raw 126-day forward return
            {asset}_ar_pred   — expanding-window AR(5) prediction
            {asset}_residual  — residual target (fwd_ret - ar_pred)
    """
    if assets is None:
        assets = ASSETS

    result_cols: dict[str, pd.Series] = {}

    for asset in assets:
        print(f"      Building targets for {asset}...")
        prices = df[asset]

        # Forward return at t: (price[t+horizon] / price[t]) - 1
        # pct_change(horizon) computes (price[t] - price[t-horizon]) / price[t-horizon]
        # shift(-horizon) maps that back so index t holds the return *starting* at t
        fwd_ret = prices.pct_change(horizon).shift(-horizon)

        # AR(5) features: lag the forward-return series by 1..ar_lags
        lags = pd.concat(
            {f"lag_{k}": fwd_ret.shift(k) for k in range(1, ar_lags + 1)},
            axis=1,
        )

        # Rows where both fwd_ret and all lags are finite
        valid = fwd_ret.notna() & lags.notna().all(axis=1)
        X_valid = lags.loc[valid].values          # (n_valid, ar_lags)
        y_valid = fwd_ret.loc[valid].values       # (n_valid,)
        idx_valid = df.index[valid]               # DatetimeIndex for valid rows

        ar_pred = pd.Series(np.nan, index=df.index, name=f"{asset}_ar_pred")

        n_valid = len(X_valid)
        for i in range(min_train, n_valid):
            model = LinearRegression(fit_intercept=True)
            model.fit(X_valid[:i], y_valid[:i])
            ar_pred.loc[idx_valid[i]] = float(model.predict(X_valid[[i]])[0])

        result_cols[f"{asset}_fwd_ret"] = fwd_ret
        result_cols[f"{asset}_ar_pred"] = ar_pred
        result_cols[f"{asset}_residual"] = fwd_ret - ar_pred

        n_filled = ar_pred.notna().sum()
        print(f"        {n_filled:,} predictions made  "
              f"({n_valid - min_train} valid rows after {min_train}-obs warmup)")

    return pd.DataFrame(result_cols, index=df.index)


def plot_target_distributions(
    targets: pd.DataFrame,
    assets: list[str] | None = None,
    output_dir: str | Path = "data",
) -> None:
    """Plot raw forward-return vs residual distributions for each asset.

    Overlaid histograms (density-normalised) show how the AR(5) removal
    affects shape and spread. Tighter, more centred residuals confirm that
    the AR model captured serial structure.

    File written:
        <output_dir>/target_distributions.png
    """
    if assets is None:
        assets = ASSETS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 4 * len(assets)))
    if len(assets) == 1:
        axes = [axes]

    for ax, asset in zip(axes, assets):
        raw = targets[f"{asset}_fwd_ret"].dropna()
        res = targets[f"{asset}_residual"].dropna()

        bins = np.linspace(
            min(raw.quantile(0.005), res.quantile(0.005)),
            max(raw.quantile(0.995), res.quantile(0.995)),
            60,
        )

        ax.hist(raw.values, bins=bins, density=True, alpha=0.55,
                color="#3498db", label="Raw 126-day return")
        ax.hist(res.values, bins=bins, density=True, alpha=0.55,
                color="#e74c3c", label="Residual (AR-adjusted)")

        raw_std = raw.std()
        res_std = res.std()
        reduction_pct = 100.0 * (raw_std - res_std) / raw_std

        ax.set_title(
            f"{asset.upper()}  —  σ raw={raw_std:.4f}  "
            f"σ residual={res_std:.4f}  "
            f"(σ reduction {reduction_pct:+.1f}%)",
            fontsize=11,
        )
        ax.set_xlabel("Return", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate autocorrelation stats
        ac1_raw = raw.autocorr(lag=1)
        ac1_res = res.autocorr(lag=1)
        ax.text(
            0.98, 0.92,
            f"AC(1) raw={ac1_raw:.3f}   AC(1) residual={ac1_res:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8.5,
            color="#555555",
        )

    fig.suptitle(
        "Forward Return Distribution: Raw vs AR(5)-Adjusted Residual",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    path = output_dir / "target_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {path}")
