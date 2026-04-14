"""Macro regime labeling based on 20-day rolling direction of 10y yield and SPX."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REGIME_NAMES = {
    1: "Inflationary Growth",
    2: "Stagflation / Risk-off",
    3: "Deflationary Growth",
    4: "Recession / Crisis",
    5: "Transitional",
}

REGIME_COLORS = {
    1: "#2ecc71",  # green
    2: "#e74c3c",  # red
    3: "#3498db",  # blue
    4: "#8e44ad",  # purple
    5: "#bdc3c7",  # light gray
}


def label_regimes(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Label each trading day with a macro regime using only past information.

    Direction is determined by the sign of the rolling `window`-day return.
    pct_change(window) is right-aligned, so value at time t uses only
    price[t] and price[t-window] — no lookahead bias.

    Regime assignment:
        1 - Inflationary Growth:   tnote rising  & spx rising
        2 - Stagflation/Risk-off:  tnote rising  & spx falling
        3 - Deflationary Growth:   tnote falling & spx rising
        4 - Recession/Crisis:      tnote falling & spx falling
        5 - Transitional:          either direction is zero or undetermined
                                   (includes the first `window` rows where
                                   history is insufficient)

    Args:
        df:     DataFrame with at least 'tnote' and 'spx' price columns.
        window: Look-back window in trading days for direction (default 20).

    Returns:
        Copy of df with an additional 'regime' column (int8).
    """
    tnote_dir = np.sign(df["tnote"].pct_change(window))
    spx_dir = np.sign(df["spx"].pct_change(window))

    # Default: Transitional (covers flat, NaN, or ambiguous cases)
    regime = pd.Series(5, index=df.index, dtype="int8", name="regime")

    # NaN comparisons evaluate to False in pandas, so first `window` rows
    # correctly remain 5 without any special-casing.
    regime[(tnote_dir > 0) & (spx_dir > 0)] = 1
    regime[(tnote_dir > 0) & (spx_dir < 0)] = 2
    regime[(tnote_dir < 0) & (spx_dir > 0)] = 3
    regime[(tnote_dir < 0) & (spx_dir < 0)] = 4

    return df.assign(regime=regime)


def plot_regimes(df: pd.DataFrame, output_dir: str | Path = "data") -> None:
    """Save two plots: regime distribution bar chart and SPX timeline with regime shading.

    Files written:
        <output_dir>/regime_distribution.png
        <output_dir>/regime_timeline.png

    Args:
        df:         DataFrame with 'regime', 'spx', and 'tnote' columns.
        output_dir: Directory to write PNG files (created if absent).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_distribution(df, output_dir)
    _plot_timeline(df, output_dir)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _plot_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["regime"].value_counts().sort_index()
    total = counts.sum()
    pcts = 100.0 * counts / total

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        [REGIME_NAMES[r] for r in counts.index],
        counts.values,
        color=[REGIME_COLORS[r] for r in counts.index],
        edgecolor="white",
        height=0.6,
    )

    # Annotate each bar with count and percentage
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({pct:.1f}%)",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Trading days", fontsize=11)
    ax.set_title("Macro Regime Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, counts.max() * 1.22)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = output_dir / "regime_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"      Saved: {path}")


def _plot_timeline(df: pd.DataFrame, output_dir: Path) -> None:
    fig, (ax_spx, ax_tnote) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    _shade_regime_bands(ax_spx, df)
    _shade_regime_bands(ax_tnote, df)

    ax_spx.plot(df.index, df["spx"], color="#1a1a2e", linewidth=0.8, label="SPX")
    ax_spx.set_ylabel("SPX", fontsize=10)
    ax_spx.set_title("Macro Regime Timeline", fontsize=13, fontweight="bold", pad=12)

    ax_tnote.plot(df.index, df["tnote"], color="#16213e", linewidth=0.8, label="10y Yield")
    ax_tnote.set_ylabel("10y Yield", fontsize=10)
    ax_tnote.set_xlabel("Date", fontsize=10)

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], label=f"{r}. {REGIME_NAMES[r]}")
        for r in sorted(REGIME_NAMES)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    for ax in (ax_spx, ax_tnote):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    path = output_dir / "regime_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {path}")


def _shade_regime_bands(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Fill axis background with colored bands for each consecutive regime run."""
    dates = df.index
    regimes = df["regime"].values

    prev_regime = regimes[0]
    seg_start = dates[0]

    for i in range(1, len(dates)):
        if regimes[i] != prev_regime:
            ax.axvspan(
                seg_start, dates[i],
                alpha=0.25,
                color=REGIME_COLORS[int(prev_regime)],
                linewidth=0,
            )
            prev_regime = regimes[i]
            seg_start = dates[i]

    # Close the final segment
    ax.axvspan(
        seg_start, dates[-1],
        alpha=0.25,
        color=REGIME_COLORS[int(prev_regime)],
        linewidth=0,
    )
