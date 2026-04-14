from pathlib import Path

from data.loader import load_all
from data.preprocessing import run_pipeline
from utils.regime_labels import label_regimes, plot_regimes
from utils.targets import build_targets, plot_target_distributions


def main() -> None:
    raw_dir = Path("data/raw")
    processed_path = Path("data/processed.parquet")
    labeled_path = Path("data/labeled.parquet")

    # ── Step 1-2: Load and preprocess ────────────────────────────────────────
    frames = load_all(raw_dir)
    processed = run_pipeline(frames)

    # ── Step 3: Save processed data ───────────────────────────────────────────
    print(f"[3/4] Saving processed data to {processed_path}")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(processed_path, index=True)
    print(f"      Output shape : {processed.shape}")
    print(f"      Columns      : {', '.join(processed.columns.tolist())}\n")

    # ── Step 4: Regime labeling ───────────────────────────────────────────────
    print(f"[4/4] Labeling regimes...")
    labeled = label_regimes(processed, window=20)

    dist = labeled["regime"].value_counts().sort_index()
    total = len(labeled)
    for regime, count in dist.items():
        print(f"      Regime {regime}: {count:>5,} days ({100*count/total:.1f}%)")

    plot_regimes(labeled, output_dir="data")

    labeled.to_parquet(labeled_path, index=True)
    print(f"      Labeled data  → {labeled_path}  shape {labeled.shape}\n")

    # ── Step 5: Residual targets ──────────────────────────────────────────────
    targets_path = Path("data/targets.parquet")
    print(f"[5/5] Building AR(5) residual targets (expanding window)...")
    targets = build_targets(labeled)
    plot_target_distributions(targets, output_dir="data")
    targets.to_parquet(targets_path, index=True)
    print(f"      Targets       → {targets_path}  shape {targets.shape}")
    print(f"      Done.")


if __name__ == "__main__":
    main()
