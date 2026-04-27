"""Train one TemporalMLP expert per market regime (1-5) on oil residual targets.

Windows are extracted from the *full* merged array so each 60-day lookback is
always available even when regime rows are non-contiguous in time.
Checkpoints are saved in BaseExpert format and can be reloaded with
BaseExpert().load().
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.experts.base_expert import TemporalMLP

# constants -------------------------------------------------------------------

FEATURE_COLS = [
    "oil_zscore", "gold_zscore", "copper_zscore",
    "spx_zscore", "tnote_zscore", "usd_zscore",
]
TARGET_COL     = "oil_residual"
SEQ_LEN        = 60
HIDDEN         = (16, 8)
MAX_EPOCHS     = 200
PATIENCE       = 20
BATCH_SIZE     = 256
LR             = 3e-4
CHECKPOINT_DIR = Path("models/checkpoints")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helpers ---------------------------------------------------------------------


def _extract_windows(
    X_norm: np.ndarray, y: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extract flattened 60-day windows at the given (full-array) indices."""
    Xw = np.stack(
        [X_norm[i - SEQ_LEN : i].reshape(-1) for i in indices], axis=0
    ).astype(np.float32)
    yw = y[indices].astype(np.float32)
    return Xw, yw


def _train_one(
    Xw_tr: np.ndarray,
    yw_tr: np.ndarray,
    Xw_val: np.ndarray,
    yw_val: np.ndarray,
) -> TemporalMLP:
    input_dim = Xw_tr.shape[1]
    model = TemporalMLP(input_dim, HIDDEN).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=MAX_EPOCHS, eta_min=LR * 0.01
    )
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xw_tr), torch.from_numpy(yw_tr)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    Xv = torch.from_numpy(Xw_val).to(DEVICE)
    yv = torch.from_numpy(yw_val).to(DEVICE)

    best_val = math.inf
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()

        train_loss = epoch_loss / len(Xw_tr)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(
                f"    epoch {epoch:>4}  train={train_loss:.5f}  val={val_loss:.5f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if no_improve >= PATIENCE:
            print(f"    early stop at epoch {epoch} (best val={best_val:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# main ------------------------------------------------------------------------


def main() -> None:
    print(f"Device: {DEVICE}\n")

    labeled = pd.read_parquet("data/labeled.parquet")
    targets = pd.read_parquet("data/targets.parquet")
    merged  = labeled.join(targets[[TARGET_COL]], how="inner")
    merged  = merged.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X      = merged[FEATURE_COLS].values          # (N, 6)
    y      = merged[TARGET_COL].values            # (N,)
    regime = merged["regime"].values.astype(int)  # (N,)
    N      = len(merged)

    print(f"Merged dataset: {N:,} rows after dropna\n")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    experts: dict[int, TemporalMLP] = {}
    norm_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    test_windows: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for r in range(1, 6):
        usable = np.where((regime == r) & (np.arange(N) >= SEQ_LEN))[0]

        print(f"-- Regime {r} -------------------------------------------")
        print(f"  Usable windows : {len(usable):,}")

        if len(usable) < 2 * SEQ_LEN:
            print(f"  WARNING: only {len(usable)} usable rows -- results may be unreliable.")
        if len(usable) < 10:
            print(f"  SKIP: too few samples to train.\n")
            continue

        n_tr = int(0.70 * len(usable))
        n_vl = int(0.15 * len(usable))
        train_idx = usable[:n_tr]
        val_idx   = usable[n_tr : n_tr + n_vl]
        test_idx  = usable[n_tr + n_vl :]

        print(f"  Train/Val/Test : {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

        col_mean = X[train_idx].mean(axis=0)
        col_std  = X[train_idx].std(axis=0) + 1e-8
        X_norm   = (X - col_mean) / col_std

        norm_stats[r] = (col_mean, col_std)

        Xw_test_raw = np.stack(
            [X[i - SEQ_LEN : i].reshape(-1) for i in test_idx], axis=0
        ).astype(np.float32)
        test_windows[r] = (Xw_test_raw, y[test_idx].astype(np.float32))

        Xw_tr, yw_tr   = _extract_windows(X_norm, y, train_idx)
        Xw_val, yw_val = _extract_windows(X_norm, y, val_idx)

        print(f"  Training {len(Xw_tr)} windows (input_dim={Xw_tr.shape[1]})...")
        model = _train_one(Xw_tr, yw_tr, Xw_val, yw_val)
        experts[r] = model

        with torch.no_grad():
            Xv = torch.from_numpy(Xw_val).to(DEVICE)
            yv = torch.from_numpy(yw_val).to(DEVICE)
            val_mse = nn.MSELoss()(model(Xv), yv).item()
        print(f"  Final val MSE  : {val_mse:.6f}")

        ckpt_path = CHECKPOINT_DIR / f"expert_{r}.pt"
        torch.save(
            {
                "state_dict":   model.state_dict(),
                "input_dim":    SEQ_LEN * len(FEATURE_COLS),
                "hidden":       HIDDEN,
                "col_mean":     col_mean,
                "col_std":      col_std,
                "seq_len":      SEQ_LEN,
                "feature_cols": FEATURE_COLS,
            },
            ckpt_path,
        )
        print(f"  Saved to {ckpt_path}\n")

    # 5x5 cross-regime evaluation ---------------------------------------------
    trained = sorted(experts.keys())
    print("-- 5x5 Cross-Regime MSE -----------------------------------------")
    mse_matrix = np.full((5, 5), np.nan)
    loss_fn = nn.MSELoss()

    for i in trained:
        model_i = experts[i]
        col_mean_i, col_std_i = norm_stats[i]
        # tile per-column stats to match the flattened window layout: (SEQ_LEN*6,)
        mean_flat = np.tile(col_mean_i, SEQ_LEN).astype(np.float32)
        std_flat  = np.tile(col_std_i,  SEQ_LEN).astype(np.float32)

        for j in test_windows:
            Xw_raw_j, yw_j = test_windows[j]
            if len(Xw_raw_j) == 0:
                continue
            Xw_norm_j = (Xw_raw_j - mean_flat) / std_flat
            with torch.no_grad():
                Xv = torch.from_numpy(Xw_norm_j).to(DEVICE)
                yv = torch.from_numpy(yw_j).to(DEVICE)
                mse_matrix[i - 1, j - 1] = loss_fn(model_i(Xv), yv).item()

    print("Expert\\Test" + "".join(f"  Regime {j}" for j in range(1, 6)))
    for i in range(5):
        vals = "".join(
            f"  {mse_matrix[i, j]:8.5f}" if not np.isnan(mse_matrix[i, j]) else "       nan"
            for j in range(5)
        )
        print(f"  Regime {i+1}  " + vals)
    print()

    fig, ax = plt.subplots(figsize=(7, 5))
    valid = mse_matrix[~np.isnan(mse_matrix)]
    im = ax.imshow(mse_matrix, cmap="viridis_r", aspect="auto",
                   vmin=valid.min() if len(valid) else 0,
                   vmax=valid.max() if len(valid) else 1)
    plt.colorbar(im, ax=ax, label="MSE")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"Regime {j}" for j in range(1, 6)])
    ax.set_yticklabels([f"Expert {i}" for i in range(1, 6)])
    ax.set_xlabel("Test regime")
    ax.set_ylabel("Expert trained on regime")
    ax.set_title("Cross-Regime MSE (lower = better)")

    threshold = valid.mean() if len(valid) else 0.5
    for i in range(5):
        for j in range(5):
            if not np.isnan(mse_matrix[i, j]):
                color = "white" if mse_matrix[i, j] < threshold else "black"
                ax.text(j, i, f"{mse_matrix[i, j]:.4f}",
                        ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    out_path = Path("data/cross_regime_mse.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {out_path}")


if __name__ == "__main__":
    main()
