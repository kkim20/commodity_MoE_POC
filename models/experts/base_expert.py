"""Temporal MLP expert: forecasts 6-month residual return from a 60-day window.

Parameter budget note
---------------------
With seq_len=60 the flattened input_dim = 60 × n_features. The first Linear layer
dominates:  60×6×64 ≈ 23k params (6 features), 60×12×64 ≈ 46k (12 features).
Use hidden=(16,8) instead of (64,32) to stay under 10k for up to ~10 features.
The default hidden=(64,32) is kept for flexibility; override via `hidden` arg.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BaseExpert:
    """Lightweight CPU-only temporal MLP for one commodity's residual forecast.

    Parameters
    ----------
    seq_len : int
        Number of look-back days used as input (default 60).
    feature_cols : list[str]
        Column names to use from the input DataFrame.
    lr : float
        Initial AdamW learning rate.
    max_epochs : int
        Hard cap on training epochs.
    patience : int
        Early-stopping patience (epochs without val-loss improvement).
    batch_size : int
    """

    def __init__(
        self,
        seq_len: int = 60,
        feature_cols: Optional[list[str]] = None,
        hidden: tuple[int, int] = (64, 32),
        lr: float = 3e-4,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.hidden = hidden
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device or _DEVICE
        self.model: Optional[TemporalMLP] = None
        self._col_mean: Optional[np.ndarray] = None
        self._col_std: Optional[np.ndarray] = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_windows(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a window of length seq_len over X, aligning with y."""
        n, d = X.shape
        rows, targets = [], []
        for i in range(self.seq_len, n):
            rows.append(X[i - self.seq_len : i].reshape(-1))  # flatten window
            targets.append(y[i])
        return np.array(rows, dtype=np.float32), np.array(targets, dtype=np.float32)

    def _normalise(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._col_mean = X.mean(axis=0)
            self._col_std = X.std(axis=0) + 1e-8
        return (X - self._col_mean) / self._col_std

    # ── public API ────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train with MSE loss, AdamW, cosine LR schedule, early stopping.

        Parameters
        ----------
        X_train, X_val : ndarray of shape (n_samples, n_features)
            Raw (un-windowed) feature arrays.
        y_train, y_val : ndarray of shape (n_samples,)
            Residual target aligned to the *last* row of each window.

        Returns
        -------
        history : dict with 'train_loss' and 'val_loss' lists.
        """
        X_train = self._normalise(X_train, fit=True)
        X_val = self._normalise(X_val, fit=False)

        Xw_tr, yw_tr = self._build_windows(X_train, y_train)
        Xw_val, yw_val = self._build_windows(X_val, y_val)

        input_dim = Xw_tr.shape[1]
        self.model = TemporalMLP(input_dim, self.hidden).to(self.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=self.lr * 0.01
        )
        loss_fn = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(Xw_tr), torch.from_numpy(yw_tr)
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        Xv = torch.from_numpy(Xw_val).to(self.device)
        yv = torch.from_numpy(yw_val).to(self.device)

        best_val = math.inf
        best_state = None
        no_improve = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item() * len(xb)
            scheduler.step()

            train_loss = epoch_loss / len(Xw_tr)
            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(Xv), yv).item()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if verbose and epoch % 20 == 0:
                print(
                    f"  epoch {epoch:>4}  train={train_loss:.5f}  val={val_loss:.5f}"
                    f"  lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if no_improve >= self.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val={best_val:.5f})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return point-estimate forecasts aligned to rows seq_len..n-1.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Raw features; first seq_len rows are consumed as warm-up.

        Returns
        -------
        preds : ndarray of shape (n_samples - seq_len,)
        """
        if self.model is None:
            raise RuntimeError("Call train() before predict().")
        X_norm = self._normalise(X, fit=False)
        windows = np.stack(
            [X_norm[i - self.seq_len : i].reshape(-1) for i in range(self.seq_len, len(X))],
            axis=0,
        ).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.from_numpy(windows).to(self.device)).cpu().numpy()
        return preds

    def save(self, path: str | Path) -> None:
        """Save model weights and normalisation stats to a .pt checkpoint."""
        if self.model is None:
            raise RuntimeError("Nothing to save — model has not been trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": next(self.model.parameters()).shape[1],
                "hidden": self.hidden,
                "col_mean": self._col_mean,
                "col_std": self._col_std,
                "seq_len": self.seq_len,
                "feature_cols": self.feature_cols,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Restore a checkpoint saved by save()."""
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.seq_len = ckpt["seq_len"]
        self.feature_cols = ckpt["feature_cols"]
        self.hidden = ckpt.get("hidden", (64, 32))
        self._col_mean = ckpt["col_mean"]
        self._col_std = ckpt["col_std"]
        self.model = TemporalMLP(ckpt["input_dim"], self.hidden).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @property
    def n_params(self) -> int:
        if self.model is None:
            raise RuntimeError("Model not yet built.")
        return sum(p.numel() for p in self.model.parameters())
