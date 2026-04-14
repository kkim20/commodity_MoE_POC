# Commodity Price Forecasting — MoE System

## Project Goal

Forecast 6-month forward returns for oil, gold, and copper using a
Mixture-of-Experts (MoE) architecture that is macro-regime-aware.
Each expert will specialize in a different market regime; a gating
network routes each day's feature vector to the appropriate expert(s).

## Stack

- Python 3.11.9 (`.venv/`)
- Core: `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- No deep-learning libraries yet — Phase 2 will add them

## How to Run

```bash
# Activate environment (Windows)
.venv/Scripts/activate

# Fetch fresh raw data (writes to data/raw/*.csv)
python data/fetch.py

# Full preprocessing + labeling + target pipeline
python pipeline.py
```

`pipeline.py` is the single entry point. It runs all 5 steps in order and
writes three output files: `data/processed.parquet`, `data/labeled.parquet`,
`data/targets.parquet`.

## Project Structure

```
├── data/
│   ├── fetch.py            — yfinance → FRED → Stooq fallback data fetcher
│   ├── loader.py           — load_all(): reads raw CSVs → dict of DataFrames
│   └── preprocessing.py    — run_pipeline(): align, forward-fill, rolling z-score
├── models/
│   ├── experts/            — individual expert models (Phase 2)
│   ├── gating/             — gating network (Phase 2)
│   └── explainability/     — SHAP / attention maps (Phase 3)
├── utils/
│   ├── regime_labels.py    — label_regimes(), plot_regimes()
│   └── targets.py          — build_targets(), plot_target_distributions()
├── evaluation/             — metrics, backtesting (Phase 2+)
└── pipeline.py             — orchestrator: steps 1–5
```

## Phase 1 — Data Preparation (COMPLETE)

### Assets

6 daily series from 2010-01-04 to present:
`oil` (WTI), `gold`, `copper`, `spx` (S&P 500), `tnote` (10y yield), `usd` (DXY)

### Step 1–2: Load + Preprocess → `data/processed.parquet` (18 cols)

| Column group | Columns | Notes |
|---|---|---|
| Aligned prices | `oil, gold, copper, spx, tnote, usd` | float64 |
| Imputation flags | `*_was_imputed` | int8; 0.1–0.2% fill rate |
| Rolling z-scores | `*_zscore` | 252-day window, min_periods=30, no lookahead |

Key design decisions:
- **Union** of all date indexes (not intersection) to preserve all trading days
- Imputation flag recorded **before** `ffill()` — impossible to reconstruct after
- Right-aligned `rolling()` only; `center=True` never used

### Step 3: Regime Labels → `data/labeled.parquet` (19 cols = processed + `regime`)

20-day rolling direction of `tnote` and `spx` (sign of `pct_change(20)`):

| Regime | Label | Days | % |
|---|---|---|---|
| 1 | Inflationary Growth | 1,463 | 35.7% |
| 2 | Stagflation / Risk-off | 553 | 13.5% |
| 3 | Deflationary Growth | 1,273 | 31.1% |
| 4 | Recession / Crisis | 784 | 19.1% |
| 5 | Transitional | 25 | 0.6% |

### Step 4: Residual Targets → `data/targets.parquet` (9 cols)

For oil, gold, copper — columns per asset:
- `{asset}_fwd_ret` — 126-day (≈6 month) forward return
- `{asset}_ar_pred` — AR(5) expanding-window prediction (sklearn LinearRegression)
- `{asset}_residual` — target = fwd_ret − ar_pred

AR(5) autocorrelation removal result:

| Asset | AC(1) raw | AC(1) residual | σ reduction |
|---|---|---|---|
| Oil | 0.908 | 0.281 | −49.5% |
| Gold | 0.991 | 0.001 | −86.4% |
| Copper | 0.989 | 0.026 | −85.3% |

252-day warmup before first prediction. Last 126 rows have NaN targets
(forward return not yet resolved).

## Phase 2 — Model (PLANNED)

- Build expert models in `models/experts/` (one per regime or asset class)
- Gating network in `models/gating/` (takes z-score features + regime)
- Training loop with walk-forward validation to respect time ordering
- Evaluation in `evaluation/`
