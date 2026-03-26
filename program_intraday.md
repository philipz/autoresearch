# autoresearch — 台指期 (TX) 盤動態 5 分鐘快照實驗

This is an autonomous research experiment specifically for the **5-minute intraday prediction models** (remaining return/direction to close).

## Setup

1. **Agree on a run tag**: Propose a tag (e.g., `intra_mar25`).
2. **In-scope files**:
   - `prepare_intraday.py`: Fixed data prep for 5-min snapshots. **Do not modify.**
   - `train_intraday.py`: **The ONLY file you modify.** Contains feature engineering and LightGBM model training.
3. **Verify data**: Ensure `.cache/intraday_snapshots.csv` exists.
4. **Initialize results_intraday.tsv**: Track experiments here.

## Domain Context (Intraday)

We are predicting the **Remaining Return** and **Remaining Direction** from the current 5-minute snapshot to the day's close (13:45).

### Targets:
- `Remaining_Dir`: 1 if Close > Current_Price, else 0.
- `Remaining_Ret`: Log return from current to Close.

### Fixed Baseline (True Baseline):
- Dir AUC: ~0.5244
- Dir Acc: ~51.2%
- Pts MAE: ~64.9 pts
- Composite Score: ~0.5623

## Experimentation Loop

Run `python train_intraday.py`. 
Budget: Fixed time budget (scikit-learn/LightGBM are fast).

### What you CAN do in `train_intraday.py`:
- **Dynamic Feature Engineering**: Create complex interactions between `VWAP_Dist`, `Time_Progress`, `Intraday_Ret_Now`, and `Vol_Ratio`.
- **Non-linear Transforms**: Sqrt/Log of volume, squared gaps, etc.
- **Model Tuning**: LightGBM hyperparameters (num_leaves, learning_rate, feature_fraction, etc.).
- **Cross-validation**: Adjust `N_SPLITS` (TimeSeriesSplit).

### What you CANNOT do:
- Modify `prepare_intraday.py`.
- Use same-day `Remaining_Ret` (target) as a feature.

## Available Features (Snapshot):
| Feature | Type | Description |
|---|---|---|
| `Intraday_Ret_Now` | Dynamic | Current price vs Open |
| `VWAP_Dist` | Dynamic | Current price vs VWAP |
| `Vol_Ratio` | Dynamic | Current volume vs prior avg volume |
| `Time_Progress` | Dynamic | 0.0 (Open) to 1.0 (Close) |
| `Prev_Intraday_Ret` | Static | Yesterday's total intraday return |
| `NetOI_Diff` | Static | Yesterday's Net OI change (Institutional) |
| `TSM_Ret` / `SOX_Ret` | Static | Yesterday's US market returns |

## Logging
Log to `results_intraday.tsv`.
Columns: `commit`, `composite_score`, `val_dir_auc`, `status`, `description`.

---
**GOAL**: Push the real Dir AUC above **0.55** without any look-ahead bias!
