# autoresearch — 台指期 (TX) 自動化實驗

This is an autonomous research experiment for Taiwan Futures (TX) prediction models using AI agents.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The sandbox is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, `get_data()` function. Do not modify.
   - `train.py` — the file you modify. Feature engineering, model selection, hyperparameters, training loop.
4. **Verify data exists**: Check that `autoresearch_sandbox/.cache/` contains `prepared_data.csv`. If not, tell the human to run `python prepare.py` to fetch and cache the data.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Domain Context

This project predicts **Taiwan Futures (TAIEX Futures, TX)** trading signals:

- **Gap Direction**: Will tomorrow's market gap up or down? (Binary classification)
  - Features: TSM ADR return, SOX index return (both T-1 shifted)
- **Gap Magnitude**: How many points will the gap be? (Regression)
  - Same features as Gap Direction
- **Intraday Trend**: What's the intraday return (Open-to-Close)? (Regression)
  - Features: Gap size, FINI (foreign institutional) net OI change

All models currently use **scikit-learn GradientBoosting** (Classifier/Regressor).

## Experimentation

Each experiment runs on CPU (macOS, no GPU). The training script has a **fixed time budget of 5 minutes** (wall clock), though typical runs complete in seconds since scikit-learn is fast on small datasets. You launch it simply as: `python train.py`

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - **Feature engineering**: create new features from existing columns (lags, rolling stats, interactions, polynomial features, volatility indicators, etc.)
  - **Model selection**: swap GradientBoosting for RandomForest, XGBoost, LightGBM, or any model available in scikit-learn
  - **Hyperparameters**: n_estimators, learning_rate, max_depth, subsample, min_samples, etc.
  - **Cross-validation**: change N_SPLITS, use different CV strategies
  - **Composite metric formula**: adjust weights, normalization, or the formula itself
  - **Feature selection**: try different subsets of available features

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed data pipeline, data splits, and constants.
- Install new packages or add dependencies beyond what's in `requirements.txt`.
- Access the test set during training. The test set is reserved for final human evaluation.

**The goal is simple: get the lowest `composite_score`.** Since the data is small, training is fast. Use the extra time budget for thorough cross-validation and feature exploration.

**Available raw features** (from `prepare.py`):
| Feature | Description | Source |
|---|---|---|
| `TSM_Ret` | TSMC ADR log return, T-1 shifted | Yahoo Finance |
| `SOX_Ret` | Philadelphia Semiconductor Index log return, T-1 | Yahoo Finance |
| `NetOI_Diff` | FINI Net OI daily change, T-1 shifted | TAIFEX |
| `Open_Gap` | Open - Previous Close (points) | TAIFEX |
| `Intraday_Ret` | log(Close/Open) — intraday log return | TAIFEX |
| `Intraday_Point` | Close - Open (points) | TAIFEX |
| `TX_Ret` | Daily close-to-close log return | TAIFEX |
| `MA5` | 5-day simple moving average | Computed |
| `MA20` | 20-day simple moving average | Computed |
| `RSI` | 14-day RSI | Computed |

> **CAUTION**: When engineering features for prediction models, be careful about **data leakage**. Do NOT use same-day `Intraday_Ret`, `Intraday_Point`, or `TX_Ret` as features for gap prediction (they are targets or future information). Only use **T-1 or earlier** data as predictive features.

## Output format

Once the script finishes it prints a summary like this:

```
---
composite_score:  0.345678
val_gap_auc:      0.567890
val_gap_acc:      0.560000
val_gap_mae:      78.50
val_intra_mse:    0.00012345
val_intra_mae:    0.008765
cv_gap_auc_mean:  0.545678
cv_gap_mae_mean:  82.30
cv_intra_mse_mean:0.00015678
training_seconds: 2.5
total_seconds:    3.1
n_splits:         5
train_rows:       180
val_rows:         39
```

You can extract the key metric from the log file:

```
grep "^composite_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	composite_score	val_gap_auc	status	description
```

1. git commit hash (short, 7 chars)
2. composite_score achieved (e.g. 0.345678) — use 0.000000 for crashes
3. val_gap_auc (e.g. 0.567) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^composite_score:\|^val_gap_auc:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If composite_score improved (lower), you "advance" the branch, keeping the git commit
9. If composite_score is equal or worse, you git reset back to where you started

## Research directions to explore

Here are promising avenues, roughly ordered by expected impact:

### High Priority
- **Feature interactions**: Try `TSM_Ret * SOX_Ret`, `TSM_Ret / SOX_Ret`, quadratic terms
- **Lagged features**: Add `TSM_Ret_lag2`, `SOX_Ret_lag2` (2-day lookback)
- **Rolling statistics**: 5-day, 10-day rolling mean/std of returns
- **Cross-model features**: Use gap prediction as input feature for intraday model

### Medium Priority
- **Model ensembles**: Combine GradientBoosting + RandomForest predictions
- **Hyperparameter tuning**: Grid/random search over key parameters
- **Alternative models**: Try RandomForest, ExtraTrees, or AdaBoost as drop-in replacements
- **Feature selection**: Use feature importance to prune weak features

### Lower Priority (more speculative)
- **Volatility features**: ATR-like measures, Bollinger Band width
- **Regime detection**: Cluster market states and use as categorical feature
- **Non-linear transforms**: Log, sqrt, or polynomial transforms of features
- **Composite metric redesign**: Different weighting schemes

**Timeout**: Each experiment should take a few seconds. If a run exceeds 2 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, fix simple bugs (typos, missing imports) and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical feature engineering, try different model families. The loop runs until the human interrupts you.
