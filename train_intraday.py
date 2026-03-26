"""
Autoresearch Sandbox — Intraday Model Training Script (AI MODIFIABLE)

This is the ONLY file the AI agent should modify for intraday experiments.
It contains the full training pipeline:
  A. Feature Engineering
  B. Model Definition & Hyperparameters
  C. Training & Cross-Validation
  D. Composite Metric Calculation

Usage:
    python train_intraday.py

The script prints a final summary with the composite metric (lower is better).
"""

import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
)

from prepare_intraday import get_intraday_data, TIME_BUDGET, RANDOM_SEED

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)

t_start = time.time()

# ---------------------------------------------------------------------------
# A. Feature Engineering (AI: feel free to modify)
# ---------------------------------------------------------------------------
train_df, val_df, _ = get_intraday_data()

def engineer_features(df):
    """
    Transform raw snapshot features into model-ready features.
    AI agent: modify this function to try different feature combinations.

    Available dynamic features (change every 5 min):
        Intraday_Ret_Now    - current return since open
        VWAP_Dist           - distance from VWAP (positive = above)
        Vol_Ratio           - current bar volume / average bar volume
        Day_Range           - running high-low range / open
        Range_Position      - position in today's range (0=low, 1=high)
        Time_Progress       - progress through the day (0=open, 1=close)
        Bar_Body            - current bar body / open
        Bar_Range           - current bar range / open
        Mom3                - 3-bar momentum (15 min)
        Mom6                - 6-bar momentum (30 min)
        Cum_Volume          - cumulative volume so far

    Available static features (constant within a day):
        TSM_Ret, SOX_Ret, NetOI_Diff, Open_Gap,
        Intraday_Ret (previous day), TX_Ret, MA5, MA20, RSI,
        Sentiment_Score, Sentiment_Conf

    Returns:
        dict with keys:
            'X': feature matrix
            'y_dir': target for remaining direction (binary)
            'y_ret': target for remaining return (regression)
            'y_pts': target for remaining points (regression)
    """
    # --- Dynamic features ---
    X = pd.DataFrame(index=df.index)

    # Core dynamic signals
    X['Intraday_Ret_Now'] = df['Intraday_Ret_Now']
    X['VWAP_Dist'] = df['VWAP_Dist']
    X['Vol_Ratio'] = df['Vol_Ratio']
    X['Day_Range'] = df['Day_Range']
    X['Range_Position'] = df['Range_Position']
    X['Time_Progress'] = df['Time_Progress']
    X['Bar_Body'] = df['Bar_Body']
    X['Bar_Range'] = df['Bar_Range']
    X['Mom3'] = df['Mom3']
    X['Mom6'] = df['Mom6']

    # Volume feature (log-scaled)
    X['Log_Cum_Volume'] = np.log1p(df['Cum_Volume'])

    # Interaction features
    X['Ret_x_Time'] = df['Intraday_Ret_Now'] * df['Time_Progress']
    X['VWAP_x_Time'] = df['VWAP_Dist'] * df['Time_Progress']
    X['Range_x_Time'] = df['Day_Range'] * df['Time_Progress']
    X['VWAP_x_RangePos'] = df['VWAP_Dist'] * df['Range_Position']
    X['Mom3_x_Time'] = df['Mom3'] * df['Time_Progress']
    X['Mom6_x_Time'] = df['Mom6'] * df['Time_Progress']
    
    # --- New Interaction (Exp 10) ---
    X['Ret_abs_x_Time'] = np.abs(df['Intraday_Ret_Now']) * df['Time_Progress']
    X['Vol_x_Time'] = df['Vol_Ratio'] * df['Time_Progress']
    
    # Removed Exp 8: RSI_x_Time, Range_x_Vol (Worse AUC)
    
    # Non-linear transforms
    X['Ret_abs'] = np.abs(df['Intraday_Ret_Now'])
    X['Open_Gap_sq'] = df['Open_Gap'] ** 2
    X['Open_Gap_abs'] = np.abs(df['Open_Gap'])
    X['Log_Vol_Ratio'] = np.log1p(df['Vol_Ratio'])
    X['Sqrt_Vol_Ratio'] = np.sqrt(df['Vol_Ratio'])

    # --- Static features (same value for entire day) ---
    X['TSM_Ret'] = df['TSM_Ret']
    X['SOX_Ret'] = df['SOX_Ret']
    X['NetOI_Diff'] = df['NetOI_Diff']
    X['Open_Gap'] = df['Open_Gap']
    X['TX_Ret'] = df['TX_Ret']  # previous day's total return
    X['RSI'] = df['RSI']

    # MA spread
    if 'MA5' in df.columns and 'MA20' in df.columns:
        X['MA5_MA20_Spread'] = df['MA5'] - df['MA20']

    # Sentiment
    if 'Sentiment_Score' in df.columns:
        X['Sentiment_Score'] = df['Sentiment_Score']
    if 'Sentiment_Conf' in df.columns:
        X['Sentiment_Conf'] = df['Sentiment_Conf']

    # Previous day intraday return
    if 'Intraday_Ret' in df.columns:
        X['Prev_Intraday_Ret'] = df['Intraday_Ret']

    # Static feature interactions
    X['TSM_SOX_Spread'] = df['TSM_Ret'] - df['SOX_Ret']
    X['Gap_x_TSM'] = df['Open_Gap'] * df['TSM_Ret']
    
    # --- Clean Core Set (Exp 26: The 0.60 Push) ---
    # Volatility-Adjusted Momentum: Signal divided by recent noise
    rolling_range = df['Bar_Range'].rolling(5).mean().fillna(df['Bar_Range'].iloc[0])
    X['Vol_Adj_Mom'] = df['Mom3'] / (rolling_range + 0.001)
    
    # Champion base (Exp 19)
    X['Ret_Persistence'] = df['Intraday_Ret_Now'].rolling(3).sum().fillna(0)
    X['Vol_x_Time'] = df['Vol_Ratio'] * df['Time_Progress']
    X['NetOI_x_Gap'] = df['NetOI_Diff'] * df['Open_Gap']

    leaky_cols = ['Remaining_Ret', 'Remaining_Dir', 'Remaining_Points']
    for c in leaky_cols:
        assert c not in X.columns, f"Data leakage detected: {c} in features!"

    X.fillna(0, inplace=True)

    # --- Targets ---
    y_dir = df['Remaining_Dir'].copy()
    y_ret = df['Remaining_Ret'].copy()
    y_pts = df['Remaining_Points'].copy()

    return {
        'X': X,
        'y_dir': y_dir,
        'y_ret': y_ret,
        'y_pts': y_pts,
    }


# ---------------------------------------------------------------------------
# B. Model Definition & Hyperparameters (AI: feel free to modify)
# ---------------------------------------------------------------------------

# Direction Classifier (LightGBM)
DIR_CLF_PARAMS = dict(
    n_estimators=3000,
    max_depth=6,
    num_leaves=63,
    learning_rate=0.003,
    subsample=0.75,
    colsample_bytree=0.75,
    min_child_samples=30,
    reg_alpha=3.0,
    reg_lambda=15.0,
    bagging_freq=5,
    bagging_fraction=0.7,
    random_state=RANDOM_SEED,
    verbose=-1,
)

# Remaining Points Regressor (LightGBM)
PTS_REG_PARAMS = dict(
    n_estimators=3000,
    max_depth=6,
    num_leaves=63,
    learning_rate=0.003,
    subsample=0.75,
    colsample_bytree=0.75,
    min_child_samples=30,
    reg_alpha=3.0,
    reg_lambda=15.0,
    bagging_freq=5,
    bagging_fraction=0.7,
    random_state=RANDOM_SEED,
    verbose=-1,
)

# Cross-validation
N_SPLITS = 5  # TimeSeriesSplit folds


# ---------------------------------------------------------------------------
# C. Training & Evaluation (AI: feel free to modify)
# ---------------------------------------------------------------------------

def train_and_evaluate_cv(train_df, val_df):
    """
    分時段訓練：Morning / Midday / Afternoon 三個獨立 LightGBM 模型。
    各時段分別做 TimeSeriesSplit CV，最後在 val_df 合併評估。
    """
    SEGMENTS = {
        'morning':   lambda df: df['Time_Progress'] < 0.30,
        'midday':    lambda df: (df['Time_Progress'] >= 0.30) & (df['Time_Progress'] < 0.70),
        'afternoon': lambda df: df['Time_Progress'] >= 0.70,
    }

    trained_clfs = {}
    trained_regs = {}
    cv_dir_auc_all = []
    cv_pts_mse_all = []

    # ── 各時段訓練 ──
    for seg_name, seg_mask_fn in SEGMENTS.items():
        seg_train = train_df[seg_mask_fn(train_df)].copy()
        if len(seg_train) == 0:
            print(f"  [{seg_name}] 無訓練資料，跳過")
            continue

        seg_data = engineer_features(seg_train)
        X_seg = seg_data['X']
        y_dir_seg = seg_data['y_dir']
        y_pts_seg = seg_data['y_pts']

        print(f"\n[{seg_name.upper()}] {len(X_seg)} snapshots")
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        seg_auc = []
        seg_mse = []

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_seg)):
            clf = lgb.LGBMClassifier(**DIR_CLF_PARAMS)
            clf.fit(X_seg.iloc[tr_idx], y_dir_seg.iloc[tr_idx])
            probs = clf.predict_proba(X_seg.iloc[te_idx])
            try:
                auc = roc_auc_score(y_dir_seg.iloc[te_idx], probs[:, 1]) if probs.shape[1] == 2 else 0.5
            except ValueError:
                auc = 0.5
            seg_auc.append(auc)

            reg = lgb.LGBMRegressor(**PTS_REG_PARAMS)
            reg.fit(X_seg.iloc[tr_idx], y_pts_seg.iloc[tr_idx])
            preds = reg.predict(X_seg.iloc[te_idx])
            seg_mse.append(mean_squared_error(y_pts_seg.iloc[te_idx], preds))

        print(f"  CV AUC: {np.mean(seg_auc):.4f} ± {np.std(seg_auc):.4f}")
        cv_dir_auc_all.extend(seg_auc)
        cv_pts_mse_all.extend(seg_mse)

        # 全段訓練資料 final model
        clf_final = lgb.LGBMClassifier(**DIR_CLF_PARAMS)
        clf_final.fit(X_seg, y_dir_seg)
        trained_clfs[seg_name] = clf_final

        reg_final = lgb.LGBMRegressor(**PTS_REG_PARAMS)
        reg_final.fit(X_seg, y_pts_seg)
        trained_regs[seg_name] = reg_final

    print(f"\nCV Averages (all segments):")
    print(f"  Dir AUC: {np.mean(cv_dir_auc_all):.4f} ± {np.std(cv_dir_auc_all):.4f}")
    print(f"  Pts MSE: {np.mean(cv_pts_mse_all):.2f}")

    # ── Validation 評估（合併各時段預測）──
    print("\nValidation set evaluation:")
    val_dir_probs_all = []
    val_dir_true_all  = []
    val_pts_preds_all = []
    val_pts_true_all  = []

    for seg_name, seg_mask_fn in SEGMENTS.items():
        if seg_name not in trained_clfs:
            continue
        seg_val = val_df[seg_mask_fn(val_df)].copy()
        if len(seg_val) == 0:
            continue

        seg_data = engineer_features(seg_val)
        X_val_seg = seg_data['X']
        y_dir_val = seg_data['y_dir']
        y_pts_val = seg_data['y_pts']

        dir_probs = trained_clfs[seg_name].predict_proba(X_val_seg)
        pts_preds = trained_regs[seg_name].predict(X_val_seg)

        val_dir_probs_all.append(dir_probs[:, 1])
        val_dir_true_all.append(y_dir_val.values)
        val_pts_preds_all.append(pts_preds)
        val_pts_true_all.append(y_pts_val.values)

        # 每時段獨立 AUC
        try:
            seg_auc = roc_auc_score(y_dir_val, dir_probs[:, 1])
        except ValueError:
            seg_auc = 0.5
        seg_acc = accuracy_score(y_dir_val, trained_clfs[seg_name].predict(X_val_seg))
        seg_mae = mean_absolute_error(y_pts_val, pts_preds)
        print(f"  [{seg_name:9s}] AUC={seg_auc:.4f}  Acc={seg_acc:.4f}  MAE={seg_mae:.1f}pts")

    # 合併全時段
    y_dir_all   = np.concatenate(val_dir_true_all)
    probs_all   = np.concatenate(val_dir_probs_all)
    y_pts_all   = np.concatenate(val_pts_true_all)
    pts_pred_all = np.concatenate(val_pts_preds_all)

    try:
        val_dir_auc = roc_auc_score(y_dir_all, probs_all)
    except ValueError:
        val_dir_auc = 0.5
    val_dir_acc  = accuracy_score(y_dir_all, (probs_all >= 0.5).astype(int))
    val_pts_mse  = mean_squared_error(y_pts_all, pts_pred_all)
    val_pts_mae  = mean_absolute_error(y_pts_all, pts_pred_all)

    print(f"\n  [overall   ] AUC={val_dir_auc:.4f}  Acc={val_dir_acc:.4f}  MAE={val_pts_mae:.1f}pts  MSE={val_pts_mse:.2f}")

    # Feature importance（用 afternoon 模型，收盤前信號最重要）
    print("\n[Feature Importance - Afternoon Classifier]")
    if 'afternoon' in trained_clfs:
        aft_data = engineer_features(train_df[SEGMENTS['afternoon'](train_df)])
        feat_names = aft_data['X'].columns.tolist()
        for feat, imp in sorted(
            zip(feat_names, trained_clfs['afternoon'].feature_importances_),
            key=lambda x: -x[1]
        )[:10]:
            print(f"  {feat}: {imp:.4f}")

    return {
        'val_dir_auc': val_dir_auc,
        'val_dir_acc': val_dir_acc,
        'val_pts_mse': val_pts_mse,
        'val_pts_mae': val_pts_mae,
        'cv_dir_auc_mean': np.mean(cv_dir_auc_all),
        'cv_pts_mse_mean': np.mean(cv_pts_mse_all),
    }


# ---------------------------------------------------------------------------
# D. Composite Metric (AI: feel free to modify weights/formula)
# ---------------------------------------------------------------------------

def composite_metric(metrics):
    """
    Compute a single "north star" metric. LOWER IS BETTER.

    - Direction loss:    (1 - AUC)     → 0 is perfect
    - Points loss:       MAE / 100     → normalized by typical move (~50-100 pts)

    Weights:
    - 50% direction (most important for entry signal)
    - 50% magnitude (for position sizing)
    """
    dir_loss = 1.0 - metrics['val_dir_auc']
    pts_loss = metrics['val_pts_mae'] / 100.0

    composite = 0.5 * dir_loss + 0.5 * pts_loss
    return composite


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Autoresearch Sandbox — Intraday Model Training")
    print("=" * 60)

    t_train_start = time.time()

    metrics = train_and_evaluate_cv(train_df, val_df)
    score = composite_metric(metrics)

    t_end = time.time()
    training_time = t_end - t_train_start
    total_time = t_end - t_start

    # --- Final summary (parsed by autoresearch loop) ---
    print()
    print("---")
    print(f"composite_score:  {score:.6f}")
    print(f"val_dir_auc:      {metrics['val_dir_auc']:.6f}")
    print(f"val_dir_acc:      {metrics['val_dir_acc']:.6f}")
    print(f"val_pts_mse:      {metrics['val_pts_mse']:.2f}")
    print(f"val_pts_mae:      {metrics['val_pts_mae']:.2f}")
    print(f"cv_dir_auc_mean:  {metrics['cv_dir_auc_mean']:.6f}")
    print(f"cv_pts_mse_mean:  {metrics['cv_pts_mse_mean']:.2f}")
    print(f"training_seconds: {training_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"n_splits:         {N_SPLITS}")
    print(f"train_rows:       {len(train_df)}")
    print(f"val_rows:         {len(val_df)}")
