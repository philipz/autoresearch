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
    Train models with TimeSeriesSplit cross-validation on train_df,
    then evaluate on val_df for the final metrics.
    """
    train_data = engineer_features(train_df)
    val_data = engineer_features(val_df)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    cv_dir_auc = []
    cv_pts_mse = []

    X_train = train_data['X']
    y_dir_train = train_data['y_dir']
    y_pts_train = train_data['y_pts']

    print(f"\nRunning {N_SPLITS}-fold TimeSeriesSplit CV on {len(X_train)} snapshots...")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
        # Direction Classifier
        clf = lgb.LGBMClassifier(**DIR_CLF_PARAMS)
        clf.fit(X_train.iloc[tr_idx], y_dir_train.iloc[tr_idx])
        probs = clf.predict_proba(X_train.iloc[te_idx])
        if probs.shape[1] == 2:
            try:
                auc = roc_auc_score(y_dir_train.iloc[te_idx], probs[:, 1])
            except ValueError:
                auc = 0.5
        else:
            auc = 0.5
        cv_dir_auc.append(auc)

        # Points Regressor
        reg = lgb.LGBMRegressor(**PTS_REG_PARAMS)
        reg.fit(X_train.iloc[tr_idx], y_pts_train.iloc[tr_idx])
        preds = reg.predict(X_train.iloc[te_idx])
        mse = mean_squared_error(y_pts_train.iloc[te_idx], preds)
        cv_pts_mse.append(mse)

        print(f"  Fold {fold+1}: AUC={auc:.4f}, MSE={mse:.2f}")

    print(f"\nCV Averages:")
    print(f"  Dir AUC:     {np.mean(cv_dir_auc):.4f} ± {np.std(cv_dir_auc):.4f}")
    print(f"  Pts MSE:     {np.mean(cv_pts_mse):.2f} ± {np.std(cv_pts_mse):.2f}")

    # --- Final models trained on full training set ---
    print("\nTraining final models on full training set...")

    dir_clf = lgb.LGBMClassifier(**DIR_CLF_PARAMS)
    dir_clf.fit(X_train, y_dir_train)

    pts_reg = lgb.LGBMRegressor(**PTS_REG_PARAMS)
    pts_reg.fit(X_train, y_pts_train)

    # --- Validation set evaluation ---
    print("\nValidation set evaluation:")

    X_val = val_data['X']
    y_dir_val = val_data['y_dir']
    y_pts_val = val_data['y_pts']

    # Direction
    dir_probs = dir_clf.predict_proba(X_val)
    dir_preds = dir_clf.predict(X_val)
    try:
        val_dir_auc = roc_auc_score(y_dir_val, dir_probs[:, 1])
    except ValueError:
        val_dir_auc = 0.5
    val_dir_acc = accuracy_score(y_dir_val, dir_preds)
    print(f"  Dir Accuracy: {val_dir_acc:.4f}")
    print(f"  Dir AUC:      {val_dir_auc:.4f}")

    # Points
    pts_preds = pts_reg.predict(X_val)
    val_pts_mse = mean_squared_error(y_pts_val, pts_preds)
    val_pts_mae = mean_absolute_error(y_pts_val, pts_preds)
    print(f"  Pts MSE:      {val_pts_mse:.2f}")
    print(f"  Pts MAE:      {val_pts_mae:.2f} points")

    # Feature importance
    print("\n[Feature Importance - Direction Classifier]")
    feat_names = X_train.columns.tolist()
    for feat, imp in sorted(zip(feat_names, dir_clf.feature_importances_), key=lambda x: -x[1])[:10]:
        print(f"  {feat}: {imp:.4f}")

    return {
        'val_dir_auc': val_dir_auc,
        'val_dir_acc': val_dir_acc,
        'val_pts_mse': val_pts_mse,
        'val_pts_mae': val_pts_mae,
        'cv_dir_auc_mean': np.mean(cv_dir_auc),
        'cv_pts_mse_mean': np.mean(cv_pts_mse),
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
