"""
Autoresearch Sandbox — Training Script (AI MODIFIABLE)

This is the ONLY file the AI agent should modify.
It contains the full training pipeline:
  A. Feature Engineering
  B. Model Definition & Hyperparameters
  C. Training & Cross-Validation
  D. Composite Metric Calculation

Usage:
    python train.py

The script prints a final summary with the composite metric (lower is better).
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesRegressor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
)

from prepare import get_data, TIME_BUDGET, RANDOM_SEED

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)

t_start = time.time()

# ---------------------------------------------------------------------------
# A. Feature Engineering (AI: feel free to modify)
# ---------------------------------------------------------------------------
# Load data from prepare.py (immutable splits)
train_df, val_df, test_df = get_data()

def engineer_features(df):
    """
    Transform raw features into model-ready features.
    AI agent: modify this function to try different feature combinations,
    transformations, lags, technical indicators, etc.

    Available raw columns from prepare.py:
        TSM_Ret, SOX_Ret, NetOI_Diff, Open_Gap, Intraday_Ret,
        Intraday_Point, TX_Ret, MA5, MA20, RSI

    Returns:
        dict with keys:
            'X_gap': features for gap prediction
            'X_intra': features for intraday prediction
            'y_gap_dir': target for gap direction (binary)
            'y_gap_val': target for gap value (regression)
            'y_intra': target for intraday return (regression)
    """
    # --- Gap model features ---
    # US market signals from previous day predict next-day gap
    X_gap = df[['TSM_Ret', 'SOX_Ret']].copy()
    X_gap['TSM_Ret_lag2'] = X_gap['TSM_Ret'].shift(1)
    X_gap['SOX_Ret_lag2'] = X_gap['SOX_Ret'].shift(1)
    X_gap['TX_Ret_lag1'] = df['TX_Ret'].shift(1)
    X_gap['TSM_SOX_Spread'] = X_gap['TSM_Ret'] - X_gap['SOX_Ret']
    X_gap['TX_Vol5'] = df['TX_Ret'].rolling(5).std()
    X_gap['TX_Mom5'] = df['TX_Ret'].rolling(5).sum()
    X_gap['TX_Vol10'] = df['TX_Ret'].rolling(10).std()
    X_gap['TX_Mom10'] = df['TX_Ret'].rolling(10).sum()
    # 20-day rolling stats
    X_gap['TX_Vol20'] = df['TX_Ret'].rolling(20).std()
    X_gap['TX_Mom20'] = df['TX_Ret'].rolling(20).sum()
    # Institutional flow
    X_gap['NetOI_Diff'] = df['NetOI_Diff']
    # Lagged gap and intraday info
    X_gap['Open_Gap_lag1'] = df['Open_Gap'].shift(1)
    X_gap['Open_Gap_lag2'] = df['Open_Gap'].shift(2)
    X_gap['Open_Gap_5d_mean'] = df['Open_Gap'].rolling(5).mean()
    X_gap['Open_Gap_5d_std'] = df['Open_Gap'].rolling(5).std()
    X_gap['Intra_Ret_lag1'] = df['Intraday_Ret'].shift(1)
    X_gap['Intra_Ret_lag2'] = df['Intraday_Ret'].shift(2)
    X_gap['TX_Ret_cumret3'] = df['TX_Ret'].rolling(3).sum()
    # Deeper lags
    X_gap['TX_Ret_lag3'] = df['TX_Ret'].shift(3)
    X_gap['TX_Ret_lag4'] = df['TX_Ret'].shift(4)
    X_gap['TSM_Ret_lag3'] = df['TSM_Ret'].shift(2)  # actual lag3 of TSM
    # Day of week seasonality
    if hasattr(df.index, 'dayofweek'):
        X_gap['day_of_week'] = df.index.dayofweek
    else:
        X_gap['day_of_week'] = pd.to_datetime(df.index).dayofweek
    # EMA features
    X_gap['TX_EMA3'] = df['TX_Ret'].ewm(span=3).mean()
    X_gap['TX_EMA5'] = df['TX_Ret'].ewm(span=5).mean()
    X_gap['TX_EMA10'] = df['TX_Ret'].ewm(span=10).mean()
    X_gap['TX_EMA20'] = df['TX_Ret'].ewm(span=20).mean()
    X_gap['TX_EMA30'] = df['TX_Ret'].ewm(span=30).mean()
    X_gap['EMA3_EMA10_Spread'] = X_gap['TX_EMA3'] - X_gap['TX_EMA10']
    X_gap['EMA5_EMA10_Spread'] = X_gap['TX_EMA5'] - X_gap['TX_EMA10']
    X_gap['EMA5_EMA20_Spread'] = X_gap['TX_EMA5'] - X_gap['TX_EMA20']
    X_gap['EMA10_EMA30_Spread'] = X_gap['TX_EMA10'] - X_gap['TX_EMA30']
    # Intraday EMA for gap model
    X_gap['Intra_EMA5'] = df['Intraday_Ret'].ewm(span=5).mean()
    X_gap['Intra_EMA10'] = df['Intraday_Ret'].ewm(span=10).mean()
    # NetOI EMA
    X_gap['NetOI_EMA5'] = df['NetOI_Diff'].ewm(span=5).mean()
    # TSM/SOX EMA
    X_gap['TSM_EMA5'] = df['TSM_Ret'].ewm(span=5).mean()
    X_gap['SOX_EMA5'] = df['SOX_Ret'].ewm(span=5).mean()
    X_gap['TSM_EMA10'] = df['TSM_Ret'].ewm(span=10).mean()
    X_gap['SOX_EMA10'] = df['SOX_Ret'].ewm(span=10).mean()
    # Open_Gap EMA
    X_gap['Gap_EMA5'] = df['Open_Gap'].ewm(span=5).mean()
    X_gap['Gap_EMA10'] = df['Open_Gap'].ewm(span=10).mean()
    X_gap['Gap_EMA20'] = df['Open_Gap'].ewm(span=20).mean()
    X_gap['Gap_EMA5_EMA20_Spread'] = X_gap['Gap_EMA5'] - X_gap['Gap_EMA20']
    # NetOI longer EMA
    X_gap['NetOI_EMA10'] = df['NetOI_Diff'].ewm(span=10).mean()
    # SOX/TSM longer EMA
    X_gap['TSM_EMA10'] = df['TSM_Ret'].ewm(span=10).mean()
    X_gap['SOX_EMA10'] = df['SOX_Ret'].ewm(span=10).mean()
    X_gap['TSM_SOX_EMA_Spread'] = X_gap['TSM_EMA5'] - X_gap['SOX_EMA5']
    # Open_Gap EWM volatility
    X_gap['Gap_EWM_Vol5'] = df['Open_Gap'].ewm(span=5).std()
    # Technical indicators for gap model
    X_gap['RSI'] = df['RSI']
    X_gap['MA5_MA20_Spread'] = df['MA5'] - df['MA20']
    # Sentiment features (T-1, pre-shifted in prepare.py)
    if 'Sentiment_Score' in df.columns:
        X_gap['Sentiment_Score'] = df['Sentiment_Score']
        X_gap['Sentiment_Conf'] = df['Sentiment_Conf']
        # Interaction: sentiment amplified by momentum
        X_gap['Sent_Mom'] = df['Sentiment_Score'] * X_gap['TX_Mom5'].fillna(0)
    X_gap.fillna(0, inplace=True) # Fill initial lag NAs

    # --- Intraday model features ---
    # Gap size + institutional flow predict intraday movement
    X_intra = df[['Open_Gap', 'NetOI_Diff']].copy()
    X_intra['NetOI_Diff_lag2'] = X_intra['NetOI_Diff'].shift(1)
    X_intra['TX_Ret_lag1'] = df['TX_Ret'].shift(1)
    X_intra['TX_Ret_lag2'] = df['TX_Ret'].shift(2)
    X_intra['Intra_Ret_lag1'] = df['Intraday_Ret'].shift(1)
    X_intra['Intra_Ret_lag2'] = df['Intraday_Ret'].shift(2)
    X_intra['TX_Ret_cumret3'] = df['TX_Ret'].rolling(3).sum()
    X_intra['TX_Vol5'] = df['TX_Ret'].rolling(5).std()
    X_intra['TX_Mom5'] = df['TX_Ret'].rolling(5).sum()
    X_intra['TX_Vol10'] = df['TX_Ret'].rolling(10).std()
    X_intra['TX_Mom10'] = df['TX_Ret'].rolling(10).sum()
    # Technical indicators
    X_intra['RSI'] = df['RSI']
    X_intra['MA5_MA20_Spread'] = df['MA5'] - df['MA20']
    # US market signals
    X_intra['TSM_Ret'] = df['TSM_Ret']
    X_intra['SOX_Ret'] = df['SOX_Ret']
    # Sentiment features (T-1, pre-shifted in prepare.py)
    if 'Sentiment_Score' in df.columns:
        X_intra['Sentiment_Score'] = df['Sentiment_Score']
        X_intra['Sentiment_Conf'] = df['Sentiment_Conf']
    # 20-day rolling stats
    X_intra['TX_Vol20'] = df['TX_Ret'].rolling(20).std()
    X_intra['TX_Mom20'] = df['TX_Ret'].rolling(20).sum()
    # Non-linear gap effect
    X_intra['Open_Gap_sq'] = df['Open_Gap'] ** 2
    X_intra['Open_Gap_abs'] = df['Open_Gap'].abs()
    # Intraday volatility and range features
    X_intra['Intra_Vol5'] = df['Intraday_Ret'].rolling(5).std()
    X_intra['Intra_Mean5'] = df['Intraday_Ret'].rolling(5).mean()
    # Intraday point info (lag)
    X_intra['Intra_Point_lag1'] = df['Intraday_Point'].shift(1)
    X_intra['Open_Gap_lag1'] = df['Open_Gap'].shift(1)
    # EMA features for intraday
    X_intra['TX_EMA3'] = df['TX_Ret'].ewm(span=3).mean()
    X_intra['TX_EMA5'] = df['TX_Ret'].ewm(span=5).mean()
    X_intra['TX_EMA10'] = df['TX_Ret'].ewm(span=10).mean()
    X_intra['TX_EMA20'] = df['TX_Ret'].ewm(span=20).mean()
    X_intra['TX_EMA30'] = df['TX_Ret'].ewm(span=30).mean()
    X_intra['EMA3_EMA10_Spread'] = X_intra['TX_EMA3'] - X_intra['TX_EMA10']
    X_intra['EMA5_EMA10_Spread'] = X_intra['TX_EMA5'] - X_intra['TX_EMA10']
    X_intra['EMA5_EMA20_Spread'] = X_intra['TX_EMA5'] - X_intra['TX_EMA20']
    X_intra['EMA10_EMA30_Spread'] = X_intra['TX_EMA10'] - X_intra['TX_EMA30']
    X_intra['Intra_EMA3'] = df['Intraday_Ret'].ewm(span=3).mean()
    X_intra['Intra_EMA5'] = df['Intraday_Ret'].ewm(span=5).mean()
    X_intra['Intra_EMA10'] = df['Intraday_Ret'].ewm(span=10).mean()
    X_intra['Intra_EMA20'] = df['Intraday_Ret'].ewm(span=20).mean()
    X_intra['Intra_EMA3_EMA10_Spread'] = X_intra['Intra_EMA3'] - X_intra['Intra_EMA10']
    X_intra['Intra_EMA5_EMA20_Spread'] = X_intra['Intra_EMA5'] - X_intra['Intra_EMA20']
    X_intra['NetOI_EMA5'] = df['NetOI_Diff'].ewm(span=5).mean()
    # TSM/SOX EMA for intraday
    X_intra['TSM_EMA5'] = df['TSM_Ret'].ewm(span=5).mean()
    X_intra['SOX_EMA5'] = df['SOX_Ret'].ewm(span=5).mean()
    # Gap EMA for intraday
    X_intra['Gap_EMA5'] = df['Open_Gap'].ewm(span=5).mean()
    X_intra['Gap_EMA10'] = df['Open_Gap'].ewm(span=10).mean()
    # TSM/SOX longer EMA for intraday
    X_intra['TSM_EMA10'] = df['TSM_Ret'].ewm(span=10).mean()
    X_intra['SOX_EMA10'] = df['SOX_Ret'].ewm(span=10).mean()
    # Intraday Point EMA
    X_intra['IntraPoint_EMA5'] = df['Intraday_Point'].ewm(span=5).mean()
    # Deeper lags for intraday
    X_intra['TX_Ret_lag3'] = df['TX_Ret'].shift(3)
    X_intra['TX_Ret_lag4'] = df['TX_Ret'].shift(4)
    X_intra['Intra_Ret_lag3'] = df['Intraday_Ret'].shift(3)
    # Day of week seasonality
    if hasattr(df.index, 'dayofweek'):
        X_intra['day_of_week'] = df.index.dayofweek
    else:
        X_intra['day_of_week'] = pd.to_datetime(df.index).dayofweek
    X_intra.fillna(0, inplace=True)

    # --- Targets ---
    y_gap_dir = df['Gap_Direction'].copy()
    y_gap_val = df['Open_Gap'].copy()
    y_intra = df['Intraday_Ret'].copy()

    return {
        'X_gap': X_gap,
        'X_intra': X_intra,
        'y_gap_dir': y_gap_dir,
        'y_gap_val': y_gap_val,
        'y_intra': y_intra,
    }


# ---------------------------------------------------------------------------
# B. Model Definition & Hyperparameters (AI: feel free to modify)
# ---------------------------------------------------------------------------

# Gap Direction Classifier (XGBoost)
GAP_CLF_PARAMS = dict(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.015,
    subsample=0.7,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    use_label_encoder=False,
    eval_metric='logloss',
)
USE_XGB_CLF = True

# Gap Value Regressor (XGBRegressor)
GAP_REG_PARAMS = dict(
    n_estimators=3000,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
)
USE_XGB_REG = True

# Intraday Return Regressor
INTRA_REG_PARAMS = dict(
    n_estimators=700,
    max_depth=None,
    min_samples_split=5,
    random_state=RANDOM_SEED,
)

# Cross-validation
N_SPLITS = 7  # TimeSeriesSplit folds


# ---------------------------------------------------------------------------
# C. Training & Evaluation (AI: feel free to modify)
# ---------------------------------------------------------------------------

def train_and_evaluate_cv(train_df, val_df):
    """
    Train models with TimeSeriesSplit cross-validation on train_df,
    then evaluate on val_df for the final metrics.

    Returns:
        dict with metric values
    """
    # Combine train+val for CV, then final eval on val
    train_data = engineer_features(train_df)
    val_data = engineer_features(val_df)

    # --- Cross-Validation on training set ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    cv_gap_auc = []
    cv_gap_mae = []
    cv_intra_mse = []

    X_gap_train = train_data['X_gap']
    y_gap_dir_train = train_data['y_gap_dir']
    y_gap_val_train = train_data['y_gap_val']
    X_intra_train = train_data['X_intra']
    y_intra_train = train_data['y_intra']

    print(f"\nRunning {N_SPLITS}-fold TimeSeriesSplit CV...")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_gap_train)):
        # Gap Classifier
        clf = xgb.XGBClassifier(**GAP_CLF_PARAMS) if USE_XGB_CLF else GradientBoostingClassifier(**GAP_CLF_PARAMS)
        clf.fit(X_gap_train.iloc[tr_idx], y_gap_dir_train.iloc[tr_idx])
        probs = clf.predict_proba(X_gap_train.iloc[te_idx])
        if probs.shape[1] == 2:
            try:
                auc = roc_auc_score(y_gap_dir_train.iloc[te_idx], probs[:, 1])
            except ValueError:
                auc = 0.5  # degenerate case
        else:
            auc = 0.5
        cv_gap_auc.append(auc)

        # Gap Regressor
        reg = xgb.XGBRegressor(**GAP_REG_PARAMS) if USE_XGB_REG else ExtraTreesRegressor(**GAP_REG_PARAMS)
        reg.fit(X_gap_train.iloc[tr_idx], y_gap_val_train.iloc[tr_idx])
        preds = reg.predict(X_gap_train.iloc[te_idx])
        mae = mean_absolute_error(y_gap_val_train.iloc[te_idx], preds)
        cv_gap_mae.append(mae)

        # Intraday Regressor (with cross-model feature)
        cv_gap_prob_tr = clf.predict_proba(X_gap_train.iloc[tr_idx])[:, 1]
        cv_gap_prob_te = probs[:, 1] if probs.shape[1] == 2 else np.full(len(te_idx), 0.5)
        X_intra_tr_aug = X_intra_train.iloc[tr_idx].copy()
        X_intra_tr_aug['gap_up_prob'] = cv_gap_prob_tr
        X_intra_te_aug = X_intra_train.iloc[te_idx].copy()
        X_intra_te_aug['gap_up_prob'] = cv_gap_prob_te
        intra = ExtraTreesRegressor(**INTRA_REG_PARAMS)
        intra.fit(X_intra_tr_aug, y_intra_train.iloc[tr_idx])
        intra_preds = intra.predict(X_intra_te_aug)
        mse = mean_squared_error(y_intra_train.iloc[te_idx], intra_preds)
        cv_intra_mse.append(mse)

        print(f"  Fold {fold+1}: AUC={auc:.4f}, MAE={mae:.2f}, MSE={mse:.8f}")

    print(f"\nCV Averages:")
    print(f"  Gap AUC:     {np.mean(cv_gap_auc):.4f} ± {np.std(cv_gap_auc):.4f}")
    print(f"  Gap MAE:     {np.mean(cv_gap_mae):.2f} ± {np.std(cv_gap_mae):.2f}")
    print(f"  Intra MSE:   {np.mean(cv_intra_mse):.8f} ± {np.std(cv_intra_mse):.8f}")

    # --- Final models trained on full training set ---
    print("\nTraining final models on full training set...")

    gap_clf = xgb.XGBClassifier(**GAP_CLF_PARAMS) if USE_XGB_CLF else GradientBoostingClassifier(**GAP_CLF_PARAMS)
    gap_clf.fit(X_gap_train, y_gap_dir_train)

    gap_reg = xgb.XGBRegressor(**GAP_REG_PARAMS) if USE_XGB_REG else ExtraTreesRegressor(**GAP_REG_PARAMS)
    gap_reg.fit(X_gap_train, y_gap_val_train)

    # Cross-model feature: add gap_clf predicted probability to intraday features
    gap_prob_train = gap_clf.predict_proba(X_gap_train)[:, 1]
    X_intra_train_aug = X_intra_train.copy()
    X_intra_train_aug['gap_up_prob'] = gap_prob_train

    intra_reg = ExtraTreesRegressor(**INTRA_REG_PARAMS)
    intra_reg.fit(X_intra_train_aug, y_intra_train)

    # --- Validation set evaluation ---
    print("\nValidation set evaluation:")

    X_gap_val = val_data['X_gap']
    y_gap_dir_val = val_data['y_gap_dir']
    y_gap_val_val = val_data['y_gap_val']
    X_intra_val = val_data['X_intra']
    y_intra_val = val_data['y_intra']

    # Gap Classifier
    gap_probs = gap_clf.predict_proba(X_gap_val)
    gap_preds = gap_clf.predict(X_gap_val)
    try:
        val_gap_auc = roc_auc_score(y_gap_dir_val, gap_probs[:, 1])
    except ValueError:
        val_gap_auc = 0.5
    val_gap_acc = accuracy_score(y_gap_dir_val, gap_preds)
    print(f"  Gap Accuracy: {val_gap_acc:.4f}")
    print(f"  Gap AUC:      {val_gap_auc:.4f}")

    # Gap Regressor
    gap_val_preds = gap_reg.predict(X_gap_val)
    val_gap_mae = mean_absolute_error(y_gap_val_val, gap_val_preds)
    print(f"  Gap MAE:      {val_gap_mae:.2f} points")

    # Intraday Regressor (with cross-model feature)
    X_intra_val_aug = X_intra_val.copy()
    X_intra_val_aug['gap_up_prob'] = gap_probs[:, 1]
    intra_val_preds = intra_reg.predict(X_intra_val_aug)
    val_intra_mse = mean_squared_error(y_intra_val, intra_val_preds)
    val_intra_mae = mean_absolute_error(y_intra_val, intra_val_preds)
    print(f"  Intra MSE:    {val_intra_mse:.8f}")
    print(f"  Intra MAE:    {val_intra_mae:.6f}")

    return {
        'val_gap_auc': val_gap_auc,
        'val_gap_acc': val_gap_acc,
        'val_gap_mae': val_gap_mae,
        'val_intra_mse': val_intra_mse,
        'val_intra_mae': val_intra_mae,
        'cv_gap_auc_mean': np.mean(cv_gap_auc),
        'cv_gap_mae_mean': np.mean(cv_gap_mae),
        'cv_intra_mse_mean': np.mean(cv_intra_mse),
    }


# ---------------------------------------------------------------------------
# D. Composite Metric (AI: feel free to modify weights/formula)
# ---------------------------------------------------------------------------

def composite_metric(metrics):
    """
    Compute a single "north star" metric from individual model metrics.
    LOWER IS BETTER.

    Current formula: weighted sum of normalized losses.
      - Gap classification loss: (1 - AUC)       range [0, 0.5]
      - Gap regression loss:     MAE / 100        normalized by typical gap magnitude
      - Intraday MSE:            MSE * 10000      scaled for readability

    Weights reflect trading strategy priorities:
      - 30% gap direction (affects position direction)
      - 30% gap magnitude (affects position size)
      - 40% intraday trend (affects intraday management)
    """
    gap_clf_loss = 1.0 - metrics['val_gap_auc']
    gap_reg_loss = metrics['val_gap_mae'] / 100.0
    intra_loss = metrics['val_intra_mse'] * 10000.0

    w_clf = 0.3
    w_gap = 0.3
    w_intra = 0.4

    composite = w_clf * gap_clf_loss + w_gap * gap_reg_loss + w_intra * intra_loss
    return composite


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Autoresearch Sandbox — Training")
    print("=" * 60)

    t_train_start = time.time()

    # Run training and evaluation
    metrics = train_and_evaluate_cv(train_df, val_df)

    # Compute composite metric
    score = composite_metric(metrics)

    t_end = time.time()
    training_time = t_end - t_train_start
    total_time = t_end - t_start

    # --- Final summary (parsed by autoresearch loop) ---
    print()
    print("---")
    print(f"composite_score:  {score:.6f}")
    print(f"val_gap_auc:      {metrics['val_gap_auc']:.6f}")
    print(f"val_gap_acc:      {metrics['val_gap_acc']:.6f}")
    print(f"val_gap_mae:      {metrics['val_gap_mae']:.2f}")
    print(f"val_intra_mse:    {metrics['val_intra_mse']:.8f}")
    print(f"val_intra_mae:    {metrics['val_intra_mae']:.6f}")
    print(f"cv_gap_auc_mean:  {metrics['cv_gap_auc_mean']:.6f}")
    print(f"cv_gap_mae_mean:  {metrics['cv_gap_mae_mean']:.2f}")
    print(f"cv_intra_mse_mean:{metrics['cv_intra_mse_mean']:.8f}")
    print(f"training_seconds: {training_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"n_splits:         {N_SPLITS}")
    print(f"train_rows:       {len(train_df)}")
    print(f"val_rows:         {len(val_df)}")
