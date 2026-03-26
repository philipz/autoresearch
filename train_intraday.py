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

import joblib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prepare_intraday import get_intraday_data, TIME_BUDGET, RANDOM_SEED
from src.config import Config

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
    Full feature set with OFI.

    Returns:
        dict with keys:
            'X': feature matrix
            'y_dir': target for remaining direction (binary)
            'y_ret': target for remaining return (regression)
            'y_pts': target for remaining points (regression)
    """
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
    X['Log_Cum_Volume'] = np.log1p(df['Cum_Volume'])

    # Interaction features
    X['Ret_x_Time'] = df['Intraday_Ret_Now'] * df['Time_Progress']
    X['VWAP_x_Time'] = df['VWAP_Dist'] * df['Time_Progress']
    X['Range_x_Time'] = df['Day_Range'] * df['Time_Progress']
    X['VWAP_x_RangePos'] = df['VWAP_Dist'] * df['Range_Position']
    X['Mom3_x_Time'] = df['Mom3'] * df['Time_Progress']
    X['Mom6_x_Time'] = df['Mom6'] * df['Time_Progress']
    X['Ret_abs_x_Time'] = np.abs(df['Intraday_Ret_Now']) * df['Time_Progress']
    X['Vol_x_Time'] = df['Vol_Ratio'] * df['Time_Progress']

    # Non-linear transforms
    X['Ret_abs'] = np.abs(df['Intraday_Ret_Now'])
    X['Open_Gap_sq'] = df['Open_Gap'] ** 2
    X['Open_Gap_abs'] = np.abs(df['Open_Gap'])
    X['Log_Vol_Ratio'] = np.log1p(df['Vol_Ratio'])
    X['Sqrt_Vol_Ratio'] = np.sqrt(df['Vol_Ratio'])

    # Static features
    X['TSM_Ret'] = df['TSM_Ret']
    X['SOX_Ret'] = df['SOX_Ret']
    X['NetOI_Diff'] = df['NetOI_Diff']
    X['Open_Gap'] = df['Open_Gap']
    X['TX_Ret'] = df['TX_Ret']
    X['RSI'] = df['RSI']

    if 'MA5' in df.columns and 'MA20' in df.columns:
        X['MA5_MA20_Spread'] = df['MA5'] - df['MA20']
    if 'Sentiment_Score' in df.columns:
        X['Sentiment_Score'] = df['Sentiment_Score']
    if 'Sentiment_Conf' in df.columns:
        X['Sentiment_Conf'] = df['Sentiment_Conf']
    if 'Intraday_Ret' in df.columns:
        X['Prev_Intraday_Ret'] = df['Intraday_Ret']

    X['TSM_SOX_Spread'] = df['TSM_Ret'] - df['SOX_Ret']
    X['Gap_x_TSM'] = df['Open_Gap'] * df['TSM_Ret']

    # Vol-Adjusted Momentum (Exp 26)
    rolling_range = df['Bar_Range'].rolling(5).mean().fillna(df['Bar_Range'].iloc[0])
    X['Vol_Adj_Mom'] = df['Mom3'] / (rolling_range + 0.001)

    # Champion features (Exp 19)
    X['Ret_Persistence'] = df['Intraday_Ret_Now'].rolling(3).sum().fillna(0)
    X['NetOI_x_Gap'] = df['NetOI_Diff'] * df['Open_Gap']

    # --- OFI 逐筆流量特徵 ---
    if 'OFI' in df.columns:
        X['OFI'] = df['OFI']
        if 'OFI_SMA3' in df.columns:
            X['OFI_SMA3'] = df['OFI_SMA3']
        if 'Cum_OFI' in df.columns:
            X['Cum_OFI_neg'] = -df['Cum_OFI']
            X['OFI_x_Time'] = df['OFI'] * df['Time_Progress']
        if 'Large_Trade_Ratio' in df.columns:
            X['Large_Trade_Ratio'] = df['Large_Trade_Ratio']
        X['NetOI_x_OFI'] = df['NetOI_Diff'] * df['OFI']

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

# Direction Classifier (LightGBM) — Morning+Midday ~12k 筆
DIR_CLF_PARAMS = dict(
    n_estimators=1500,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.005,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=1.0,
    reg_lambda=10.0,
    bagging_freq=5,
    bagging_fraction=0.8,
    random_state=RANDOM_SEED,
    verbose=-1,
)

# Remaining Points Regressor (LightGBM) — Morning+Midday ~12k 筆
PTS_REG_PARAMS = dict(
    n_estimators=1500,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.005,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=1.0,
    reg_lambda=10.0,
    bagging_freq=5,
    bagging_fraction=0.8,
    random_state=RANDOM_SEED,
    verbose=-1,
)

# Cross-validation
N_SPLITS = 5  # TimeSeriesSplit folds

# Time-segment boundaries for confidence analysis
MORNING_CUTOFF = 0.30    # 08:45 ~ 10:15 (開盤後 90 分鐘)
AFTERNOON_CUTOFF = 0.70  # 10:15 ~ 12:15 (midday 結束)


# ---------------------------------------------------------------------------
# C. Training & Evaluation (AI: feel free to modify)
# ---------------------------------------------------------------------------

def train_and_evaluate_cv(train_df, val_df):
    """
    全天訓練模型（方案B）：使用全天資料訓練，
    輸出分時段 AUC 信心曲線供應用層決策。

    部署建議：Time_Progress >= 0.70（下午12:15後）信心度接近隨機，
    應用層可選擇忽略該時段預測。
    """
    # 全天訓練（使用所有資料）
    train_active = train_df.copy()
    print(f"Training on full day: {len(train_active)} rows")

    train_data = engineer_features(train_active)
    val_data   = engineer_features(val_df)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_dir_auc = []
    cv_pts_mse = []

    X_train     = train_data['X']
    y_dir_train = train_data['y_dir']
    y_pts_train = train_data['y_pts']

    print(f"\nRunning {N_SPLITS}-fold TimeSeriesSplit CV on {len(X_train)} snapshots...")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
        clf = lgb.LGBMClassifier(**DIR_CLF_PARAMS)
        clf.fit(X_train.iloc[tr_idx], y_dir_train.iloc[tr_idx])
        probs = clf.predict_proba(X_train.iloc[te_idx])
        try:
            auc = roc_auc_score(y_dir_train.iloc[te_idx], probs[:, 1]) if probs.shape[1] == 2 else 0.5
        except ValueError:
            auc = 0.5
        cv_dir_auc.append(auc)

        reg = lgb.LGBMRegressor(**PTS_REG_PARAMS)
        reg.fit(X_train.iloc[tr_idx], y_pts_train.iloc[tr_idx])
        preds = reg.predict(X_train.iloc[te_idx])
        cv_pts_mse.append(mean_squared_error(y_pts_train.iloc[te_idx], preds))

        print(f"  Fold {fold+1}: AUC={auc:.4f}, MSE={cv_pts_mse[-1]:.2f}")

    print(f"\nCV Averages:")
    print(f"  Dir AUC: {np.mean(cv_dir_auc):.4f} ± {np.std(cv_dir_auc):.4f}")
    print(f"  Pts MSE: {np.mean(cv_pts_mse):.2f}")

    # Final models on full training set
    print("\nTraining final models on full training set...")
    dir_clf = lgb.LGBMClassifier(**DIR_CLF_PARAMS)
    dir_clf.fit(X_train, y_dir_train)

    pts_reg = lgb.LGBMRegressor(**PTS_REG_PARAMS)
    pts_reg.fit(X_train, y_pts_train)

    # --- Overall validation ---
    print("\nValidation set evaluation (全天):")
    X_val      = val_data['X']
    y_dir_val  = val_data['y_dir']
    y_pts_val  = val_data['y_pts']

    dir_probs  = dir_clf.predict_proba(X_val)
    dir_preds  = dir_clf.predict(X_val)
    try:
        val_dir_auc = roc_auc_score(y_dir_val, dir_probs[:, 1])
    except ValueError:
        val_dir_auc = 0.5
    val_dir_acc = accuracy_score(y_dir_val, dir_preds)
    print(f"  Dir Accuracy: {val_dir_acc:.4f}")
    print(f"  Dir AUC:      {val_dir_auc:.4f}")

    pts_preds   = pts_reg.predict(X_val)
    val_pts_mse = mean_squared_error(y_pts_val, pts_preds)
    val_pts_mae = mean_absolute_error(y_pts_val, pts_preds)
    print(f"  Pts MSE:      {val_pts_mse:.2f}")
    print(f"  Pts MAE:      {val_pts_mae:.2f} points")

    # --- Morning+Midday 合併 AUC（主要部署指標）---
    val_tp = val_df['Time_Progress']
    mm_mask = (val_tp < AFTERNOON_CUTOFF).values

    # Morning+Midday 範圍的 MAE（部署指標）
    val_mm_mae = mean_absolute_error(y_pts_val[mm_mask], pts_preds[mm_mask])
    print(f"  MM Pts MAE:   {val_mm_mae:.2f} points (Morning+Midday)")
    try:
        val_mm_auc = roc_auc_score(y_dir_val[mm_mask], dir_probs[mm_mask, 1])
    except ValueError:
        val_mm_auc = 0.5
    print(f"\n  Morning+Midday AUC (部署指標): {val_mm_auc:.4f}  (n={mm_mask.sum()})")

    # --- 分時段 AUC 信心分析 ---
    print("\n[時段信心分析 — 單一模型各時段 AUC]")
    segments = {
        'Morning  (0.00~0.30)': val_tp < MORNING_CUTOFF,
        'Midday   (0.30~0.70)': (val_tp >= MORNING_CUTOFF) & (val_tp < AFTERNOON_CUTOFF),
        'Afternoon(0.70~1.00)': val_tp >= AFTERNOON_CUTOFF,
    }
    seg_aucs = {}
    for seg_name, mask in segments.items():
        seg_mask = mask.values
        if seg_mask.sum() < 20:
            print(f"  {seg_name}: 樣本不足 ({seg_mask.sum()})")
            seg_aucs[seg_name] = 0.5
            continue
        try:
            seg_auc = roc_auc_score(y_dir_val[seg_mask], dir_probs[seg_mask, 1])
        except ValueError:
            seg_auc = 0.5
        seg_aucs[seg_name] = seg_auc
        n = seg_mask.sum()
        print(f"  {seg_name}: AUC={seg_auc:.4f}  (n={n})")

    # 信心曲線（超額 AUC 歸一化，Morning=1.0）
    print("\n[信心曲線 — 基於超額 AUC 歸一化]")
    excess_aucs = {k: max(v - 0.5, 0.0) for k, v in seg_aucs.items()}
    max_excess = max(excess_aucs.values()) if max(excess_aucs.values()) > 0 else 1.0
    conf_scores = {}
    print(f"  {'時段':<30} {'超額AUC':>10} {'信心分數':>10}  建議")
    for seg_name, excess in excess_aucs.items():
        conf = excess / max_excess
        conf_scores[seg_name] = conf
        advice = "可信" if conf >= 0.5 else ("謹慎" if conf >= 0.2 else "忽略")
        print(f"  {seg_name:<30} {excess:>10.4f} {conf:>10.2f}  {advice}")

    # 動態產生信心建議（基於實測數據，非硬編碼）
    seg_list = list(conf_scores.items())
    morning_conf = seg_list[0][1]
    midday_conf  = seg_list[1][1]
    afternoon_conf = seg_list[2][1]
    print("\n  實作建議：prediction_confidence = f(Time_Progress)")
    print(f"    if Time_Progress < {MORNING_CUTOFF:.2f}:  confidence = {morning_conf:.2f}  (Morning)")
    print(f"    if Time_Progress < {AFTERNOON_CUTOFF:.2f}:  confidence = {midday_conf:.2f}  (Midday)")
    print(f"    else:                     confidence = {afternoon_conf:.2f}  (Afternoon)")

    # Feature importance
    print("\n[Feature Importance - Direction Classifier]")
    feat_names = X_train.columns.tolist()
    for feat, imp in sorted(zip(feat_names, dir_clf.feature_importances_), key=lambda x: -x[1])[:10]:
        print(f"  {feat}: {imp:.4f}")

    return {
        'val_dir_auc': val_dir_auc,
        'val_mm_auc': val_mm_auc,
        'val_dir_acc': val_dir_acc,
        'val_pts_mse': val_pts_mse,
        'val_pts_mae': val_pts_mae,
        'cv_dir_auc_mean': np.mean(cv_dir_auc),
        'cv_pts_mse_mean': np.mean(cv_pts_mse),
        'val_mm_mae': val_mm_mae,
        '_dir_clf': dir_clf,    # 供 main 儲存用
        '_pts_reg': pts_reg,
    }


# ---------------------------------------------------------------------------
# D. Composite Metric (AI: feel free to modify weights/formula)
# ---------------------------------------------------------------------------

def composite_metric(metrics):
    """
    Compute a single "north star" metric. LOWER IS BETTER.

    使用全天 AUC 作為方向損失（跨實驗可比）。
    注意：Morning+Midday AUC (val_mm_auc) 另行輸出作為部署指標參考。

    - Direction loss:    (1 - val_dir_auc)  → 全天 AUC（跨實驗可比）
    - Points loss:       MAE / 100          → normalized by typical move (~50-100 pts)

    Weights:
    - 50% direction (most important for entry signal)
    - 50% magnitude (for position sizing)
    """
    dir_loss = 1.0 - metrics['val_dir_auc']   # 全天 AUC（跨實驗可比）
    pts_loss = metrics['val_pts_mae'] / 100.0  # 全天 MAE（跨實驗可比）

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
    print(f"val_mm_auc:       {metrics['val_mm_auc']:.6f}")
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

    # --- 儲存模型到 models/ ---
    Config.ensure_directories()
    joblib.dump(metrics['_dir_clf'], Config.INTRADAY_CLF_LGBM_PATH)
    joblib.dump(metrics['_pts_reg'], Config.INTRADAY_REG_LGBM_PATH)
    print(f"\nModels saved:")
    print(f"  {Config.INTRADAY_CLF_LGBM_PATH}")
    print(f"  {Config.INTRADAY_REG_LGBM_PATH}")
