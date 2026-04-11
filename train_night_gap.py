"""
Autoresearch Sandbox — Night Session Gap Model Training Script (AI MODIFIABLE)

預測台指期日盤開盤跳空方向（Gap_Direction）與點數（Open_Gap）。
使用夜盤 OHLCV 聚合特徵 + 靜態日線特徵。

Usage:
    uv run train_night_gap.py > run_night_gap.log 2>&1

Composite score (LOWER IS BETTER):
    composite_score = 0.5 × (1 - val_gap_auc) + 0.5 × (val_gap_mae / 100)
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

from prepare_night_gap import get_night_gap_data, TIME_BUDGET, RANDOM_SEED

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)

t_start = time.time()

# ---------------------------------------------------------------------------
# A. Feature Engineering (AI: feel free to modify)
# ---------------------------------------------------------------------------

train_df, val_df, test_df = get_night_gap_data()


def engineer_features(df):
    """
    Transform night session daily features into model-ready features.
    AI agent: modify this function to try different feature combinations.

    Available raw columns from prepare_night_gap.py:
        Night_Open, Night_High, Night_Low, Night_Close, Night_Vol, Night_Bars
        Night_Ret_vs_Day, Night_Body, Night_Range, Night_Vol_Change, Night_Gap_Open
        TSM_Ret, SOX_Ret, NetOI_Diff_lag1, NetOI_Diff_lag2
        TX_Ret, TX_Ret_lag1, TX_Vol5, TX_Mom5
        RSI, MA5, MA20, TSM_Ret_lag2, SOX_Ret_lag2, TSM_SOX_Spread, Intra_Ret_lag1

    Targets (do NOT use as features — data leakage!):
        Gap_Direction, Open_Gap

    Returns:
        dict with keys:
            'X_clf': features for direction classifier
            'X_reg': features for gap points regressor
            'y_dir': target for gap direction (binary)
            'y_gap': target for gap points (regression)
    """
    # --- Baseline features (mirrors NightSessionPredictor) ---
    X = pd.DataFrame(index=df.index)

    # Core night session signals
    X['Night_Ret_vs_Day']  = df['Night_Ret_vs_Day']    # total overnight return
    X['Night_Range']       = df['Night_Range']          # night session volatility
    X['Night_Vol_Change']  = df['Night_Vol_Change']     # volume momentum
    X['Night_Gap_Open']    = df['Night_Gap_Open']       # 夜盤自身跳空

    # US market signals (T-1 shifted, no leakage)
    X['TSM_Ret']           = df['TSM_Ret']
    X['SOX_Ret']           = df['SOX_Ret']
    X['NetOI_Diff_lag1']   = df['NetOI_Diff_lag1']

    # Market momentum & volatility regime
    X['TX_Vol5']           = df['TX_Vol5']
    X['TX_Mom5']           = df['TX_Mom5']

    # Cross-asset interactions
    X['TSM_x_Night_Ret']   = df['TSM_Ret'] * df['Night_Ret_vs_Day']
    X['SOX_x_Night_Ret']   = df['SOX_Ret'] * df['Night_Ret_vs_Day']
    X['Night_Range_x_Vol'] = df['Night_Range'] * df['Night_Vol_Change']
    X['TSM_SOX_Spread']    = df['TSM_SOX_Spread']

    # Non-linear transforms (magnitude signals for regressor)
    X['Night_Range_sq']    = df['Night_Range'] ** 2
    X['Night_Ret_abs']     = np.abs(df['Night_Ret_vs_Day'])
    X['TSM_Ret_abs']       = np.abs(df['TSM_Ret'])
    X['SOX_Ret_abs']       = np.abs(df['SOX_Ret'])

    # Lag features (momentum persistence)
    X['TSM_Ret_lag2']      = df['TSM_Ret_lag2']
    X['SOX_Ret_lag2']      = df['SOX_Ret_lag2']
    X['TX_Ret_lag1']       = df['TX_Ret_lag1']
    X['Intra_Ret_lag1']    = df['Intra_Ret_lag1']
    X['NetOI_Diff_lag2']   = df['NetOI_Diff_lag2']

    # Institutional flow × night move interaction
    X['NetOI_x_Night_Ret'] = df['NetOI_Diff_lag1'] * df['Night_Ret_vs_Day']
    X['NetOI_momentum']    = df['NetOI_Diff_lag1'] - df['NetOI_Diff_lag2']  # 籌碼加速度
    X['Vol5_x_Night_Range']= df['TX_Vol5'] * df['Night_Range']              # 波動機制 × 夜盤振幅

    # Momentum interactions (趨勢共振)
    X['TXMom5_x_Night_Ret']= df['TX_Mom5'] * df['Night_Ret_vs_Day']        # 市場趨勢 × 夜盤報酬
    X['Intra_x_Night_Ret'] = df['Intra_Ret_lag1'] * df['Night_Ret_vs_Day'] # 昨日盤中慣性 × 夜盤
    X['TSM_lag2_x_SOX_lag2'] = df['TSM_Ret_lag2'] * df['SOX_Ret_lag2']     # 2日前跨資產
    X['Night_Gap_Open_sq'] = df['Night_Gap_Open'] ** 2                      # 夜盤跳空非線性

    # More regime interactions (波動機制 × 信號強度)
    X['SOX_x_Vol5']        = df['SOX_Ret'] * df['TX_Vol5']                  # SOX × 波動機制
    X['Night_Ret_x_Vol5']  = df['Night_Ret_vs_Day'] * df['TX_Vol5']        # 夜盤幅度 × 市場波動
    X['NetOI_x_TXMom5']   = df['NetOI_Diff_lag1'] * df['TX_Mom5']          # 籌碼 × 市場趨勢

    X.fillna(0, inplace=True)

    leaky = ['Gap_Direction', 'Open_Gap']
    for c in leaky:
        assert c not in X.columns, f"Data leakage: {c}"

    y_dir = df['Gap_Direction'].copy()
    y_gap = df['Open_Gap'].copy()

    return {'X_clf': X, 'X_reg': X, 'y_dir': y_dir, 'y_gap': y_gap}


# ---------------------------------------------------------------------------
# B. Model Definition & Hyperparameters (AI: feel free to modify)
# ---------------------------------------------------------------------------

# Gap Direction Classifier (LightGBM)
CLF_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.5,
    reg_lambda=5.0,
    bagging_freq=5,
    bagging_fraction=0.8,
    random_state=RANDOM_SEED,
    verbose=-1,
)

# Gap Points Regressor (LightGBM)
REG_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.5,
    reg_lambda=5.0,
    bagging_freq=5,
    bagging_fraction=0.8,
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
    Train with TimeSeriesSplit CV, then evaluate on val_df.
    Returns dict of metrics.
    """
    train_data = engineer_features(train_df)
    val_data   = engineer_features(val_df)

    X_clf_train = train_data['X_clf']
    X_reg_train = train_data['X_reg']
    y_dir_train = train_data['y_dir']
    y_gap_train = train_data['y_gap']

    print(f"\nRunning {N_SPLITS}-fold TimeSeriesSplit CV...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    cv_auc, cv_mae = [], []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_clf_train)):
        # Direction classifier
        clf = lgb.LGBMClassifier(**CLF_PARAMS)
        clf.fit(X_clf_train.iloc[tr_idx], y_dir_train.iloc[tr_idx])
        probs = clf.predict_proba(X_clf_train.iloc[te_idx])
        try:
            auc = roc_auc_score(y_dir_train.iloc[te_idx], probs[:, 1]) if probs.shape[1] == 2 else 0.5
        except ValueError:
            auc = 0.5
        cv_auc.append(auc)

        # Gap regressor
        reg = lgb.LGBMRegressor(**REG_PARAMS)
        reg.fit(X_reg_train.iloc[tr_idx], y_gap_train.iloc[tr_idx])
        preds = reg.predict(X_reg_train.iloc[te_idx])
        mae = mean_absolute_error(y_gap_train.iloc[te_idx], preds)
        cv_mae.append(mae)

        print(f"  Fold {fold+1}: AUC={auc:.4f}, MAE={mae:.2f} pts")

    print(f"\nCV Averages:")
    print(f"  Gap AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")
    print(f"  Gap MAE: {np.mean(cv_mae):.2f} ± {np.std(cv_mae):.2f} pts")

    # Final models on full training set
    print("\nTraining final models on full training set...")
    gap_clf = lgb.LGBMClassifier(**CLF_PARAMS)
    gap_clf.fit(X_clf_train, y_dir_train)

    gap_reg = lgb.LGBMRegressor(**REG_PARAMS)
    gap_reg.fit(X_reg_train, y_gap_train)

    # Validation evaluation
    print("\nValidation set evaluation:")
    X_clf_val = val_data['X_clf']
    X_reg_val = val_data['X_reg']
    y_dir_val = val_data['y_dir']
    y_gap_val = val_data['y_gap']

    gap_probs = gap_clf.predict_proba(X_clf_val)
    gap_preds = gap_clf.predict(X_clf_val)
    try:
        val_gap_auc = roc_auc_score(y_dir_val, gap_probs[:, 1])
    except ValueError:
        val_gap_auc = 0.5
    val_gap_acc = accuracy_score(y_dir_val, gap_preds)
    print(f"  Gap Accuracy: {val_gap_acc:.4f}")
    print(f"  Gap AUC:      {val_gap_auc:.4f}")

    gap_val_preds = gap_reg.predict(X_reg_val)
    val_gap_mae   = mean_absolute_error(y_gap_val, gap_val_preds)
    val_gap_mse   = mean_squared_error(y_gap_val, gap_val_preds)
    print(f"  Gap MAE:      {val_gap_mae:.2f} pts")
    print(f"  Gap MSE:      {val_gap_mse:.2f}")

    # Feature importance
    print("\n[Top 10 Feature Importance — Direction Classifier]")
    for feat, imp in sorted(
        zip(X_clf_train.columns, gap_clf.feature_importances_), key=lambda x: -x[1]
    )[:10]:
        print(f"  {feat}: {imp:.4f}")

    return {
        'val_gap_auc':      val_gap_auc,
        'val_gap_acc':      val_gap_acc,
        'val_gap_mae':      val_gap_mae,
        'val_gap_mse':      val_gap_mse,
        'cv_gap_auc_mean':  np.mean(cv_auc),
        'cv_gap_mae_mean':  np.mean(cv_mae),
    }


# ---------------------------------------------------------------------------
# D. Composite Metric (AI: feel free to modify weights/formula)
# ---------------------------------------------------------------------------

def composite_metric(metrics):
    """
    Single north-star metric. LOWER IS BETTER.

    Formula:
      0.5 × (1 - AUC)     — direction loss (affects entry decision)
      0.5 × MAE / 100     — magnitude loss (affects position sizing)

    Weights reflect equal importance of direction and point accuracy for profitability.
    """
    dir_loss = 1.0 - metrics['val_gap_auc']
    mag_loss = metrics['val_gap_mae'] / 100.0
    return 0.5 * dir_loss + 0.5 * mag_loss


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Autoresearch Sandbox — Night Gap Model Training")
    print("=" * 60)

    t_train_start = time.time()
    metrics = train_and_evaluate_cv(train_df, val_df)
    score   = composite_metric(metrics)
    t_end   = time.time()

    print()
    print("---")
    print(f"composite_score:  {score:.6f}")
    print(f"val_gap_auc:      {metrics['val_gap_auc']:.6f}")
    print(f"val_gap_acc:      {metrics['val_gap_acc']:.6f}")
    print(f"val_gap_mae:      {metrics['val_gap_mae']:.2f}")
    print(f"val_gap_mse:      {metrics['val_gap_mse']:.2f}")
    print(f"cv_gap_auc_mean:  {metrics['cv_gap_auc_mean']:.6f}")
    print(f"cv_gap_mae_mean:  {metrics['cv_gap_mae_mean']:.2f}")
    print(f"training_seconds: {t_end - t_train_start:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"n_splits:         {N_SPLITS}")
    print(f"train_rows:       {len(train_df)}")
    print(f"val_rows:         {len(val_df)}")
