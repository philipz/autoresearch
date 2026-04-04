"""
Night Intraday Model Training

使用夜盤 5 分 K 快照資料訓練 LightGBM 分類器（方向）與回歸器（點數）。
輸出：models/night_intraday_clf_lgbm.pkl, models/night_intraday_reg_lgbm.pkl,
      models/night_intraday_feature_cols.pkl

夜盤時段邊界（Time_Progress）：
  EARLY_CUTOFF  = 0.35  ≈ 15:00 + 294min = 19:54
  LATE_CUTOFF   = 0.65  ≈ 15:00 + 546min = 00:06
"""

import time, warnings, sys, os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from prepare_night_intraday import get_night_intraday_data, TIME_BUDGET, RANDOM_SEED
except ImportError:
    from autoresearch_sandbox.prepare_night_intraday import get_night_intraday_data, TIME_BUDGET, RANDOM_SEED

try:
    from src.config import Config
except ImportError:
    Config = None

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)

t_start = time.time()

EARLY_CUTOFF = 0.35
LATE_CUTOFF  = 0.65

# ─────────────── A. Feature Engineering ───────────────

def check_feature_availability(df: pd.DataFrame) -> None:
    """夜盤靜態特徵可用性檢查，在訓練前呼叫，防止靜默降級。"""
    missing_groups = []
    night_static_cols = ['Night_TSM_Ret', 'Night_SOX_Ret', 'Night_NetOI_Diff_lag1']
    missing_static = [c for c in night_static_cols if c not in df.columns or df[c].eq(0).all()]
    if missing_static:
        missing_groups.append(f"夜盤靜態特徵（{missing_static}）")
    if missing_groups:
        print(f"\n[!] 警告：部分特徵群組缺失或全為 0，模型將以降級模式運行：{', '.join(missing_groups)}")
        print("    請確認 prepare_night_intraday.py 的靜態特徵對齊邏輯。\n")

def engineer_features(df: pd.DataFrame) -> dict:
    X = pd.DataFrame(index=df.index)

    # 核心動態特徵
    X['Intraday_Ret_Now'] = df['Intraday_Ret_Now']
    X['VWAP_Dist']        = df['VWAP_Dist']
    X['Vol_Ratio']        = df['Vol_Ratio']
    X['Day_Range']        = df['Day_Range']
    X['Range_Position']   = df['Range_Position']
    X['Time_Progress']    = df['Time_Progress']
    X['Bar_Body']         = df['Bar_Body']
    X['Bar_Range']        = df['Bar_Range']
    X['Mom3']             = df['Mom3']
    X['Mom6']             = df['Mom6']
    X['Log_Cum_Volume']   = np.log1p(df['Cum_Volume'])

    # 互動特徵
    X['Ret_x_Time']      = df['Intraday_Ret_Now'] * df['Time_Progress']
    X['VWAP_x_Time']     = df['VWAP_Dist']        * df['Time_Progress']
    X['Range_x_Time']    = df['Day_Range']         * df['Time_Progress']
    X['VWAP_x_RangePos'] = df['VWAP_Dist']         * df['Range_Position']
    X['Mom3_x_Time']     = df['Mom3']              * df['Time_Progress']
    X['Mom6_x_Time']     = df['Mom6']              * df['Time_Progress']
    X['Ret_abs_x_Time']  = np.abs(df['Intraday_Ret_Now']) * df['Time_Progress']
    X['Vol_x_Time']      = df['Vol_Ratio']         * df['Time_Progress']

    # 非線性變換
    X['Ret_abs']      = np.abs(df['Intraday_Ret_Now'])
    X['Log_Vol_Ratio'] = np.log1p(df['Vol_Ratio'].clip(lower=0))

    # 靜態特徵（T-1 日）
    for col in ['Night_TSM_Ret', 'Night_SOX_Ret', 'Night_NetOI_Diff_lag1',
                'Night_TX_Ret', 'Night_TSM_Ret_lag2', 'Night_SOX_Ret_lag2',
                'Night_TX_Vol5', 'Night_TX_Mom5']:
        if col in df.columns:
            X[col] = df[col]

    # 注意：夜盤無「開盤跳空」概念（Open_Gap），故不包含 Open_Gap_sq / Open_Gap_abs
    # 相關信號已由 Night_TSM_Ret、Night_SOX_Ret 等隔夜資訊取代。

    # 動能慣性（每日內滾動）
    # Ret_Persistence: 當日內 3-bar 報酬滾動和（groupby TradingDate 確保不跨日）
    # 注意：此值在 CV loop 中以整個 train_df 計算後用 mask 切分，
    # 因 groupby TradingDate 隔離跨日洩漏，但 rolling 在單日邊界內是正確的。
    X['Ret_Persistence'] = df.groupby('TradingDate')['Intraday_Ret_Now'].transform(
        lambda x: x.rolling(3).sum().fillna(0)
    )

    # 洩漏檢查
    for leaky in ['Remaining_Ret', 'Remaining_Dir', 'Remaining_Points']:
        assert leaky not in X.columns, f"Data leakage: {leaky}"

    X.fillna(0, inplace=True)

    return {
        'X':     X,
        'y_dir': df['Remaining_Dir'].copy(),
        'y_ret': df['Remaining_Ret'].copy(),
        'y_pts': df['Remaining_Points'].copy(),
    }


# ─────────────── B. Hyperparameters ───────────────

DIR_CLF_PARAMS = dict(
    n_estimators=1000, max_depth=5, num_leaves=31,
    learning_rate=0.01, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=1.0, reg_lambda=10.0,
    bagging_freq=5, bagging_fraction=0.8,
    random_state=RANDOM_SEED, verbose=-1,
)

PTS_REG_PARAMS = dict(
    n_estimators=1000, max_depth=5, num_leaves=31,
    learning_rate=0.01, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=1.0, reg_lambda=10.0,
    bagging_freq=5, bagging_fraction=0.8,
    random_state=RANDOM_SEED, verbose=-1,
)

N_SPLITS = 5


# ─────────────── C. Training & Evaluation ───────────────

def train_and_evaluate(train_df, val_df):
    train_data = engineer_features(train_df)
    val_data   = engineer_features(val_df)

    X_train     = train_data['X']
    y_dir_train = train_data['y_dir']
    y_pts_train = train_data['y_pts']

    print(f"\nRunning {N_SPLITS}-fold Date-Aware TimeSeriesSplit CV...")
    unique_dates = train_df['TradingDate'].unique()
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_dir_auc, cv_pts_mse = [], []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(unique_dates)):
        tr_dates = set(unique_dates[tr_idx])
        te_dates = set(unique_dates[te_idx])
        tr_mask  = train_df['TradingDate'].isin(tr_dates)
        te_mask  = train_df['TradingDate'].isin(te_dates)

        clf = lgb.LGBMClassifier(**DIR_CLF_PARAMS)
        clf.fit(X_train[tr_mask], y_dir_train[tr_mask])
        probs = clf.predict_proba(X_train[te_mask])
        try:
            auc = roc_auc_score(y_dir_train[te_mask], probs[:, 1]) if probs.shape[1] == 2 else 0.5
        except ValueError:
            auc = 0.5
        cv_dir_auc.append(auc)

        reg = lgb.LGBMRegressor(**PTS_REG_PARAMS)
        reg.fit(X_train[tr_mask], y_pts_train[tr_mask])
        preds = reg.predict(X_train[te_mask])
        cv_pts_mse.append(mean_squared_error(y_pts_train[te_mask], preds))
        print(f"  Fold {fold+1}: AUC={auc:.4f}, MSE={cv_pts_mse[-1]:.2f}")

    print(f"\nCV Averages: Dir AUC={np.mean(cv_dir_auc):.4f} ± {np.std(cv_dir_auc):.4f}, "
          f"Pts MSE={np.mean(cv_pts_mse):.2f}")

    print("\nTraining final models on full training set...")
    dir_clf = lgb.LGBMClassifier(**DIR_CLF_PARAMS)
    dir_clf.fit(X_train, y_dir_train)

    pts_reg = lgb.LGBMRegressor(**PTS_REG_PARAMS)
    pts_reg.fit(X_train, y_pts_train)

    # Validation
    X_val     = val_data['X']
    y_dir_val = val_data['y_dir']
    y_pts_val = val_data['y_pts']

    dir_probs = dir_clf.predict_proba(X_val)
    dir_preds = dir_clf.predict(X_val)
    try:
        val_dir_auc = roc_auc_score(y_dir_val, dir_probs[:, 1])
    except ValueError:
        val_dir_auc = 0.5
    val_dir_acc = accuracy_score(y_dir_val, dir_preds)
    pts_preds   = pts_reg.predict(X_val)
    val_pts_mae = mean_absolute_error(y_pts_val, pts_preds)
    val_pts_mse = mean_squared_error(y_pts_val, pts_preds)

    print(f"\nValidation: Dir Acc={val_dir_acc:.4f}, Dir AUC={val_dir_auc:.4f}, "
          f"Pts MAE={val_pts_mae:.2f} pts")

    # 分時段 AUC 信心分析
    print("\n[夜盤時段信心分析]")
    val_tp = val_df['Time_Progress']
    segments = {
        f'Early   (0.00~{EARLY_CUTOFF:.2f} ≈15:00~19:54)': val_tp < EARLY_CUTOFF,
        f'Evening ({EARLY_CUTOFF:.2f}~{LATE_CUTOFF:.2f} ≈19:54~00:06)': (val_tp >= EARLY_CUTOFF) & (val_tp < LATE_CUTOFF),
        f'Late    ({LATE_CUTOFF:.2f}~1.00 ≈00:06~05:00)': val_tp >= LATE_CUTOFF,
    }
    seg_aucs = {}
    for seg_name, mask in segments.items():
        seg_mask = mask.values
        if seg_mask.sum() < 20:
            seg_aucs[seg_name] = 0.5
            print(f"  {seg_name}: 樣本不足 ({seg_mask.sum()})")
            continue
        try:
            seg_auc = roc_auc_score(y_dir_val[seg_mask], dir_probs[seg_mask, 1])
        except ValueError:
            seg_auc = 0.5
        seg_aucs[seg_name] = seg_auc
        print(f"  {seg_name}: AUC={seg_auc:.4f} (n={seg_mask.sum()})")

    # Top 10 特徵重要性
    print("\n[Top 10 Feature Importance - Direction]")
    for feat, imp in sorted(zip(X_train.columns, dir_clf.feature_importances_), key=lambda x: -x[1])[:10]:
        print(f"  {feat}: {imp:.4f}")

    return {
        'val_dir_auc':     val_dir_auc,
        'val_dir_acc':     val_dir_acc,
        'val_pts_mse':     val_pts_mse,
        'val_pts_mae':     val_pts_mae,
        'cv_dir_auc_mean': np.mean(cv_dir_auc),
        'cv_pts_mse_mean': np.mean(cv_pts_mse),
        '_dir_clf':        dir_clf,
        '_pts_reg':        pts_reg,
        '_feature_cols':   X_train.columns.tolist(),
    }


# ─────────────── D. Composite Metric ───────────────

def composite_metric(metrics):
    """LOWER IS BETTER. 50% 方向損失 + 50% 點數損失。"""
    return 0.5 * (1.0 - metrics['val_dir_auc']) + 0.5 * metrics['val_pts_mae'] / 100.0


# ─────────────── Main ───────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Night Intraday Model Training")
    print("=" * 60)

    train_df, val_df, _ = get_night_intraday_data()

    check_feature_availability(train_df)

    t_train_start = time.time()
    metrics = train_and_evaluate(train_df, val_df)
    score   = composite_metric(metrics)
    t_end   = time.time()

    print()
    print("---")
    print(f"composite_score:  {score:.6f}")
    print(f"val_dir_auc:      {metrics['val_dir_auc']:.6f}")
    print(f"val_dir_acc:      {metrics['val_dir_acc']:.6f}")
    print(f"val_pts_mae:      {metrics['val_pts_mae']:.2f}")
    print(f"cv_dir_auc_mean:  {metrics['cv_dir_auc_mean']:.6f}")
    print(f"training_seconds: {t_end - t_train_start:.1f}")
    print(f"train_rows:       {len(train_df)}")
    print(f"val_rows:         {len(val_df)}")

    # 儲存模型
    if Config:
        Config.ensure_directories()
        clf_path  = Config.NIGHT_INTRADAY_CLF_PATH
        reg_path  = Config.NIGHT_INTRADAY_REG_PATH
        feat_path = Config.NIGHT_INTRADAY_FEAT_PATH
    else:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        clf_path  = os.path.join(models_dir, 'night_intraday_clf_lgbm.pkl')
        reg_path  = os.path.join(models_dir, 'night_intraday_reg_lgbm.pkl')
        feat_path = os.path.join(models_dir, 'night_intraday_feature_cols.pkl')

    joblib.dump(metrics['_dir_clf'], clf_path)
    joblib.dump(metrics['_pts_reg'], reg_path)
    joblib.dump(metrics['_feature_cols'], feat_path)
    print(f"\nModels saved: {clf_path}, {reg_path}, {feat_path}")
