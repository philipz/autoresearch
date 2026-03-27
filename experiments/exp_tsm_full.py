"""
實驗（完整版）：2330.TW 1分K → 5分K 特徵 vs 台指期 snapshot cache 完整比較
===========================================================================
資料：
  - data/2330/*.csv  (1分K, 2024-04 ~ 2026-03)
  - autoresearch_sandbox/.cache/intraday_snapshots.csv  (snapshot 2024-01 ~ 2025-12)

方法：
  - 將 2330 1m resample → 5m，計算盤中特徵
  - 與 snapshot cache 按 [TradingDate, Time] 合併
  - A/B date-aware CV 比較（同超參數）

執行：
    cd /path/to/futures-research
    ./venv/bin/python3 autoresearch_sandbox/experiments/exp_tsm_full.py
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
TSM_DIR       = os.path.join(PROJECT_ROOT, 'data', '2330')
SNAPSHOT_FILE = os.path.join(PROJECT_ROOT, 'autoresearch_sandbox', '.cache', 'intraday_snapshots.csv')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LGBM_PARAMS = dict(
    n_estimators=800, max_depth=5, num_leaves=31, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    reg_alpha=1.0, reg_lambda=5.0, random_state=RANDOM_SEED, verbose=-1,
)
N_SPLITS = 5


# ── Step 1：載入並整合所有 2330 CSV，resample 成 5m ───────────────────────
def load_tsm_1m() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(TSM_DIR, '2330-*.csv')))
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding='utf-8-sig')  # handle BOM
        df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close',
                      'SMA8', 'SMA17', 'SMA34', 'SMA68', 'Volume', 'MA40', 'MA80']
        df = df[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)

    # Parse datetime
    raw['Datetime'] = pd.to_datetime(raw['Date'].str.replace('/', '-') + ' ' + raw['Time'])
    raw = raw.sort_values('Datetime').reset_index(drop=True)
    raw['TradingDate'] = raw['Datetime'].dt.strftime('%Y-%m-%d')

    print(f"2330 1m loaded: {len(raw):,} rows, {raw['TradingDate'].min()} ~ {raw['TradingDate'].max()}")
    return raw


def resample_to_5m(raw: pd.DataFrame) -> pd.DataFrame:
    """1m → 5m OHLCV（每個交易日獨立 resample，避免跨日污染）"""
    records = []
    for date, grp in raw.groupby('TradingDate'):
        grp = grp.set_index('Datetime').sort_index()
        grp5 = grp[['Open', 'High', 'Low', 'Close', 'Volume']].resample('5min', label='left').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna(subset=['Close'])
        grp5 = grp5[grp5['Volume'] > 0]  # 移除零成交量的虛假 bar

        if len(grp5) < 5:
            continue

        open_price = grp5['Open'].iloc[0]
        if open_price == 0:
            continue

        if not grp5.empty:
            grp5['TradingDate'] = date
            grp5['Time_HM'] = grp5.index.strftime('%H:%M')
            grp5['TSM_DayOpen'] = open_price
            grp5.rename(columns={
                'Open': 'TSM_Open', 'High': 'TSM_High', 'Low': 'TSM_Low',
                'Close': 'TSM_Close', 'Volume': 'TSM_Volume'
            }, inplace=True)
            records.append(grp5)

    df5 = pd.concat(records, ignore_index=True)
    print(f"2330 5m resampled: {len(df5):,} rows, {df5['TradingDate'].nunique()} days")
    return df5


def compute_tsm_features(df5: pd.DataFrame) -> pd.DataFrame:
    """計算盤中 TSM 特徵（在 5m 粒度上）"""
    def _compute_day(grp):
        open_p = grp['TSM_DayOpen'].iloc[0]
        closes = grp['TSM_Close']
        vols   = grp['TSM_Volume']
        
        grp = grp.copy()
        grp['TSM_Intraday_Ret'] = (closes - open_p) / open_p
        
        prev_closes = closes.shift(1).fillna(closes.iloc[0])
        grp['TSM_5m_Ret'] = np.where(prev_closes != 0, (closes - prev_closes) / prev_closes, 0.0)
        
        closes_lag3 = closes.shift(3).fillna(closes.iloc[0])
        grp['TSM_Mom3'] = (closes - closes_lag3) / open_p
        
        avg_vols = vols.rolling(window=20, min_periods=1).mean()
        grp['TSM_Vol_Ratio'] = np.where(avg_vols > 0, vols / avg_vols, 1.0)
        
        return grp[['TradingDate', 'Time_HM', 'TSM_5m_Ret', 'TSM_Intraday_Ret', 'TSM_Mom3', 'TSM_Vol_Ratio']]

    feat = df5.groupby('TradingDate', group_keys=False).apply(_compute_day)
    print(f"TSM features computed: {len(feat):,} rows")
    return feat


# ── Step 2：載入 snapshot cache ────────────────────────────────────────────
def load_snapshots() -> pd.DataFrame:
    df = pd.read_csv(SNAPSHOT_FILE)

    # 確認必要欄位
    needed = ['TradingDate', 'Time', 'Remaining_Dir', 'Time_Progress',
              'Intraday_Ret_Now', 'VWAP_Dist', 'Vol_Ratio', 'Day_Range',
              'Range_Position', 'Bar_Body', 'Bar_Range', 'Mom3', 'Mom6',
              'Cum_Volume']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"WARNING: snapshot cache 缺少欄位：{missing}")
        return pd.DataFrame()

    df = df[needed].dropna(subset=['Remaining_Dir']).copy()
    df['Log_Cum_Volume'] = np.log1p(df['Cum_Volume'])
    df['TradingDate'] = df['TradingDate'].astype(str)
    print(f"Snapshot cache: {len(df):,} rows, {df['TradingDate'].nunique()} days "
          f"({df['TradingDate'].min()} ~ {df['TradingDate'].max()})")
    return df


# ── Step 3：合併 ───────────────────────────────────────────────────────────
def merge_all(snap: pd.DataFrame, tsm_feat: pd.DataFrame) -> pd.DataFrame:
    snap = snap.copy()
    snap['Time_HM'] = snap['Time'].str[:5]

    merged = pd.merge(snap, tsm_feat, on=['TradingDate', 'Time_HM'], how='left')

    tsm_cols = [c for c in tsm_feat.columns if c not in ['TradingDate', 'Time_HM']]
    merged[tsm_cols] = merged[tsm_cols].fillna(0.0)

    # 衍生：相對強弱（需要 Intraday_Ret_Now 存在）
    if 'TSM_Intraday_Ret' in merged.columns and 'Intraday_Ret_Now' in merged.columns:
        merged['TSM_vs_TX'] = merged['TSM_Intraday_Ret'] - merged['Intraday_Ret_Now']

    # 有 TSM 資料的 rows 比例
    has_tsm = (merged['TSM_5m_Ret'] != 0) | (merged['TSM_Intraday_Ret'] != 0)
    print(f"Merged: {len(merged):,} rows, TSM 資料覆蓋率 {has_tsm.mean()*100:.1f}%")
    print(f"有效重疊：{merged[has_tsm]['TradingDate'].min()} ~ {merged[has_tsm]['TradingDate'].max()}")
    return merged


# ── Step 4：CV 評估 ────────────────────────────────────────────────────────
BASELINE_FEATURES = [
    'Time_Progress', 'Intraday_Ret_Now', 'VWAP_Dist', 'Vol_Ratio',
    'Day_Range', 'Range_Position', 'Bar_Body', 'Bar_Range',
    'Mom3', 'Mom6', 'Log_Cum_Volume',
]
TSM_EXTRA_FEATURES = [
    'TSM_5m_Ret', 'TSM_Intraday_Ret', 'TSM_Mom3', 'TSM_Vol_Ratio', 'TSM_vs_TX',
]


def run_cv(df: pd.DataFrame, features: list, label: str) -> dict:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    unique_dates = np.array(sorted(df['TradingDate'].unique()))

    auc_scores, morning_auc, midday_auc, afternoon_auc = [], [], [], []

    for tr_idx, te_idx in tscv.split(unique_dates):
        tr_dates = set(unique_dates[tr_idx])
        te_dates = set(unique_dates[te_idx])

        tr = df[df['TradingDate'].isin(tr_dates)]
        te = df[df['TradingDate'].isin(te_dates)]

        X_tr = tr[features].fillna(0)
        y_tr = tr['Remaining_Dir']
        X_te = te[features].fillna(0)
        y_te = te['Remaining_Dir']

        clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]

        try:
            auc_scores.append(roc_auc_score(y_te, probs))
        except ValueError:
            auc_scores.append(0.5)

        te_prog = te['Time_Progress'].values
        for seg, mask in [
            ('morning',   te_prog < 0.30),
            ('midday',    (te_prog >= 0.30) & (te_prog < 0.70)),
            ('afternoon', te_prog >= 0.70),
        ]:
            if mask.sum() < 20:
                continue
            try:
                seg_auc = roc_auc_score(y_te.values[mask], probs[mask])
            except ValueError:
                seg_auc = 0.5
            if seg == 'morning':   morning_auc.append(seg_auc)
            elif seg == 'midday':  midday_auc.append(seg_auc)
            else:                  afternoon_auc.append(seg_auc)

    return {
        'label':         label,
        'cv_auc_mean':   np.mean(auc_scores),
        'cv_auc_std':    np.std(auc_scores),
        'morning_auc':   np.mean(morning_auc)   if morning_auc   else float('nan'),
        'midday_auc':    np.mean(midday_auc)    if midday_auc    else float('nan'),
        'afternoon_auc': np.mean(afternoon_auc) if afternoon_auc else float('nan'),
        'n_features':    len(features),
        'n_rows':        len(df),
        'n_days':        df['TradingDate'].nunique(),
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("實驗（完整版）：2330 1m→5m 特徵對台指期盤中預測的影響")
    print("=" * 68)

    # 1. Load TSM 1m data
    raw1m   = load_tsm_1m()
    df5m    = resample_to_5m(raw1m)
    tsm_feat = compute_tsm_features(df5m)

    # 2. Load snapshot cache
    snap = load_snapshots()
    if snap.empty:
        print("ERROR: snapshot cache 無法載入"); return

    # 3. Merge
    df = merge_all(snap, tsm_feat)

    # Only use rows where TSM data is available (to keep A/B on same data)
    has_tsm = (df['TSM_5m_Ret'] != 0) | (df['TSM_Intraday_Ret'] != 0)
    df_overlap = df[has_tsm].copy()
    print(f"\n使用有 TSM 覆蓋的資料：{len(df_overlap):,} rows / {df_overlap['TradingDate'].nunique()} 天")

    avail_tsm = [f for f in TSM_EXTRA_FEATURES if f in df_overlap.columns]

    print(f"\n{'─'*68}")
    print(f"{'Model':<38} {'CV AUC':>8} {'±Std':>7} {'Morning':>8} {'Midday':>8}")
    print(f"{'─'*68}")

    results = []
    for features, label in [
        (BASELINE_FEATURES,           'A: Baseline (TX only)'),
        (BASELINE_FEATURES + avail_tsm, 'B: Enhanced (+TSM 2330)'),
    ]:
        r = run_cv(df_overlap, features, label)
        results.append(r)
        print(f"{r['label']:<38} {r['cv_auc_mean']:>8.4f} {r['cv_auc_std']:>7.4f} "
              f"{r['morning_auc']:>8.4f} {r['midday_auc']:>8.4f}")

    print(f"{'─'*68}")

    delta         = results[1]['cv_auc_mean'] - results[0]['cv_auc_mean']
    delta_morning = results[1]['morning_auc'] - results[0]['morning_auc']

    print(f"\n△ CV AUC     (B − A) : {delta:+.4f}")
    print(f"△ Morning AUC (B − A) : {delta_morning:+.4f}")
    print(f"\n資料範圍：{df_overlap['TradingDate'].min()} ~ {df_overlap['TradingDate'].max()}")
    print(f"訓練天數：{df_overlap['TradingDate'].nunique()} 天 / {len(df_overlap):,} 筆")

    threshold = 0.003
    if delta >= threshold:
        print(f"\n✅ 顯著提升（△≥{threshold}）→ 建議整合進 prepare_intraday.py 並重新訓練模型")
    elif delta > 0:
        print(f"\n⚠️  小幅提升但未達門檻 → 建議加入更多 TSM 特徵後再評估")
    else:
        print(f"\n❌ 未改善 → 不建議加入")

    # Feature importance
    print("\nTop 12 特徵重要性（Model B, 全量訓練）：")
    feat_b = BASELINE_FEATURES + avail_tsm
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(df_overlap[feat_b].fillna(0), df_overlap['Remaining_Dir'])
    imp = pd.Series(clf.feature_importances_, index=feat_b).sort_values(ascending=False)
    for f, v in imp.head(12).items():
        marker = ' ◀ TSM' if f in avail_tsm else ''
        print(f"  {f:<32} {v:>6}{marker}")

    # Per-segment detailed breakdown
    print("\n時段別 AUC 詳細對比：")
    print(f"  {'時段':<12} {'Baseline':>10} {'Enhanced':>10} {'Delta':>8}")
    for seg in ['morning_auc', 'midday_auc', 'afternoon_auc']:
        label = seg.replace('_auc', '')
        a = results[0][seg]
        b = results[1][seg]
        d = b - a
        print(f"  {label:<12} {a:>10.4f} {b:>10.4f} {d:>+8.4f}")


if __name__ == '__main__':
    main()
