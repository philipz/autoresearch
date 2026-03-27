"""
實驗：台積電 (2330.TW) + 0050.TW 盤中 5 分 K 特徵是否能提升台指期預測 AUC
============================================================================
資料範圍：yfinance 近 60 天 5m K + tx_5m_kbar.csv 對應期間
方法：相同資料、相同模型超參數，A/B 比較
  - Model A (Baseline)：僅使用 TX 期貨本身的盤中特徵
  - Model B (Enhanced)：加入 2330.TW + 0050.TW 盤中衍生特徵

執行：
    cd autoresearch_sandbox
    python experiments/exp_tsm_intraday.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
KBAR_FILE    = os.path.join(PROJECT_ROOT, 'data', 'raw', 'tx_5m_kbar.csv')

# ── 共用 LightGBM 超參數 ──────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators=800,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=1.0,
    reg_lambda=5.0,
    random_state=RANDOM_SEED,
    verbose=-1,
)
N_SPLITS = 5


# ── Step 1：從 tx_5m_kbar.csv 建立基礎盤中快照 ────────────────────────────
def build_tx_snapshots(min_date: str) -> pd.DataFrame:
    """
    從 tx_5m_kbar.csv 計算基礎 TX 快照特徵（與 prepare_intraday.py 邏輯一致）。
    min_date: 'YYYY-MM-DD' 格式，只取此日之後的資料。
    """
    tx = pd.read_csv(KBAR_FILE)
    tx['Date'] = pd.to_datetime(tx['Date'])
    tx = tx[tx['Session'] == 'day'].copy()
    tx = tx[tx['Date'] >= min_date].copy()

    results = []
    for date, grp in tx.groupby('Date'):
        grp = grp.sort_values('Time').reset_index(drop=True)
        if len(grp) < 5:
            continue

        open_price = grp['Open'].iloc[0]
        if open_price == 0:
            continue
        prev_close = grp['Close'].iloc[-1]  # will use shift below

        day_high = grp['High'].expanding().max()
        day_low  = grp['Low'].expanding().min()
        cum_vol  = grp['Volume'].cumsum()
        vwap     = (grp['Close'] * grp['Volume']).cumsum() / cum_vol.replace(0, np.nan)

        n = len(grp)
        session_start = pd.Timestamp(str(date.date()) + ' 08:45:00')
        session_end   = pd.Timestamp(str(date.date()) + ' 13:45:00')
        session_min   = (session_end - session_start).total_seconds() / 60  # 300 min

        for i, row in grp.iterrows():
            bar_time = pd.Timestamp(str(date.date()) + ' ' + row['Time'])
            elapsed  = (bar_time - session_start).total_seconds() / 60
            time_progress = max(0.0, min(1.0, elapsed / session_min))

            close = row['Close']
            intraday_ret_now = (close - open_price) / open_price if open_price != 0 else 0.0
            vwap_val = vwap.iloc[i]
            vwap_dist = (close - vwap_val) / vwap_val if (vwap_val and not np.isnan(vwap_val)) else 0.0

            avg_vol = grp['Volume'].iloc[max(0, i-19):i+1].mean()
            vol_ratio = row['Volume'] / avg_vol if avg_vol > 0 else 1.0

            hi = day_high.iloc[i]
            lo = day_low.iloc[i]
            day_range = (hi - lo) / open_price if open_price != 0 else 0.0
            span = hi - lo
            range_position = (close - lo) / span if span > 0 else 0.5

            bar_body  = (row['Close'] - row['Open']) / open_price if open_price != 0 else 0.0
            bar_range = (row['High']  - row['Low'])  / open_price if open_price != 0 else 0.0

            # Momentum: close relative to N bars ago
            mom3 = (close - grp['Close'].iloc[max(0, i-3)]) / open_price if i >= 3 else 0.0
            mom6 = (close - grp['Close'].iloc[max(0, i-6)]) / open_price if i >= 6 else 0.0

            # Target: remaining return to close
            remaining_close = grp['Close'].iloc[-1]
            remaining_ret   = (remaining_close - close) / close if close != 0 else 0.0
            remaining_dir   = int(remaining_ret > 0)

            results.append({
                'TradingDate':    str(date.date()),
                'Time':           row['Time'],
                'Time_Progress':  time_progress,
                'Intraday_Ret_Now': intraday_ret_now,
                'VWAP_Dist':      vwap_dist,
                'Vol_Ratio':      vol_ratio,
                'Day_Range':      day_range,
                'Range_Position': range_position,
                'Bar_Body':       bar_body,
                'Bar_Range':      bar_range,
                'Mom3':           mom3,
                'Mom6':           mom6,
                'Log_Cum_Volume': np.log1p(cum_vol.iloc[i]),
                'Current_Price':  close,
                'Open_Price':     open_price,
                'Remaining_Ret':  remaining_ret,
                'Remaining_Dir':  remaining_dir,
            })

    df = pd.DataFrame(results)
    # Exclude last bar (no remaining return)
    df = df[df['Time'] < '13:45'].copy()
    print(f"TX snapshots built: {len(df)} rows, {df['TradingDate'].nunique()} days")
    return df


# ── Step 2：下載 2330.TW + 0050.TW 5m K 並計算特徵 ────────────────────────
def fetch_tsm_features() -> pd.DataFrame:
    """
    下載台積電 + 0050 盤中 5 分 K，計算：
      TSM_5m_Ret, TSM_Intraday_Ret, TSM_Mom3, TSM_vs_TX_Spread 等
    以 [TradingDate, HH:MM] 為 key 回傳。
    """
    print("Downloading 2330.TW and 0050.TW 5m data from yfinance...")
    records = []

    for symbol, prefix in [('2330.TW', 'TSM'), ('0050.TW', 'ETF50')]:
        df = yf.download(symbol, period='60d', interval='5m',
                         progress=False, auto_adjust=True)
        if df.empty:
            print(f"  WARNING: {symbol} returned no data, skipping")
            continue

        # Flatten MultiIndex columns (yfinance v0.2+ returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = df.index.tz_convert('Asia/Taipei')

        for date, grp in df.groupby(df.index.date):
            grp = grp.sort_values(df.index.name if df.index.name else 'Datetime')
            open_price = grp['Close'].iloc[0]
            if open_price == 0:
                continue

            for j, (ts, row) in enumerate(grp.iterrows()):
                close = float(row['Close'])
                intraday_ret = (close - open_price) / open_price
                ret_5m = (close - float(grp['Close'].iloc[max(0, j-1)])) / float(grp['Close'].iloc[max(0, j-1)]) if j > 0 else 0.0
                mom3   = (close - float(grp['Close'].iloc[max(0, j-3)])) / open_price if j >= 3 else 0.0
                avg_vol = float(grp['Volume'].iloc[max(0, j-19):j+1].mean())
                vol_r  = float(row['Volume']) / avg_vol if avg_vol > 0 else 1.0

                records.append({
                    'TradingDate': str(date),
                    'Time_HM':     ts.strftime('%H:%M'),
                    f'{prefix}_5m_Ret':      ret_5m,
                    f'{prefix}_Intraday_Ret': intraday_ret,
                    f'{prefix}_Mom3':         mom3,
                    f'{prefix}_Vol_Ratio':    vol_r,
                })

    if not records:
        return pd.DataFrame()

    feat = pd.DataFrame(records)
    # Pivot: one row per (TradingDate, Time_HM), columns per symbol
    feat = feat.groupby(['TradingDate', 'Time_HM']).first().reset_index()
    print(f"TSM/ETF50 features: {len(feat)} rows, {feat['TradingDate'].nunique()} days")
    return feat


# ── Step 3：對齊時間戳 ───────────────────────────────────────────────────────
def align_and_merge(tx_df: pd.DataFrame, tsm_df: pd.DataFrame) -> pd.DataFrame:
    """
    tx_df 的 Time 格式為 'HH:MM:SS'，tsm_df 的 Time_HM 為 'HH:MM'。
    對齊：TSM 09:00 → TX 09:00 (or nearest ≤ bar)。
    TX 08:45 bar (before TSM opens) → TSM features = 0.
    """
    tx_df = tx_df.copy()
    tx_df['Time_HM'] = tx_df['Time'].str[:5]  # '08:45'

    merged = pd.merge(tx_df, tsm_df, on=['TradingDate', 'Time_HM'], how='left')

    # Fill TSM features with 0 where not available (before TSM opens)
    tsm_cols = [c for c in merged.columns if c.startswith('TSM_') or c.startswith('ETF50_')]
    merged[tsm_cols] = merged[tsm_cols].fillna(0.0)

    # Add cross-market spread (TSM vs TX momentum)
    if 'TSM_Intraday_Ret' in merged.columns:
        merged['TSM_vs_TX'] = merged['TSM_Intraday_Ret'] - merged['Intraday_Ret_Now']

    print(f"Merged dataset: {len(merged)} rows, {merged['TradingDate'].nunique()} trading days")
    return merged


# ── Step 4：特徵工程 ──────────────────────────────────────────────────────────
BASELINE_FEATURES = [
    'Time_Progress', 'Intraday_Ret_Now', 'VWAP_Dist', 'Vol_Ratio',
    'Day_Range', 'Range_Position', 'Bar_Body', 'Bar_Range',
    'Mom3', 'Mom6', 'Log_Cum_Volume',
]

TSM_EXTRA_FEATURES = [
    'TSM_5m_Ret', 'TSM_Intraday_Ret', 'TSM_Mom3', 'TSM_Vol_Ratio', 'TSM_vs_TX',
    'ETF50_5m_Ret', 'ETF50_Intraday_Ret', 'ETF50_Mom3',
]


# ── Step 5：Date-Aware CV 訓練與評估 ─────────────────────────────────────────
def run_cv(df: pd.DataFrame, features: list, label: str) -> dict:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    unique_dates = df['TradingDate'].unique()

    auc_scores, morning_auc, midday_auc, afternoon_auc = [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(unique_dates)):
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
            auc = roc_auc_score(y_te, probs)
        except ValueError:
            auc = 0.5
        auc_scores.append(auc)

        # Per-segment AUC
        for seg, mask, lo, hi in [
            ('morning',   te['Time_Progress'] < 0.30,  None, None),
            ('midday',    (te['Time_Progress'] >= 0.30) & (te['Time_Progress'] < 0.70), None, None),
            ('afternoon', te['Time_Progress'] >= 0.70, None, None),
        ]:
            sub_te = te[mask]
            if len(sub_te) < 20:
                continue
            try:
                seg_auc = roc_auc_score(sub_te['Remaining_Dir'], probs[mask.values])
            except ValueError:
                seg_auc = 0.5
            if seg == 'morning':   morning_auc.append(seg_auc)
            elif seg == 'midday':  midday_auc.append(seg_auc)
            else:                  afternoon_auc.append(seg_auc)

    result = {
        'label':       label,
        'cv_auc_mean': np.mean(auc_scores),
        'cv_auc_std':  np.std(auc_scores),
        'morning_auc': np.mean(morning_auc) if morning_auc else float('nan'),
        'midday_auc':  np.mean(midday_auc)  if midday_auc  else float('nan'),
        'afternoon_auc': np.mean(afternoon_auc) if afternoon_auc else float('nan'),
        'n_features':  len(features),
        'n_rows':      len(df),
        'n_days':      df['TradingDate'].nunique(),
    }
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("實驗：台積電 5m K 特徵對台指期盤中預測的影響")
    print("=" * 65)

    # 1. Fetch TSM features (determines the experiment date range)
    tsm_feat = fetch_tsm_features()
    if tsm_feat.empty:
        print("ERROR: 無法取得 TSM 資料，終止實驗")
        return

    min_date = tsm_feat['TradingDate'].min()
    print(f"\n資料重疊起始日：{min_date}")

    # 2. Build TX snapshots for overlapping period
    tx_df = build_tx_snapshots(min_date)
    if len(tx_df) < 200:
        print("ERROR: TX 快照資料不足，終止實驗")
        return

    # 3. Merge
    df = align_and_merge(tx_df, tsm_feat)

    # Check available TSM features
    available_tsm = [f for f in TSM_EXTRA_FEATURES if f in df.columns]
    print(f"可用 TSM 特徵：{available_tsm}")

    # 4. Run A/B comparison
    print(f"\n{'─'*65}")
    print(f"{'Model':<35} {'CV AUC':>8} {'±Std':>7} {'Morning':>8} {'Midday':>8}")
    print(f"{'─'*65}")

    results = []
    for feat_set, label in [
        (BASELINE_FEATURES, 'A: Baseline (TX only)'),
        (BASELINE_FEATURES + available_tsm, 'B: Enhanced (+TSM/ETF50)'),
    ]:
        r = run_cv(df, feat_set, label)
        results.append(r)
        print(f"{r['label']:<35} {r['cv_auc_mean']:>8.4f} {r['cv_auc_std']:>7.4f} "
              f"{r['morning_auc']:>8.4f} {r['midday_auc']:>8.4f}")

    print(f"{'─'*65}")

    # 5. Conclusion
    delta = results[1]['cv_auc_mean'] - results[0]['cv_auc_mean']
    delta_morning = results[1]['morning_auc'] - results[0]['morning_auc']
    print(f"\n△ AUC (B - A)        : {delta:+.4f}")
    print(f"△ Morning AUC (B - A): {delta_morning:+.4f}")
    print(f"\n資料範圍：{df['TradingDate'].min()} ~ {df['TradingDate'].max()}")
    print(f"訓練天數：{df['TradingDate'].nunique()} 天 / {len(df)} 筆快照")

    if abs(delta) >= 0.005:
        if delta > 0:
            print("\n✅ 結論：TSM 盤中特徵顯著提升準確率 → 建議取得更長歷史資料重新訓練")
        else:
            print("\n❌ 結論：TSM 盤中特徵未改善（甚至降低）準確率 → 不建議加入")
    else:
        print("\n⚠️  結論：AUC 差距 < 0.005，效果不顯著 → 需更多資料或不同特徵設計")

    # 6. Feature importance of B model (final fold)
    print("\nTop 10 特徵重要性（Model B, 最後一折）：")
    feat_b = BASELINE_FEATURES + available_tsm
    X_all = df[feat_b].fillna(0)
    y_all = df['Remaining_Dir']
    final_clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    final_clf.fit(X_all, y_all)
    imp = pd.Series(final_clf.feature_importances_, index=feat_b).sort_values(ascending=False)
    for f, v in imp.head(10).items():
        marker = ' ◀ TSM' if f in available_tsm else ''
        print(f"  {f:<30} {v:>6}{marker}")


if __name__ == '__main__':
    main()
