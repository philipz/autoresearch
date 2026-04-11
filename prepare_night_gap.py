"""
Autoresearch Sandbox — Night Session Gap Prediction: Data Preparation (DO NOT MODIFY)

每一筆 Row = 一個交易日 T 的夜盤聚合資料。
Features: 夜盤每日 OHLCV（T-1 15:00 → T 05:00）+ prepared_data.csv 的靜態特徵。
Targets:
  - Gap_Direction: binary，1 = Day T 開盤高於 Day T-1 收盤
  - Open_Gap:      regression，Day T 開盤 - Day T-1 收盤（點數）

對齊說明：
  Night_T（TradingDate=T）= T-1 15:00 至 T 05:00 的夜盤。
  Night_Open（15:00 bar）≈ T-1 日盤收盤（proxy，誤差 ±0.1~0.2%，研究用途可接受）。

Usage:
    python prepare_night_gap.py   # 一次性建立快取
    from prepare_night_gap import get_night_gap_data, TIME_BUDGET, RANDOM_SEED
"""

import os, sys, time, warnings
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------
TIME_BUDGET  = 300   # training time budget (seconds)
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO  = 0.15 (implicit)

NIGHT_KBAR_FILE  = os.path.join(PROJECT_ROOT, 'data', 'raw', 'tx_night_5m_kbar.csv')
DAILY_CACHE_FILE = os.path.join(os.path.dirname(__file__), '.cache', 'prepared_data.csv')
NIGHT_GAP_CACHE  = os.path.join(os.path.dirname(__file__), '.cache', 'night_gap_prepared.csv')


def _night_sort_key(t: str) -> int:
    """夜盤時間字串 (HH:MM:SS) → 排序用整數，跨午夜 00:00~14:59 加 24h。"""
    h, m = int(t[:2]), int(t[3:5])
    mins = h * 60 + m
    if mins < 15 * 60:   # 跨午夜段
        mins += 24 * 60
    return mins


def _aggregate_night_ohlcv() -> pd.DataFrame:
    """將 5 分 K 夜盤資料聚合為每日 OHLCV（以 TradingDate 為鍵）。"""
    print(f"Loading night 5m K-bar from {NIGHT_KBAR_FILE}...")
    kbar = pd.read_csv(NIGHT_KBAR_FILE)
    kbar['TradingDate'] = (
        pd.to_datetime(kbar['TradingDate'].astype(str), format='%Y%m%d')
        .dt.strftime('%Y-%m-%d')
    )

    # 跨午夜排序（15:00 → 23:59 → 00:00 → 05:00）
    kbar['_sort'] = kbar['Time'].apply(_night_sort_key)
    kbar = kbar.sort_values(['TradingDate', '_sort'])

    grp = kbar.groupby('TradingDate')
    daily = pd.DataFrame({
        'Night_Open':  grp['Open'].first(),
        'Night_High':  grp['High'].max(),
        'Night_Low':   grp['Low'].min(),
        'Night_Close': grp['Close'].last(),
        'Night_Vol':   grp['Volume'].sum(),
        'Night_Bars':  grp['Close'].count(),
    })
    print(f"  Aggregated: {len(daily)} trading days")
    return daily


def _build_features(night_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """計算夜盤特徵並合併靜態日線特徵。"""
    df = night_df.copy()

    safe_open = df['Night_Open'].replace(0, np.nan)

    # 核心夜盤特徵
    # Night_Ret_vs_Day: 夜盤收盤 vs 夜盤開盤（代理 T-1 收盤）
    df['Night_Ret_vs_Day']  = np.log(df['Night_Close'] / safe_open)
    # Night_Body: 夜盤內部 K 線實體（收 vs 開）
    df['Night_Body']        = np.log(df['Night_Close'] / safe_open)
    # Night_Range: 夜盤高低波幅 / 開盤
    df['Night_Range']       = (df['Night_High'] - df['Night_Low']) / safe_open
    # Night_Vol_Change: 夜盤量能變化（log）
    prev_vol = df['Night_Vol'].shift(1).replace(0, np.nan)
    df['Night_Vol_Change']  = np.log(df['Night_Vol'] / prev_vol)
    df['Night_Vol_Change']  = df['Night_Vol_Change'].replace([np.inf, -np.inf], np.nan)
    # Night_Gap_Open: 夜盤開盤 vs 前一日夜盤收盤（夜盤跳空）
    prev_close = df['Night_Close'].shift(1).replace(0, np.nan)
    df['Night_Gap_Open']    = np.log(df['Night_Open'] / prev_close)

    # 靜態特徵（來自 prepared_data.csv，已對齊 TradingDate）
    static_cols = [
        'TSM_Ret', 'SOX_Ret', 'NetOI_Diff_lag1', 'NetOI_Diff_lag2',
        'TX_Ret', 'TX_Ret_lag1', 'TX_Vol5', 'TX_Mom5',
        'RSI', 'MA5', 'MA20', 'TSM_Ret_lag2', 'SOX_Ret_lag2', 'TSM_SOX_Spread',
        'Intra_Ret_lag1',
    ]
    available = [c for c in static_cols if c in daily_df.columns]
    df = df.join(daily_df[available], how='inner')

    # Targets
    df['Gap_Direction'] = daily_df['Gap_Direction']
    df['Open_Gap']      = daily_df['Open_Gap']

    df = df.dropna(subset=['Night_Ret_vs_Day', 'Gap_Direction', 'Open_Gap'])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def build_night_gap_dataset() -> pd.DataFrame:
    """主 pipeline：聚合夜盤 OHLCV + 合併靜態特徵。"""
    if not os.path.exists(NIGHT_KBAR_FILE):
        print(f"ERROR: Night 5m kbar not found at {NIGHT_KBAR_FILE}")
        sys.exit(1)
    if not os.path.exists(DAILY_CACHE_FILE):
        print("ERROR: Daily cache not found. Run `python prepare.py` first.")
        sys.exit(1)

    night_ohlcv = _aggregate_night_ohlcv()

    print("Loading daily static features...")
    daily = pd.read_csv(DAILY_CACHE_FILE, index_col=0)
    print(f"  Daily rows: {len(daily)}")

    print("Merging and computing features...")
    df = _build_features(night_ohlcv, daily)
    print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Gap_Direction balance: {df['Gap_Direction'].mean():.2%} UP")
    return df


def get_night_gap_data():
    """
    返回 (train_df, val_df, test_df)，按日期嚴格切分（不洩漏）。

    Split: 70% train / 15% val / 15% test
    """
    if os.path.exists(NIGHT_GAP_CACHE):
        print(f"Loading cached night gap data from {NIGHT_GAP_CACHE}...")
        df = pd.read_csv(NIGHT_GAP_CACHE, index_col=0)
    else:
        print("Cache not found. Building night gap dataset...")
        df = build_night_gap_dataset()
        os.makedirs(os.path.dirname(NIGHT_GAP_CACHE), exist_ok=True)
        df.to_csv(NIGHT_GAP_CACHE)
        print(f"Cached to {NIGHT_GAP_CACHE}")

    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    print(f"Night gap data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    print("=" * 60)
    print("Autoresearch Sandbox — Night Gap Data Preparation")
    print("=" * 60)

    t0 = time.time()
    df = build_night_gap_dataset()
    os.makedirs(os.path.dirname(NIGHT_GAP_CACHE), exist_ok=True)
    df.to_csv(NIGHT_GAP_CACHE)
    print(f"\nSaved to {NIGHT_GAP_CACHE}  ({time.time()-t0:.1f}s)")
    print("\nDataset summary:")
    print(df[['Night_Ret_vs_Day', 'Night_Range', 'Night_Vol_Change',
              'TSM_Ret', 'SOX_Ret', 'Gap_Direction', 'Open_Gap']].describe().to_string())
    print("\nSplit verification:")
    train, val, test = get_night_gap_data()
    print(f"  Train: {len(train)} rows ({len(train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val)} rows ({len(val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test)} rows ({len(test)/len(df)*100:.1f}%)")
