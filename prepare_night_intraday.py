"""
Night Intraday Snapshot Data Preparation

每一筆 Row = 夜盤某 5 分鐘時間切片，包含：
  - 動態特徵: 即時漲跌幅、VWAP 乖離、量能、區間位置等
  - 靜態特徵: T-1 日的 TSM_Ret, SOX_Ret, NetOI（在夜盤開始前即可得到）
  - Target:   該切片到夜盤收盤的「剩餘漲跌方向」

時間定義：TradingDate=T 的夜盤為 T-1 15:00 → T 05:00
Time_Progress: 0.0=15:00, 1.0=05:00，總長 840 分鐘（含跨午夜）

Usage:
    python prepare_night_intraday.py
"""

import os, sys, warnings
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings('ignore')

TIME_BUDGET  = 300
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15

NIGHT_SESSION_START_MINUTES = 15 * 60   # 900
NIGHT_SESSION_TOTAL_MINUTES = 840       # 15:00 → 05:00

NIGHT_KBAR_FILE      = os.path.join(PROJECT_ROOT, 'data', 'raw', 'tx_night_5m_kbar.csv')
DAILY_CACHE_FILE     = os.path.join(os.path.dirname(__file__), '.cache', 'prepared_data.csv')
NIGHT_INTRADAY_CACHE = os.path.join(os.path.dirname(__file__), '.cache', 'night_intraday_snapshots.csv')


def _night_time_to_elapsed_minutes(time_str: str) -> int:
    """
    夜盤時間字串 (HH:MM:SS) → 相對於 15:00 的已過分鐘數。
    處理跨午夜：00:00~05:59 視為 15:00 後的延續。

    15:00 → 0, 16:00 → 60, 00:00 → 540, 05:00 → 840
    """
    h, m = int(time_str[:2]), int(time_str[3:5])
    total = h * 60 + m
    if total < NIGHT_SESSION_START_MINUTES:   # 跨午夜 (00:00~05:59)
        total += 24 * 60
    return total - NIGHT_SESSION_START_MINUTES


def _compute_night_features(kbar_day: pd.DataFrame) -> pd.DataFrame:
    """
    單日夜盤 5 分 K → 動態特徵快照 DataFrame。
    排除最後一根 bar（Remaining_Ret 永遠為 0，無預測意義）。
    """
    if len(kbar_day) == 0:
        return pd.DataFrame()
    if len(kbar_day) < 2:
        return pd.DataFrame()

    kbar = kbar_day.sort_values('Time').reset_index(drop=True)
    night_open  = kbar['Open'].iloc[0]
    night_close = kbar['Close'].iloc[-1]

    cum_volume    = kbar['Volume'].cumsum()
    typical_price = (kbar['High'] + kbar['Low'] + kbar['Close']) / 3.0
    cum_vwap_num  = (typical_price * kbar['Volume']).cumsum()
    running_high  = kbar['High'].cummax()
    running_low   = kbar['Low'].cummin()
    current_price = kbar['Close']

    intraday_ret = (current_price - night_open) / night_open

    vwap = np.where(cum_volume > 0, cum_vwap_num / cum_volume, current_price)
    vwap_safe = np.where(vwap > 0, vwap, 1.0)
    vwap_dist = (current_price - vwap) / vwap_safe

    prior_cum_vol = (cum_volume - kbar['Volume']).values
    idx = np.arange(len(kbar))
    avg_prior_vol = np.where(idx > 0, prior_cum_vol / np.maximum(idx, 1), kbar['Volume'].values)
    vol_ratio = np.where(avg_prior_vol > 0, kbar['Volume'].values / avg_prior_vol, 1.0)

    day_range   = (running_high - running_low) / night_open
    range_span  = running_high - running_low
    range_position = np.where(range_span > 0, (current_price - running_low) / range_span, 0.5)

    elapsed = kbar['Time'].apply(_night_time_to_elapsed_minutes)
    time_progress = (elapsed / NIGHT_SESSION_TOTAL_MINUTES).clip(0, 1)

    bar_body  = (kbar['Close'] - kbar['Open']) / night_open
    bar_range = (kbar['High']  - kbar['Low'])  / night_open

    mom3 = ((current_price - kbar['Close'].shift(2)) / kbar['Close'].shift(2)).fillna(0.0)
    mom6 = ((current_price - kbar['Close'].shift(5)) / kbar['Close'].shift(5)).fillna(0.0)

    remaining_ret    = (night_close - current_price) / current_price
    remaining_dir    = (remaining_ret > 0).astype(int)
    remaining_points = night_close - current_price

    snapshots = pd.DataFrame({
        'Time':             kbar['Time'],
        'Intraday_Ret_Now': intraday_ret,
        'VWAP_Dist':        vwap_dist,
        'Vol_Ratio':        vol_ratio,
        'Day_Range':        day_range,
        'Range_Position':   range_position,
        'Time_Progress':    time_progress.values,
        'Bar_Body':         bar_body,
        'Bar_Range':        bar_range,
        'Mom3':             mom3,
        'Mom6':             mom6,
        'Cum_Volume':       cum_volume,
        'Current_Price':    current_price,
        'Remaining_Ret':    remaining_ret,
        'Remaining_Dir':    remaining_dir,
        'Remaining_Points': remaining_points,
    })
    return snapshots.iloc[:-1]   # 排除最後一根 bar


def _load_daily_features() -> pd.DataFrame:
    """從 autoresearch cache 載入日級靜態特徵，並為夜盤做 shift。"""
    if not os.path.exists(DAILY_CACHE_FILE):
        print("ERROR: Daily cache not found. Run `python prepare.py` first.")
        sys.exit(1)
    df = pd.read_csv(DAILY_CACHE_FILE, index_col=0)

    # 夜盤 TradingDate=T 實際是 T-1 夜間，靜態特徵需再 shift(1)
    night_static_cols = ['TSM_Ret', 'SOX_Ret', 'NetOI_Diff_lag1', 'TX_Ret',
                         'TSM_Ret_lag2', 'SOX_Ret_lag2', 'NetOI_Diff_lag2',
                         'TX_Vol5', 'TX_Mom5']
    for col in night_static_cols:
        if col in df.columns:
            df[f'Night_{col}'] = df[col].shift(1)

    night_nan_cols = [c for c in df.columns if c.startswith('Night_') and df[c].isna().any()]
    if night_nan_cols:
        print(f"WARNING: {len(night_nan_cols)} Night_ columns have NaN (filled with 0): {night_nan_cols[:3]}...")
    df.ffill(inplace=True)
    df.fillna(0.0, inplace=True)
    return df


def build_night_intraday_dataset() -> pd.DataFrame:
    """主 pipeline：合併夜盤 5 分 K 與靜態特徵，建立快照資料集。"""
    if not os.path.exists(NIGHT_KBAR_FILE):
        print(f"ERROR: Night 5m kbar not found at {NIGHT_KBAR_FILE}")
        sys.exit(1)

    print(f"Loading night 5m K-bar from {NIGHT_KBAR_FILE}...")
    kbar = pd.read_csv(NIGHT_KBAR_FILE)
    kbar['TradingDate'] = pd.to_datetime(
        kbar['TradingDate'].astype(str), format='%Y%m%d'
    ).dt.strftime('%Y-%m-%d')

    print("Loading daily static features...")
    daily = _load_daily_features()

    common_dates = sorted(set(kbar['TradingDate'].unique()) & set(daily.index))
    print(f"Overlapping dates: {len(common_dates)}")

    if not common_dates:
        print("ERROR: No overlapping dates!")
        sys.exit(1)

    valid_kbar = kbar[kbar['TradingDate'].isin(common_dates)]
    all_snaps = (
        valid_kbar.groupby('TradingDate', group_keys=False)
        .apply(_compute_night_features)
        .reset_index(drop=True)
    )

    night_static_cols = [c for c in daily.columns if c.startswith('Night_')]
    result = pd.merge(
        all_snaps,
        daily[night_static_cols],
        left_on='TradingDate',
        right_index=True,
        how='left'
    )
    result.fillna(0.0, inplace=True)
    return result


def get_night_intraday_data():
    """返回 (train_df, val_df, test_df)，按日期切分（不洩漏）。"""
    if os.path.exists(NIGHT_INTRADAY_CACHE):
        print(f"Loading cached night intraday snapshots from {NIGHT_INTRADAY_CACHE}...")
        df = pd.read_csv(NIGHT_INTRADAY_CACHE)
    else:
        print("Cache not found. Building night intraday snapshots...")
        df = build_night_intraday_dataset()
        os.makedirs(os.path.dirname(NIGHT_INTRADAY_CACHE), exist_ok=True)
        df.to_csv(NIGHT_INTRADAY_CACHE, index=False)
        print(f"Cached to {NIGHT_INTRADAY_CACHE}")

    dates     = sorted(df['TradingDate'].unique())
    n         = len(dates)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df[df['TradingDate'].isin(set(dates[:train_end]))].copy()
    val_df   = df[df['TradingDate'].isin(set(dates[train_end:val_end]))].copy()
    test_df  = df[df['TradingDate'].isin(set(dates[val_end:]))].copy()

    print(f"Night data loaded: train={len(train_df)} ({train_end} days), "
          f"val={len(val_df)} ({val_end - train_end} days), "
          f"test={len(test_df)} ({n - val_end} days)")
    return train_df, val_df, test_df


if __name__ == "__main__":
    df = build_night_intraday_dataset()
    os.makedirs(os.path.dirname(NIGHT_INTRADAY_CACHE), exist_ok=True)
    df.to_csv(NIGHT_INTRADAY_CACHE, index=False)
    print(f"\nNight intraday dataset: {len(df)} rows, {df['TradingDate'].nunique()} dates")
    print(df[['Time_Progress', 'Remaining_Dir', 'Remaining_Points']].describe().to_string())
