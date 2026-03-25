"""
Autoresearch Sandbox - Intraday Snapshot Data Preparation (DO NOT MODIFY)

將「5分K」與「日級別靜態特徵」合併為盤中快照 (Intraday Snapshot) 資料集，
供 train_intraday.py 使用。

每一筆 Row = 某交易日的某個 5 分鐘時間切片，包含：
  - 靜態特徵: TSM_Ret, SOX_Ret, NetOI_Diff, Open_Gap 等 (當日不變)
  - 動態特徵: 即時漲跌幅, VWAP 乖離, 量能變化, 盤中區間等
  - Target:   該切片到收盤的「剩餘漲跌幅」

Usage:
    # 一次性資料準備 (建立 cache)
    python prepare_intraday.py

    # 由 train_intraday.py import
    from prepare_intraday import get_intraday_data, TIME_BUDGET, RANDOM_SEED
"""

import os
import sys
import time
import warnings

import pandas as pd
import numpy as np

# ───────────────────────────────────────────────────────
# Path setup
# ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# ───────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────

TIME_BUDGET = 300           # training time budget (5 min)
RANDOM_SEED = 42

# Paths
KBAR_5M_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'tx_5m_kbar.csv')
DAILY_CACHE_FILE = os.path.join(os.path.dirname(__file__), '.cache', 'prepared_data.csv')
SENTIMENT_CACHE_FILE = os.path.join(os.path.dirname(__file__), '.cache', 'sentiment_cache.csv')
INTRADAY_CACHE_FILE = os.path.join(os.path.dirname(__file__), '.cache', 'intraday_snapshots.csv')

# Data split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# ───────────────────────────────────────────────────────
# Build intraday snapshots
# ───────────────────────────────────────────────────────

def _load_daily_features():
    """Load existing daily features from autoresearch cache."""
    if not os.path.exists(DAILY_CACHE_FILE):
        print("ERROR: Daily cache not found. Run `python prepare.py` first.")
        sys.exit(1)
    df = pd.read_csv(DAILY_CACHE_FILE, index_col=0)
    # Merge sentiment if available
    if os.path.exists(SENTIMENT_CACHE_FILE):
        sent = pd.read_csv(SENTIMENT_CACHE_FILE)
        sent = sent[['date', 'sentiment_score', 'sentiment_conf']].copy()
        sent.columns = ['date', 'Sentiment_Score', 'Sentiment_Conf']
        sent = sent.set_index('date')
        df['Sentiment_Score'] = sent['Sentiment_Score'].reindex(df.index).shift(1).fillna(0.0).values
        df['Sentiment_Conf'] = sent['Sentiment_Conf'].reindex(df.index).shift(1).fillna(0.0).values
    else:
        df['Sentiment_Score'] = 0.0
        df['Sentiment_Conf'] = 0.0
    return df


def _load_5m_kbar():
    """Load 5-minute K-bar data."""
    if not os.path.exists(KBAR_5M_FILE):
        print(f"ERROR: 5-min K-bar file not found at {KBAR_5M_FILE}.")
        print("Run: python scripts/rpt_to_5m_kbar.py --input data/TX_RPT --output data/raw/tx_5m_kbar.csv --session day")
        sys.exit(1)
    df = pd.read_csv(KBAR_5M_FILE)
    # Standardize TradingDate to YYYY-MM-DD format for joining
    df['TradingDate'] = pd.to_datetime(df['TradingDate'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
    return df


def _compute_intraday_features(kbar_day):
    """
    For a single day's 5-min K-bars, compute dynamic features at each snapshot.
    
    Args:
        kbar_day: DataFrame of 5-min bars for one day, sorted by Time.
        
    Returns:
        DataFrame with one row per snapshot, indexed by Time.
    """
    if kbar_day.empty:
        return pd.DataFrame()
    
    kbar = kbar_day.sort_values('Time').reset_index(drop=True)
    day_open = kbar['Open'].iloc[0]
    day_close = kbar['Close'].iloc[-1]
    
    snapshots = []
    
    cum_volume = 0
    cum_vwap_numerator = 0.0  # sum(typical_price * volume)
    running_high = -np.inf
    running_low = np.inf
    
    for i, row in kbar.iterrows():
        bar_typical_price = (row['High'] + row['Low'] + row['Close']) / 3.0
        cum_volume += row['Volume']
        cum_vwap_numerator += bar_typical_price * row['Volume']
        running_high = max(running_high, row['High'])
        running_low = min(running_low, row['Low'])
        
        current_price = row['Close']
        
        # ── Dynamic Features ──
        # 1. Intraday return since open (%)
        intraday_ret = (current_price - day_open) / day_open
        
        # 2. VWAP and distance
        vwap = cum_vwap_numerator / cum_volume if cum_volume > 0 else current_price
        vwap_dist = (current_price - vwap) / vwap  # positive = above VWAP
        
        # 3. Volume momentum (current bar volume vs average so far)
        avg_vol = cum_volume / (i + 1) if (i + 1) > 0 else row['Volume']
        vol_ratio = row['Volume'] / avg_vol if avg_vol > 0 else 1.0
        
        # 4. Running range (high-low spread normalized by open)
        day_range = (running_high - running_low) / day_open
        
        # 5. Position in day's range (0=at low, 1=at high)
        range_span = running_high - running_low
        range_position = (current_price - running_low) / range_span if range_span > 0 else 0.5
        
        # 6. Time of day (normalized: 0=08:45, 1=13:45)
        # There are typically ~60 bars, so bar_index / total gives progress
        time_progress = i / max(len(kbar) - 1, 1)
        
        # 7. Bar body and shadow ratios
        bar_body = (row['Close'] - row['Open']) / day_open  # positive=bullish bar
        bar_range = (row['High'] - row['Low']) / day_open
        
        # 8. Short-term momentum (last 3 bars return)
        if i >= 2:
            mom3_price = kbar['Close'].iloc[i-2]
            mom3 = (current_price - mom3_price) / mom3_price
        else:
            mom3 = 0.0
            
        # 9. Short-term momentum (last 6 bars = 30 min)
        if i >= 5:
            mom6_price = kbar['Close'].iloc[i-5]
            mom6 = (current_price - mom6_price) / mom6_price
        else:
            mom6 = 0.0
        
        # ── Target ──
        # Remaining return from this snapshot to day close
        remaining_ret = (day_close - current_price) / current_price
        # Direction: will it go up from here?
        remaining_dir = 1 if remaining_ret > 0 else 0
        # Remaining points
        remaining_points = day_close - current_price
        
        snapshots.append({
            'Time': row['Time'],
            # Dynamic features
            'Intraday_Ret_Now': intraday_ret,
            'VWAP_Dist': vwap_dist,
            'Vol_Ratio': vol_ratio,
            'Day_Range': day_range,
            'Range_Position': range_position,
            'Time_Progress': time_progress,
            'Bar_Body': bar_body,
            'Bar_Range': bar_range,
            'Mom3': mom3,
            'Mom6': mom6,
            'Cum_Volume': cum_volume,
            'Current_Price': current_price,
            # Targets
            'Remaining_Ret': remaining_ret,
            'Remaining_Dir': remaining_dir,
            'Remaining_Points': remaining_points,
        })
    
    return pd.DataFrame(snapshots)


def build_intraday_dataset():
    """
    Main pipeline: merge daily static features with 5-min dynamic features
    to create the full intraday snapshot dataset.
    """
    print("Loading daily static features...")
    daily = _load_daily_features()
    
    print(f"Loading 5-min K-bar data from {KBAR_5M_FILE}...")
    kbar = _load_5m_kbar()
    
    # Find overlapping dates
    daily_dates = set(daily.index)
    kbar_dates = set(kbar['TradingDate'].unique())
    common_dates = sorted(daily_dates & kbar_dates)
    print(f"Overlapping dates: {len(common_dates)} (daily={len(daily_dates)}, kbar={len(kbar_dates)})")
    
    if not common_dates:
        print("ERROR: No overlapping dates between daily and 5-min data!")
        sys.exit(1)
    
    all_snapshots = []
    
    for dt in common_dates:
        day_kbar = kbar[kbar['TradingDate'] == dt]
        day_static = daily.loc[dt]
        
        # Compute dynamic features for this day
        snap_df = _compute_intraday_features(day_kbar)
        if snap_df.empty:
            continue
        
        # Attach static features (broadcast across all snapshots of this day)
        for col in daily.columns:
            snap_df[col] = day_static[col]
        
        snap_df['TradingDate'] = dt
        all_snapshots.append(snap_df)
    
    result = pd.concat(all_snapshots, ignore_index=True)
    
    # Reorder columns: TradingDate, Time, dynamic features, static features, targets
    dynamic_cols = [
        'Intraday_Ret_Now', 'VWAP_Dist', 'Vol_Ratio', 'Day_Range',
        'Range_Position', 'Time_Progress', 'Bar_Body', 'Bar_Range',
        'Mom3', 'Mom6', 'Cum_Volume', 'Current_Price',
    ]
    static_cols = [c for c in daily.columns if c not in ['Gap_Direction']]
    target_cols = ['Remaining_Ret', 'Remaining_Dir', 'Remaining_Points']
    
    ordered = ['TradingDate', 'Time'] + dynamic_cols + static_cols + target_cols
    # Keep only existing columns
    ordered = [c for c in ordered if c in result.columns]
    result = result[ordered]
    
    return result


# ───────────────────────────────────────────────────────
# Runtime API (imported by train_intraday.py)
# ───────────────────────────────────────────────────────

def get_intraday_data():
    """
    Returns (train_df, val_df, test_df) of intraday snapshots.
    
    Split is done by DATE (not by row) to prevent look-ahead bias.
    All snapshots from the same date stay together in the same split.
    """
    if os.path.exists(INTRADAY_CACHE_FILE):
        print(f"Loading cached intraday snapshots from {INTRADAY_CACHE_FILE}...")
        df = pd.read_csv(INTRADAY_CACHE_FILE)
    else:
        print("Cache not found. Building intraday snapshots (this may take a minute)...")
        df = build_intraday_dataset()
        os.makedirs(os.path.dirname(INTRADAY_CACHE_FILE), exist_ok=True)
        df.to_csv(INTRADAY_CACHE_FILE, index=False)
        print(f"Cached to {INTRADAY_CACHE_FILE}")
    
    # Split by unique dates (chronological)
    dates = sorted(df['TradingDate'].unique())
    n_dates = len(dates)
    train_end = int(n_dates * TRAIN_RATIO)
    val_end = int(n_dates * (TRAIN_RATIO + VAL_RATIO))
    
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])
    
    train_df = df[df['TradingDate'].isin(train_dates)].copy()
    val_df = df[df['TradingDate'].isin(val_dates)].copy()
    test_df = df[df['TradingDate'].isin(test_dates)].copy()
    
    print(f"Intraday data loaded: train={len(train_df)} ({len(train_dates)} days), "
          f"val={len(val_df)} ({len(val_dates)} days), "
          f"test={len(test_df)} ({len(test_dates)} days)")
    
    return train_df, val_df, test_df


# ───────────────────────────────────────────────────────
# Main: one-time data preparation
# ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Autoresearch Sandbox — Intraday Snapshot Data Preparation")
    print("=" * 60)
    print()
    
    t0 = time.time()
    df = build_intraday_dataset()
    t1 = time.time()
    
    # Save cache
    cache_dir = os.path.dirname(INTRADAY_CACHE_FILE)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(INTRADAY_CACHE_FILE, index=False)
    
    print(f"\nData preparation completed in {t1 - t0:.1f}s")
    print()
    print("Dataset Summary:")
    print(f"  Total snapshots: {len(df)}")
    print(f"  Unique dates:    {df['TradingDate'].nunique()}")
    print(f"  Columns:         {len(df.columns)}")
    print(f"  Date range:      {df['TradingDate'].iloc[0]} ~ {df['TradingDate'].iloc[-1]}")
    print()
    
    # Show feature stats
    print("Dynamic feature stats:")
    dynamic = ['Intraday_Ret_Now', 'VWAP_Dist', 'Vol_Ratio', 'Day_Range', 'Range_Position', 'Time_Progress']
    print(df[dynamic].describe().to_string())
    print()
    
    # Target stats
    print("Target stats (Remaining_Ret):")
    print(df['Remaining_Ret'].describe().to_string())
    print()
    print(f"Remaining_Dir distribution: {df['Remaining_Dir'].value_counts().to_dict()}")
    print()
    
    # Verify split
    train, val, test = get_intraday_data()
    print(f"\nSplit verification:")
    print(f"  Train: {len(train)} rows")
    print(f"  Val:   {len(val)} rows")
    print(f"  Test:  {len(test)} rows")
    print()
    print("Done! Ready for intraday experiments.")
