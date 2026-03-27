"""
Autoresearch Sandbox - Data Preparation (DO NOT MODIFY)

This file provides the fixed data pipeline for the autoresearch experiment loop.
It wraps the main project's data modules (src/data/, src/model/data_prep.py)
and provides clean, immutable datasets for train.py.

Usage:
    # One-time data preparation (fetches from TAIFEX & Yahoo Finance, caches locally)
    python prepare.py

    # Imported by train.py at runtime
    from prepare import get_data, TIME_BUDGET, FEATURE_COLS, TARGET_COLS
"""

import os
import sys
import time
import warnings

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup: allow importing from the main project
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300           # training time budget in seconds (5 minutes)
RANDOM_SEED = 42            # reproducibility seed

# Data date range
DATA_START = "2024-01-01"
DATA_END = "2026-03-31"

# Data split ratios (time-series order: train → val → test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (implicit)

# Feature columns available to train.py
FEATURE_COLS = [
    'TSM_Ret',          # TSMC ADR log return (T-1, shifted)
    'SOX_Ret',          # Philadelphia Semiconductor Index log return (T-1, shifted)
    'NetOI_Diff_lag1',  # FINI Net OI daily change (T-1, provided by DataPreprocessor)
    'Open_Gap',         # Open - Previous Close (points)
    'Intraday_Ret',     # Log(Close / Open) — intraday log return
    'Intraday_Point',   # Close - Open (points)
    'TX_Ret',           # Daily close-to-close log return
    'MA5',              # 5-day simple moving average
    'MA20',             # 20-day simple moving average
    'RSI',              # 14-day RSI
]

# Target columns
TARGET_COLS = {
    'gap_direction': 'Gap_Direction',       # Binary: 1 = gap up, 0 = gap down
    'gap_value': 'Open_Gap',                # Regression: gap size in points
    'intraday_return': 'Intraday_Ret',      # Regression: intraday log return
}

# Cache file for prepared data
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'prepared_data.csv')

# ---------------------------------------------------------------------------
# Data preparation (fetches and caches)
# ---------------------------------------------------------------------------

def _fetch_and_prepare():
    """
    Fetch data from all sources via the main project's DataPreprocessor,
    apply NetOI_Diff shift to prevent data leakage, and cache locally.
    """
    warnings.filterwarnings('ignore')

    from src.model.data_prep import DataPreprocessor

    print(f"Fetching data from {DATA_START} to {DATA_END}...")
    print("This may take a few minutes (TAIFEX has per-month rate limits).\n")

    prep = DataPreprocessor()
    df = prep.get_merged_data(DATA_START, DATA_END)

    if df.empty:
        print("ERROR: No data returned from DataPreprocessor!")
        sys.exit(1)

    # DataPreprocessor already handles NetOI_Diff shifting and naming it _lag1.
    # The shift(1) logic here was redundant and used an old column name.

    # Drop rows with NaN after shift
    df = df.dropna()

    print(f"\nPrepared dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")

    # Cache to csv
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(CACHE_FILE)
    print(f"\nCached to {CACHE_FILE}")

    return df


def _load_cached():
    """Load cached data from CSV file."""
    return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=False)


# ---------------------------------------------------------------------------
# Runtime API (imported by train.py)
# ---------------------------------------------------------------------------

def get_data():
    """
    Returns (train_df, val_df, test_df) — immutable copies.

    Each DataFrame contains all FEATURE_COLS + TARGET_COLS.
    Data is split in strict chronological order to prevent look-ahead bias.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if not os.path.exists(CACHE_FILE):
        print("Cache not found. Run `python prepare.py` first to fetch data.")
        print("Attempting to fetch now...")
        _fetch_and_prepare()

    df = _load_cached()

    # Strict time-series split
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    return train_df, val_df, test_df


def get_available_features(df):
    """
    Return list of feature columns that actually exist in the DataFrame.
    Useful for train.py to discover available features.
    """
    return [c for c in FEATURE_COLS if c in df.columns]


# ---------------------------------------------------------------------------
# Main: one-time data preparation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Autoresearch Sandbox — Data Preparation")
    print("=" * 60)
    print()

    t0 = time.time()
    df = _fetch_and_prepare()
    t1 = time.time()

    print(f"\nData preparation completed in {t1 - t0:.1f}s")
    print()

    # Show dataset summary
    print("Dataset Summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print()
    print(df.describe().to_string())
    print()

    # Verify split
    train, val, test = get_data()
    print(f"\nSplit verification:")
    print(f"  Train: {len(train)} rows ({len(train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val)} rows ({len(val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test)} rows ({len(test)/len(df)*100:.1f}%)")
    print()
    print("Done! Ready for experiments.")
