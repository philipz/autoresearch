# autoresearch_sandbox/fetch_sentiment.py
"""
Sentiment Cache Builder — 人類手動執行

功能：
  1. 讀取 prepared_data.csv 取得所有交易日
  2. 對近期日期抓取 RSS 新聞並呼叫 Gemini API 分析情緒
  3. 結果存入 .cache/sentiment_cache.csv

執行方式：
    cd autoresearch_sandbox
    export GEMINI_API_KEY="your-key-here"
    python fetch_sentiment.py

注意：
  - 歷史日期（>2天前）RSS 無法取得，自動填入中性值 score=0.0，不呼叫 API
  - 快取中已存在的日期不重複呼叫 API
  - 建議每日盤前（早上 5:30）執行一次更新最新情緒
  - sentiment_cache.csv 在 .gitignore 排除範圍內，換機器需重新執行此腳本
"""

import os
import sys
import time
from datetime import date, datetime

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data.sentiment import SentimentFetcher

CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
SENTIMENT_CACHE = os.path.join(CACHE_DIR, 'sentiment_cache.csv')
PREPARED_DATA = os.path.join(CACHE_DIR, 'prepared_data.csv')

SENTIMENT_COLS = ['date', 'sentiment_score', 'sentiment_conf', 'key_events', 'n_headlines']
LIVE_RSS_DAYS = 2


def load_existing_cache() -> pd.DataFrame:
    if os.path.exists(SENTIMENT_CACHE):
        return pd.read_csv(SENTIMENT_CACHE)
    return pd.DataFrame(columns=SENTIMENT_COLS)


def get_trading_dates() -> list[str]:
    if not os.path.exists(PREPARED_DATA):
        print(f"ERROR: {PREPARED_DATA} not found. Run `python prepare.py` first.")
        sys.exit(1)
    df = pd.read_csv(PREPARED_DATA, index_col=0)
    return list(df.index)


def process_date(fetcher: SentimentFetcher, date_str: str) -> dict:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return {'date': date_str, 'sentiment_score': 0.0, 'sentiment_conf': 0.0, 'key_events': '', 'n_headlines': 0}

    days_ago = (date.today() - d).days

    if days_ago > LIVE_RSS_DAYS:
        return {
            'date': date_str,
            'sentiment_score': 0.0,
            'sentiment_conf': 0.0,
            'key_events': '',
            'n_headlines': 0,
        }

    headlines = fetcher.fetch_rss_headlines(d)
    result = fetcher.analyze_sentiment(headlines, d)

    return {
        'date': date_str,
        'sentiment_score': result['score'],
        'sentiment_conf': result['confidence'],
        'key_events': ' | '.join(result['key_events']),
        'n_headlines': len(headlines),
    }


def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    print("=" * 60)
    print("Sentiment Cache Builder")
    print("=" * 60)

    fetcher = SentimentFetcher(api_key=api_key)
    existing = load_existing_cache()
    cached_dates = set(existing['date'].tolist()) if len(existing) else set()

    trading_dates = get_trading_dates()
    missing = [d for d in trading_dates if d not in cached_dates]

    print(f"Total trading days: {len(trading_dates)}")
    print(f"Already cached:     {len(cached_dates)}")
    print(f"To process:         {len(missing)}")

    if not missing:
        print("Cache is up to date. Nothing to do.")
        return

    rows = list(existing.to_dict('records'))
    live_count = 0

    for i, date_str in enumerate(missing):
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        days_ago = (date.today() - d).days
        is_live = days_ago <= LIVE_RSS_DAYS

        row = process_date(fetcher, date_str)
        rows.append(row)

        if is_live:
            live_count += 1
            print(f"  [{i+1}/{len(missing)}] {date_str} | score={row['sentiment_score']:.2f} | conf={row['sentiment_conf']:.2f} | headlines={row['n_headlines']}")
            if row['key_events']:
                print(f"    Events: {row['key_events']}")
            time.sleep(1.0)
        elif (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(missing)}] Backfilled up to {date_str} (historical placeholders)")

    result_df = pd.DataFrame(rows, columns=SENTIMENT_COLS)
    result_df = result_df.sort_values('date').drop_duplicates('date', keep='last')
    os.makedirs(CACHE_DIR, exist_ok=True)
    result_df.to_csv(SENTIMENT_CACHE, index=False)

    print(f"\nDone! Saved {len(result_df)} rows to {SENTIMENT_CACHE}")
    print(f"Live RSS days processed: {live_count}")
    if missing:
        print(f"Note: {len(missing) - live_count} historical days filled with neutral (0.0) placeholders.")


def test_live():
    """--test 模式：直接抓今日 RSS 並呼叫 Gemini API，驗證連線正常。不寫入快取。"""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    print("=" * 60)
    print("Sentiment Live Test (today only, no cache write)")
    print("=" * 60)

    fetcher = SentimentFetcher(api_key=api_key)
    today = date.today()

    print(f"\nFetching RSS headlines for {today}...")
    headlines = fetcher.fetch_rss_headlines(today)
    print(f"Got {len(headlines)} headlines")
    for h in headlines:
        print(f"  - {h}")

    if not headlines:
        print("\nWARNING: No headlines fetched (RSS may be unavailable).")
        print("Sending dummy headline to test Gemini API...")
        headlines = ["US stocks rise as Fed signals rate cut pause"]

    print(f"\nCalling Gemini API ({fetcher._model_name})...")
    result = fetcher.analyze_sentiment(headlines, today)

    print(f"\nResult:")
    print(f"  score:      {result['score']:+.3f}  (-1=bearish, +1=bullish for TX)")
    print(f"  confidence: {result['confidence']:.3f}")
    print(f"  key_events: {result['key_events']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_live()
    else:
        main()
