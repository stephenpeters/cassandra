#!/usr/bin/env python3
"""
Backfill Binance price data for existing collected markets.

This script adds Binance candle data to existing market JSON files.
Uses 1-second candles for maximum granularity (900 candles per 15-min window).
"""

import json
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

# Unbuffer stdout
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path("/Users/stephenpeters/Documents/Business/predmkt/backend/strategy_backtest/data/markets")
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Default to 1-second candles for maximum granularity
DEFAULT_INTERVAL = "1s"


def fetch_binance_candles(symbol: str, window_start: int, window_end: int, interval: str = "1s") -> list:
    """
    Fetch Binance klines (candles) for the market window.
    """
    try:
        start_ms = window_start * 1000
        end_ms = window_end * 1000

        resp = requests.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )

        if resp.status_code != 200:
            print(f"  API error: {resp.status_code}")
            return []

        raw_klines = resp.json()

        candles = []
        for k in raw_klines:
            candles.append({
                "timestamp": k[0] // 1000,
                "timestamp_human": datetime.fromtimestamp(k[0] // 1000).isoformat(),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "quote_volume": float(k[7]),
                "num_trades": int(k[8]),
            })

        return candles

    except Exception as e:
        print(f"  Fetch error: {e}")
        return []


def aggregate_to_15s(candles_1s: list) -> list:
    """Aggregate 1-second candles to 15-second candles."""
    if not candles_1s:
        return []

    aggregated = []
    bucket_size = 15

    for i in range(0, len(candles_1s), bucket_size):
        bucket = candles_1s[i:i + bucket_size]
        if not bucket:
            continue

        aggregated.append({
            "timestamp": bucket[0]["timestamp"],
            "timestamp_human": bucket[0]["timestamp_human"],
            "open": bucket[0]["open"],
            "high": max(c["high"] for c in bucket),
            "low": min(c["low"] for c in bucket),
            "close": bucket[-1]["close"],
            "volume": sum(c["volume"] for c in bucket),
            "quote_volume": sum(c["quote_volume"] for c in bucket),
            "num_trades": sum(c["num_trades"] for c in bucket),
        })

    return aggregated


def backfill_market(filepath: Path, force: bool = False) -> bool:
    """Add Binance data to a single market file."""
    try:
        with open(filepath) as f:
            data = json.load(f)

        # Skip if already has 1s Binance data (unless force)
        if not force and "binance" in data and data["binance"].get("interval") == "1s":
            print(f"  Skipping {filepath.name} (already has 1s Binance data)")
            return False

        window_start = data["window_start"]
        window_end = data["window_end"]

        print(f"  Fetching Binance 1s data for {filepath.name}...")
        candles_1s = fetch_binance_candles("BTCUSDT", window_start, window_end, interval="1s")

        if not candles_1s:
            print(f"  No candles returned for {filepath.name}")
            return False

        # Also create 15-second aggregated candles for smoother charts
        candles_15s = aggregate_to_15s(candles_1s)

        # Calculate price movement
        binance_open = candles_1s[0]["open"]
        binance_close = candles_1s[-1]["close"]
        binance_high = max(c["high"] for c in candles_1s)
        binance_low = min(c["low"] for c in candles_1s)
        price_change = binance_close - binance_open
        price_change_pct = (price_change / binance_open * 100) if binance_open else 0
        actual_resolution = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"

        # Add Binance data with both granularities
        data["binance"] = {
            "symbol": "BTCUSDT",
            "interval": "1s",
            "num_candles_1s": len(candles_1s),
            "num_candles_15s": len(candles_15s),
            "open": binance_open,
            "close": binance_close,
            "high": binance_high,
            "low": binance_low,
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "actual_resolution": actual_resolution,
            "candles_1s": candles_1s,  # Full resolution
            "candles_15s": candles_15s,  # Aggregated for plotting
        }

        # Add combined analysis
        bias = data.get("bias", "NEUTRAL")
        data["whale_predicted_correct"] = bias == actual_resolution if bias != "NEUTRAL" else None

        # Save updated file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Updated {filepath.name}: {len(candles_1s)} 1s candles, {len(candles_15s)} 15s candles | "
              f"Binance {price_change_pct:+.3f}% ({actual_resolution}), "
              f"Whale bias: {bias}, Correct: {data['whale_predicted_correct']}")
        return True

    except Exception as e:
        print(f"  Error processing {filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill Binance data for market files")
    parser.add_argument("--force", action="store_true", help="Force re-download even if data exists")
    args = parser.parse_args()

    print("=" * 60)
    print("BINANCE DATA BACKFILL (1-second candles)")
    print("=" * 60)
    if args.force:
        print("FORCE MODE: Re-downloading all data")

    # Find all market files
    market_files = sorted(DATA_DIR.glob("btc-updown-15m-*.json"))
    print(f"Found {len(market_files)} market files")

    updated = 0
    skipped = 0
    errors = 0

    for filepath in market_files:
        result = backfill_market(filepath, force=args.force)
        if result:
            updated += 1
            # Small delay to avoid rate limiting
            time.sleep(0.2)
        elif result is False:
            skipped += 1
        else:
            errors += 1

    print("\n" + "=" * 60)
    print(f"COMPLETE: Updated {updated}, Skipped {skipped}, Errors {errors}")
    print("=" * 60)

    # Summary analysis
    if updated > 0 or skipped > 0:
        print("\nWhale Prediction Analysis:")
        correct = 0
        incorrect = 0
        neutral = 0

        for filepath in market_files:
            with open(filepath) as f:
                data = json.load(f)
                result = data.get("whale_predicted_correct")
                if result is True:
                    correct += 1
                elif result is False:
                    incorrect += 1
                else:
                    neutral += 1

        total = correct + incorrect
        win_rate = (correct / total * 100) if total > 0 else 0
        print(f"  Correct predictions: {correct}/{total} ({win_rate:.1f}%)")
        print(f"  Neutral (no bias): {neutral}")


if __name__ == "__main__":
    main()
