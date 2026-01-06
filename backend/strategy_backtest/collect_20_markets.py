#!/usr/bin/env python3
"""
Collect 20 BTC 15-min markets from Polymarket + Binance price data.
Each market saved to separate JSON file with:
- Polymarket whale trades
- Binance 1-minute candles for the window

Runtime: ~5 hours (20 x 15 min)
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from pathlib import Path

# Unbuffer stdout
sys.stdout.reconfigure(line_buffering=True)

GABAGOOL_WALLET = "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d"
DATA_DIR = Path("/Users/stephenpeters/Documents/Business/predmkt/backend/strategy_backtest/data/markets")
LOG_FILE = DATA_DIR.parent / "collection_log.txt"
MARKETS_TO_COLLECT = 20

# Binance API endpoint for klines (candles)
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    """Log to stdout and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def fetch_binance_candles(symbol: str, window_start: int, window_end: int, interval: str = "1m") -> list:
    """
    Fetch Binance klines (candles) for the market window.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        window_start: Unix timestamp of window start
        window_end: Unix timestamp of window end
        interval: Candle interval (default "1m" = 1 minute)

    Returns:
        List of candles with OHLCV data
    """
    try:
        # Convert to milliseconds for Binance API
        start_ms = window_start * 1000
        end_ms = window_end * 1000

        resp = requests.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,  # More than enough for 15 min window
            },
            timeout=30,
        )

        if resp.status_code != 200:
            log(f"  Binance API error: {resp.status_code}")
            return []

        raw_klines = resp.json()

        # Parse klines into readable format
        # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
        candles = []
        for k in raw_klines:
            candles.append({
                "timestamp": k[0] // 1000,  # Convert back to seconds
                "timestamp_human": datetime.fromtimestamp(k[0] // 1000).isoformat(),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "quote_volume": float(k[7]),  # Volume in USDT
                "num_trades": int(k[8]),
            })

        log(f"  Binance: Fetched {len(candles)} {interval} candles")
        return candles

    except Exception as e:
        log(f"  Binance fetch error: {e}")
        return []


def collect_market(slug, window_start, window_end):
    """Collect all trades for a single market"""
    seen_ids = set()
    all_trades = []
    last_status = time.time()

    log(f"Starting collection: {slug}")
    log(f"Window: {datetime.fromtimestamp(window_start)} to {datetime.fromtimestamp(window_end)}")

    # Wait for market to start
    now = time.time()
    if now < window_start:
        wait = window_start - now
        log(f"Waiting {wait:.0f}s for market start...")
        time.sleep(wait)

    # Collect until market ends
    while time.time() < window_end + 60:
        # Fetch all trades
        offset = 0
        while offset < 100000:
            try:
                resp = requests.get(
                    "https://data-api.polymarket.com/trades",
                    params={"maker": GABAGOOL_WALLET, "limit": 500, "offset": offset},
                    timeout=30,
                )
                if resp.status_code != 200:
                    break
                trades = resp.json()
                if not trades:
                    break

                for t in trades:
                    if t.get("eventSlug", "") == slug:
                        trade_id = f"{t.get('transactionHash', '')}_{t.get('timestamp', '')}"
                        if trade_id not in seen_ids:
                            seen_ids.add(trade_id)
                            all_trades.append(t)

                timestamps = [t.get("timestamp", 0) for t in trades if isinstance(t.get("timestamp"), int)]
                if timestamps and min(timestamps) < window_start - 300:
                    break
                offset += len(trades)
            except Exception as e:
                log(f"  Error: {e}")
                break

        # Status update every 30s
        if time.time() - last_status >= 30:
            remaining = max(0, window_end - time.time())
            net_up = net_down = 0
            for t in all_trades:
                size = float(t.get("size", 0))
                outcome = t.get("outcome", "")
                side = t.get("side", "")
                if outcome == "Up":
                    net_up += size if side == "BUY" else -size
                elif outcome == "Down":
                    net_down += size if side == "BUY" else -size

            bias = "UP" if net_up > net_down * 1.2 else "DOWN" if net_down > net_up * 1.2 else "NEUTRAL"
            log(f"  Trades: {len(all_trades):,} | Bias: {bias} | Remaining: {remaining:.0f}s")
            last_status = time.time()

        if time.time() >= window_end + 60:
            break
        time.sleep(10)

    # Final save
    net_up = net_down = total_vol = 0
    for t in all_trades:
        size = float(t.get("size", 0))
        price = float(t.get("price", 0))
        outcome = t.get("outcome", "")
        side = t.get("side", "")
        total_vol += size * price
        if outcome == "Up":
            net_up += size if side == "BUY" else -size
        elif outcome == "Down":
            net_down += size if side == "BUY" else -size

    bias = "UP" if net_up > net_down * 1.2 else "DOWN" if net_down > net_up * 1.2 else "NEUTRAL"
    conf = abs(net_up - net_down) / (net_up + net_down) * 100 if (net_up + net_down) > 0 else 0

    # Fetch Binance 1-minute candles for this window
    log(f"  Fetching Binance BTC candles...")
    binance_candles = fetch_binance_candles("BTCUSDT", window_start, window_end, interval="1m")

    # Calculate Binance price movement
    binance_open = binance_candles[0]["open"] if binance_candles else 0
    binance_close = binance_candles[-1]["close"] if binance_candles else 0
    binance_high = max(c["high"] for c in binance_candles) if binance_candles else 0
    binance_low = min(c["low"] for c in binance_candles) if binance_candles else 0
    price_change = binance_close - binance_open
    price_change_pct = (price_change / binance_open * 100) if binance_open else 0
    actual_resolution = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"

    data = {
        "slug": slug,
        "window_start": window_start,
        "window_end": window_end,
        "window_start_human": datetime.fromtimestamp(window_start).isoformat(),
        "window_end_human": datetime.fromtimestamp(window_end).isoformat(),
        "collected_at": int(time.time()),
        # Polymarket whale trade data
        "total_trades": len(all_trades),
        "net_up": net_up,
        "net_down": net_down,
        "total_volume_usd": total_vol,
        "bias": bias,
        "bias_confidence": conf,
        "trades": all_trades,
        # Binance price data
        "binance": {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "num_candles": len(binance_candles),
            "open": binance_open,
            "close": binance_close,
            "high": binance_high,
            "low": binance_low,
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "actual_resolution": actual_resolution,
            "candles": binance_candles,
        },
        # Combined analysis
        "whale_predicted_correct": bias == actual_resolution if bias != "NEUTRAL" else None,
    }

    filepath = DATA_DIR / f"{slug}.json"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    log(f"COMPLETE: {len(all_trades):,} trades | Bias: {bias} ({conf:.1f}%) | "
        f"Binance: {price_change_pct:+.3f}% ({actual_resolution}) | "
        f"Correct: {data['whale_predicted_correct']} | Saved: {filepath.name}")
    return data


def main():
    log("=" * 60)
    log("POLYMARKET TRADE COLLECTOR - 20 MARKETS")
    log("=" * 60)
    log(f"Whale: gabagool22")
    log(f"Data dir: {DATA_DIR}")
    log("")

    # Count existing markets
    existing = list(DATA_DIR.glob("btc-updown-15m-*.json"))
    log(f"Existing markets: {len(existing)}")

    collected = []
    markets_needed = MARKETS_TO_COLLECT - len(existing)

    for i in range(markets_needed):
        # Get next market window
        now = int(time.time())
        current_start = (now // 900) * 900
        next_start = current_start + 900
        next_end = next_start + 900
        slug = f"btc-updown-15m-{next_start}"

        # Skip if exists
        if (DATA_DIR / f"{slug}.json").exists():
            log(f"Skipping {slug} (exists)")
            continue

        log(f"\n[Market {len(existing) + i + 1}/{MARKETS_TO_COLLECT}]")
        result = collect_market(slug, next_start, next_end)
        collected.append(result)

        # Summary
        log(f"\nProgress: {len(existing) + len(collected)}/{MARKETS_TO_COLLECT} markets")

    log("\n" + "=" * 60)
    log("COLLECTION COMPLETE")
    log("=" * 60)
    log(f"New markets collected: {len(collected)}")
    log(f"Total markets: {len(list(DATA_DIR.glob('btc-updown-15m-*.json')))}")


if __name__ == "__main__":
    main()
