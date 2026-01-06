#!/usr/bin/env python3
"""
Trader Timing Analysis - Scatter Chart Visualization

Creates a visualization showing:
1. Scatter plot of trader order timings (x) vs size (y) for:
   - Account88888 (69% win rate whale)
   - gabagool22 (53% win rate)
   - PurpleThunderBicycleMountain (high volume trader)

2. Below: BTC price and volume chart using 15-second candles

This helps identify:
- When each trader typically enters the market
- Size patterns relative to timing
- Correlation between entries and BTC momentum
"""

import json
import sys
import time
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Unbuffer stdout
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path("/Users/stephenpeters/Documents/Business/predmkt/backend/strategy_backtest/data/markets")
OUTPUT_DIR = Path("/Users/stephenpeters/Documents/Business/predmkt/backend/strategy_backtest/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Trader wallets to analyze with PnL data from Polymarket profiles
TRADERS = {
    "Account88888": {
        "address": "0x7f69983eb28245bba0d5083502a78744a8f66162",
        "pnl": 446756.16,  # Total profit from profile
        "volume": 102_487_193.62,
        "total_trades": 10447,
        "marker": "^",  # Triangle
    },
    "gabagool22": {
        "address": "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d",
        "pnl": 529583.10,  # Total profit from profile
        "volume": 65_978_560.44,
        "total_trades": 16459,
        "marker": "v",  # Inverted triangle
    },
    "updateupdate": {
        "address": "0xd0d6053c3c37e727402d84c14069780d360993aa",
        "pnl": 66978.29,  # Total profit from profile
        "volume": 12_210_779.77,
        "total_trades": 1778,
        "marker": "D",  # Diamond
    },
}

# Colors for buy/sell
BUY_COLOR = "#10b981"   # Green
SELL_COLOR = "#ef4444"  # Red

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_binance_candles_1s(window_start: int, window_end: int) -> list:
    """Fetch 1-second Binance candles for the window."""
    try:
        start_ms = window_start * 1000
        end_ms = window_end * 1000

        resp = requests.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": "BTCUSDT",
                "interval": "1s",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )

        if resp.status_code != 200:
            print(f"  Binance API error: {resp.status_code}")
            return []

        raw = resp.json()
        candles = []
        for k in raw:
            candles.append({
                "timestamp": k[0] // 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "quote_volume": float(k[7]),
            })
        return candles

    except Exception as e:
        print(f"  Fetch error: {e}")
        return []


def aggregate_candles(candles_1s: list, bucket_seconds: int = 15) -> list:
    """Aggregate 1s candles to larger buckets."""
    if not candles_1s:
        return []

    aggregated = []
    for i in range(0, len(candles_1s), bucket_seconds):
        bucket = candles_1s[i:i + bucket_seconds]
        if not bucket:
            continue

        aggregated.append({
            "timestamp": bucket[0]["timestamp"],
            "open": bucket[0]["open"],
            "high": max(c["high"] for c in bucket),
            "low": min(c["low"] for c in bucket),
            "close": bucket[-1]["close"],
            "volume": sum(c["volume"] for c in bucket),
            "quote_volume": sum(c["quote_volume"] for c in bucket),
        })

    return aggregated


def extract_trader_trades(all_trades: list, wallet: str, window_start: int) -> list:
    """Extract trades for a specific wallet from the pre-collected trade data."""
    processed = []
    wallet_lower = wallet.lower()

    for t in all_trades:
        # Check if this trade is from the target wallet
        proxy_wallet = t.get("proxyWallet", "").lower()
        if proxy_wallet != wallet_lower:
            continue

        # Parse timestamp
        ts = t.get("timestamp", 0)
        if isinstance(ts, str):
            try:
                from datetime import datetime
                ts = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
            except:
                continue

        elapsed = ts - window_start
        if 0 <= elapsed <= 900:
            processed.append({
                "elapsed": elapsed,
                "size": float(t.get("size", 0)),
                "price": float(t.get("price", 0)),
                "usd_value": float(t.get("size", 0)) * float(t.get("price", 0)),
                "side": t.get("side", ""),
                "outcome": t.get("outcome", ""),
            })

    return processed


def analyze_single_market(market_file: Path) -> dict:
    """Analyze a single market's trader timing patterns."""
    with open(market_file) as f:
        data = json.load(f)

    slug = data["slug"]
    window_start = data["window_start"]
    window_end = data["window_end"]
    all_trades = data.get("trades", [])

    print(f"\nAnalyzing {slug}...")
    print(f"  Total trades in file: {len(all_trades)}")

    # Get Binance candles (use existing if available, otherwise fetch)
    if "binance" in data and "candles_15s" in data["binance"]:
        candles_15s = data["binance"]["candles_15s"]
        print(f"  Using cached {len(candles_15s)} 15s candles")
    elif "binance" in data and "candles_1s" in data["binance"]:
        candles_1s = data["binance"]["candles_1s"]
        candles_15s = aggregate_candles(candles_1s, 15)
        print(f"  Aggregated {len(candles_15s)} 15s candles from 1s data")
    else:
        print(f"  Fetching Binance 1s candles...")
        candles_1s = fetch_binance_candles_1s(window_start, window_end)
        candles_15s = aggregate_candles(candles_1s, 15)
        print(f"  Fetched and aggregated {len(candles_15s)} 15s candles")

    # Extract trades for each trader from existing data
    trader_data = {}
    for name, info in TRADERS.items():
        trades = extract_trader_trades(all_trades, info["address"], window_start)
        trader_data[name] = trades
        print(f"  {name}: {len(trades)} trades in window")

    return {
        "slug": slug,
        "window_start": window_start,
        "window_end": window_end,
        "resolution": data.get("binance", {}).get("actual_resolution", "UNKNOWN"),
        "price_change_pct": data.get("binance", {}).get("price_change_pct", 0),
        "candles_15s": candles_15s,
        "traders": trader_data,
    }


def create_timing_chart(analysis_data: dict, output_path: Path):
    """Create the scatter + price/volume chart with aligned x-axes."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={'height_ratios': [2.5, 1, 1], 'hspace': 0.05})
    ax1, ax2, ax3 = axes

    # =========================================================================
    # TOP: Scatter plot of trader timings - Green/Red triangles for BUY/SELL
    # Each trader has a unique marker shape for identification
    # =========================================================================
    for name, info in TRADERS.items():
        trades = analysis_data["traders"].get(name, [])
        if not trades:
            continue

        # Separate buys and sells
        buys = [t for t in trades if t["side"] == "BUY"]
        sells = [t for t in trades if t["side"] == "SELL"]

        # Get trader's unique marker shape
        marker = info["marker"]  # ^, v, D for each trader
        pnl = info.get("pnl", 0)
        pnl_str = f"+${pnl/1000:.0f}k" if pnl >= 1000 else f"+${pnl:.0f}"

        # Buys - GREEN with trader's marker shape (or ^ if marker is v)
        buy_marker = "^" if marker == "v" else marker
        if buys:
            ax1.scatter(
                [t["elapsed"] for t in buys],
                [t["usd_value"] for t in buys],
                c=BUY_COLOR, marker=buy_marker, s=80, alpha=0.85,
                label=f"{name} ({pnl_str})", edgecolors='#047857', linewidths=0.8
            )

        # Sells - RED with trader's marker shape (inverted if triangle)
        sell_marker = "v" if marker == "^" else marker
        if sells:
            ax1.scatter(
                [t["elapsed"] for t in sells],
                [t["usd_value"] for t in sells],
                c=SELL_COLOR, marker=sell_marker, s=80, alpha=0.85,
                edgecolors='#b91c1c', linewidths=0.8
            )

    ax1.set_ylabel("Trade Size (USD)", fontsize=10)
    ax1.set_xlim(0, 900)
    ax1.set_title(f"Trader Timing Analysis: {analysis_data['slug']}\n"
                  f"Resolution: {analysis_data['resolution']} ({analysis_data['price_change_pct']:+.3f}%)",
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add checkpoint lines to all axes
    checkpoints = [(180, "3m"), (360, "6m"), (450, "7:30"), (540, "9m"), (720, "12m")]
    for cp, _ in checkpoints:
        ax1.axvline(x=cp, color='#a1a1aa', linestyle='--', alpha=0.4, linewidth=0.8)

    # =========================================================================
    # MIDDLE: BTC Close Price (15s candles)
    # =========================================================================
    candles = analysis_data["candles_15s"]

    if candles:
        elapsed = [(c["timestamp"] - analysis_data["window_start"]) for c in candles]
        prices = [c["close"] for c in candles]

        # Color line based on final direction
        color = "#10b981" if analysis_data["resolution"] == "UP" else "#ef4444"
        ax2.plot(elapsed, prices, color=color, linewidth=1.5, label="BTC Close")
        ax2.fill_between(elapsed, min(prices), prices, color=color, alpha=0.15)

        # Add open price reference line
        ax2.axhline(y=candles[0]["open"], color='#6366f1', linestyle=':', alpha=0.7,
                   linewidth=1.5, label=f"Open: ${candles[0]['open']:,.0f}")

        # Format y-axis as price
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    ax2.set_ylabel("BTC Price", fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Checkpoint lines
    for cp, _ in checkpoints:
        ax2.axvline(x=cp, color='#a1a1aa', linestyle='--', alpha=0.4, linewidth=0.8)

    # =========================================================================
    # BOTTOM: Buy/Sell Volume (stacked bar chart)
    # =========================================================================
    if candles:
        elapsed = [(c["timestamp"] - analysis_data["window_start"]) for c in candles]

        # Calculate buy vs sell volume per candle (green = close > open, red = close < open)
        buy_volumes = []
        sell_volumes = []

        for c in candles:
            vol = c["volume"]
            if c["close"] >= c["open"]:
                buy_volumes.append(vol)
                sell_volumes.append(0)
            else:
                buy_volumes.append(0)
                sell_volumes.append(vol)

        bar_width = 12  # Width for 15s candles

        # Stacked bar: buy (green) on bottom, sell (red) on top
        ax3.bar(elapsed, buy_volumes, width=bar_width, color='#10b981', alpha=0.7, label='Buy Vol')
        ax3.bar(elapsed, sell_volumes, width=bar_width, color='#ef4444', alpha=0.7,
               bottom=buy_volumes, label='Sell Vol')

    ax3.set_xlabel("Seconds into Market Window", fontsize=10)
    ax3.set_ylabel("Volume (BTC)", fontsize=10)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # Checkpoint lines with labels on bottom axis
    for cp, label in checkpoints:
        ax3.axvline(x=cp, color='#a1a1aa', linestyle='--', alpha=0.4, linewidth=0.8)

    # X-axis formatting
    ax3.set_xticks([0, 180, 360, 450, 540, 720, 900])
    ax3.set_xticklabels(['0', '3m', '6m', '7:30', '9m', '12m', '15m'])

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved chart: {output_path}")


def create_aggregate_chart(all_analyses: list, output_path: Path):
    """Create aggregate chart showing patterns across all markets."""
    fig = plt.figure(figsize=(14, 8))

    # Aggregate all trader data
    all_trades = defaultdict(list)

    for analysis in all_analyses:
        for name, trades in analysis["traders"].items():
            for t in trades:
                all_trades[name].append({
                    "elapsed": t["elapsed"],
                    "usd_value": t["usd_value"],
                    "side": t["side"],
                    "outcome": t.get("outcome", ""),
                    "resolution": analysis["resolution"],
                    "correct": (t["outcome"] == "Up" and analysis["resolution"] == "UP") or \
                              (t["outcome"] == "Down" and analysis["resolution"] == "DOWN"),
                })

    # Create scatter plot
    ax = fig.add_subplot(111)

    for name, info in TRADERS.items():
        trades = all_trades.get(name, [])
        if not trades:
            continue

        marker = info["marker"]
        pnl = info.get("pnl", 0)
        pnl_str = f"+${pnl/1000:.0f}k" if pnl >= 1000 else f"+${pnl:.0f}"

        # Separate by buy/sell
        buys = [t for t in trades if t["side"] == "BUY"]
        sells = [t for t in trades if t["side"] == "SELL"]

        # Buys - GREEN triangles
        buy_marker = "^" if marker == "v" else marker
        if buys:
            ax.scatter(
                [t["elapsed"] for t in buys],
                [t["usd_value"] for t in buys],
                c=BUY_COLOR, marker=buy_marker, s=60, alpha=0.7,
                label=f"{name} ({pnl_str})", edgecolors='#047857', linewidths=0.5
            )

        # Sells - RED triangles
        sell_marker = "v" if marker == "^" else marker
        if sells:
            ax.scatter(
                [t["elapsed"] for t in sells],
                [t["usd_value"] for t in sells],
                c=SELL_COLOR, marker=sell_marker, s=60, alpha=0.7,
                edgecolors='#b91c1c', linewidths=0.5
            )

    ax.set_xlabel("Seconds into Market Window", fontsize=11)
    ax.set_ylabel("Trade Size (USD)", fontsize=11)
    ax.set_xlim(0, 900)
    ax.set_title(f"Aggregate Trader Timing Patterns ({len(all_analyses)} markets)\n"
                 f"Green △ = BUY, Red ▽ = SELL (shapes per trader)",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add checkpoint lines
    for cp, label in [(180, "3m"), (360, "6m"), (450, "7:30"), (540, "9m"), (720, "12m")]:
        ax.axvline(x=cp, color='gray', linestyle='--', alpha=0.3)
        ax.text(cp, ax.get_ylim()[1] * 0.98, label, ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved aggregate chart: {output_path}")

    # Print timing statistics
    print("\n" + "=" * 60)
    print("TIMING STATISTICS")
    print("=" * 60)

    for name in TRADERS.keys():
        trades = all_trades.get(name, [])
        if not trades:
            print(f"\n{name}: No trades found")
            continue

        elapsed_times = [t["elapsed"] for t in trades]
        correct_count = sum(1 for t in trades if t["correct"])

        print(f"\n{name}:")
        print(f"  Total trades: {len(trades)}")
        print(f"  Win rate: {correct_count}/{len(trades)} ({correct_count/len(trades)*100:.1f}%)")
        print(f"  Timing (seconds into window):")
        print(f"    Min: {min(elapsed_times):.0f}s")
        print(f"    Max: {max(elapsed_times):.0f}s")
        print(f"    Mean: {sum(elapsed_times)/len(elapsed_times):.0f}s")
        print(f"    Median: {sorted(elapsed_times)[len(elapsed_times)//2]:.0f}s")

        # Breakdown by checkpoint
        buckets = {"0-3m": 0, "3-6m": 0, "6-9m": 0, "9-12m": 0, "12-15m": 0}
        for t in elapsed_times:
            if t < 180:
                buckets["0-3m"] += 1
            elif t < 360:
                buckets["3-6m"] += 1
            elif t < 540:
                buckets["6-9m"] += 1
            elif t < 720:
                buckets["9-12m"] += 1
            else:
                buckets["12-15m"] += 1

        print(f"  Distribution by time bucket:")
        for bucket, count in buckets.items():
            pct = count / len(trades) * 100
            bar = "=" * int(pct / 5)
            print(f"    {bucket}: {count:3d} ({pct:5.1f}%) {bar}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze trader timing patterns")
    parser.add_argument("--limit", type=int, default=5, help="Max markets to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all markets")
    args = parser.parse_args()

    print("=" * 60)
    print("TRADER TIMING ANALYSIS")
    print("=" * 60)
    print(f"Traders: {', '.join(TRADERS.keys())}")

    # Find market files
    market_files = sorted(DATA_DIR.glob("btc-updown-15m-*.json"))
    print(f"Found {len(market_files)} market files")

    if not args.all:
        market_files = market_files[:args.limit]
        print(f"Analyzing first {len(market_files)} markets (use --all for all)")

    # Analyze each market
    all_analyses = []
    for i, filepath in enumerate(market_files):
        print(f"\n[{i+1}/{len(market_files)}] Processing {filepath.name}")
        analysis = analyze_single_market(filepath)
        all_analyses.append(analysis)

        # Create individual chart
        chart_path = OUTPUT_DIR / f"timing_{filepath.stem}.png"
        create_timing_chart(analysis, chart_path)

    # Create aggregate chart
    if all_analyses:
        aggregate_path = OUTPUT_DIR / "timing_aggregate.png"
        create_aggregate_chart(all_analyses, aggregate_path)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Individual charts: {OUTPUT_DIR}/timing_*.png")
    print(f"Aggregate chart: {OUTPUT_DIR}/timing_aggregate.png")


if __name__ == "__main__":
    main()
