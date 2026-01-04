#!/usr/bin/env python3
"""
Backtest for Latency Arbitrage Strategy on Polymarket 15-Minute BTC Markets.

This script validates the latency arbitrage strategy by:
1. Fetching resolved BTC markets from the last 7 days (up to 500 markets)
2. Getting Binance 15-second OHLCV data for each market window
3. Applying the strategy at 3m, 6m, 9m, 12m checkpoints
4. Comparing predicted direction vs actual market resolution

Usage:
    python backtest_strategy.py
    python backtest_strategy.py --days 7 --checkpoints 3,6,9,12
"""

import argparse
import asyncio
import aiohttp
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path


# ============================================================================
# CONFIGURATION (matches paper_trading.py defaults)
# ============================================================================

# Strategy thresholds
MIN_EDGE_PCT = 5.0  # 5% minimum edge
MIN_VOLUME_DELTA_USD = 10000.0  # $10K volume delta for confirmation
MIN_ORDERBOOK_IMBALANCE = 0.1  # 10% order book imbalance

# Checkpoints (minutes into market window)
DEFAULT_CHECKPOINTS = [3, 6, 9, 12]

# API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
BINANCE_API = "https://api.binance.com"

# Output directory
OUTPUT_DIR = Path("backtest_results")


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class MarketWindow:
    """A resolved 15-minute Polymarket market"""
    window_start: int  # Unix timestamp
    window_end: int
    slug: str
    condition_id: str
    resolution: str  # "UP" or "DOWN"
    volume: float
    liquidity: float


@dataclass
class BinanceOHLCV:
    """15-second OHLCV bar from Binance"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    num_trades: int
    taker_buy_volume: float
    taker_buy_quote_volume: float


@dataclass
class CheckpointAnalysis:
    """Analysis at a specific checkpoint"""
    checkpoint_min: int  # 3, 6, 9, or 12
    time_into_window_sec: int
    binance_open: float  # Price at window start
    binance_current: float  # Price at checkpoint
    price_change_pct: float
    implied_up_prob: float
    predicted_direction: str  # "UP", "DOWN", or "NO_SIGNAL"
    edge: float
    volume_delta: float  # Buy volume - sell volume
    volume_delta_pct: float  # As percentage
    meets_edge_threshold: bool
    meets_volume_threshold: bool
    time_remaining_sec: int


@dataclass
class MarketBacktest:
    """Complete backtest for a single market"""
    market: MarketWindow
    binance_open: float
    binance_close: float
    checkpoints: list[CheckpointAnalysis]


@dataclass
class BacktestResults:
    """Aggregate backtest results"""
    total_markets: int
    markets_analyzed: int
    markets_skipped: int
    by_checkpoint: dict[int, dict]  # checkpoint_min -> stats
    overall_accuracy: float
    avg_edge_when_correct: float
    avg_edge_when_wrong: float


# ============================================================================
# PROBABILITY CALCULATION (matches paper_trading.py)
# ============================================================================

def calculate_implied_probability(price_change_pct: float, time_remaining_sec: int) -> float:
    """
    Calculate implied probability that UP will win based on:
    1. Current price change from window open
    2. Time remaining in the window

    Matches logic from paper_trading.py
    """
    base_prob = 0.5
    abs_change = abs(price_change_pct)

    if abs_change < 0.01:
        price_impact = 0
    elif abs_change < 0.05:
        price_impact = abs_change * 2  # 0.05% move = 10% edge
    elif abs_change < 0.2:
        price_impact = 0.1 + (abs_change - 0.05) * 1.5  # 0.2% move = 32% edge
    else:
        price_impact = min(0.45, 0.32 + (abs_change - 0.2) * 0.5)  # Max out at 95%

    # Time dampening
    time_factor = 1.0
    if time_remaining_sec > 600:
        time_factor = 0.7
    elif time_remaining_sec > 300:
        time_factor = 0.85
    elif time_remaining_sec > 120:
        time_factor = 0.95

    adjusted_impact = price_impact * time_factor

    if price_change_pct >= 0:
        return min(0.95, base_prob + adjusted_impact)
    else:
        return max(0.05, base_prob - adjusted_impact)


# ============================================================================
# DATA FETCHERS
# ============================================================================

async def fetch_resolved_markets(
    session: aiohttp.ClientSession,
    start_timestamp: int,
    end_timestamp: int,
    max_markets: int = 500
) -> list[MarketWindow]:
    """Fetch resolved BTC markets from Polymarket"""
    markets = []
    current_window = (start_timestamp // 900) * 900
    end_window = (end_timestamp // 900) * 900

    # Work backwards from end to get most recent markets
    windows_to_check = []
    check_window = end_window
    while check_window >= current_window and len(windows_to_check) < max_markets * 2:
        windows_to_check.append(check_window)
        check_window -= 900  # 15 minutes

    print(f"Checking {len(windows_to_check)} potential market windows...")

    # Batch fetch - check multiple windows concurrently
    batch_size = 20
    for i in range(0, len(windows_to_check), batch_size):
        batch = windows_to_check[i:i+batch_size]

        tasks = [
            fetch_single_market(session, window_ts)
            for window_ts in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, MarketWindow):
                markets.append(result)
                if len(markets) >= max_markets:
                    break

        if len(markets) >= max_markets:
            break

        # Progress update
        print(f"  Found {len(markets)} resolved markets so far...")
        await asyncio.sleep(0.5)  # Rate limiting

    # Sort by window start (oldest first)
    markets.sort(key=lambda m: m.window_start)

    return markets


async def fetch_single_market(
    session: aiohttp.ClientSession,
    window_timestamp: int
) -> Optional[MarketWindow]:
    """Fetch a single market by timestamp"""
    slug = f"btc-updown-15m-{window_timestamp}"
    url = f"{GAMMA_API}/markets"
    params = {"slug": slug}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with session.get(url, params=params, headers=headers, timeout=10) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            if not data:
                return None

            market = data[0]

            # Check if resolved
            if not market.get("closed", False):
                return None

            # Parse outcome
            outcome_prices = market.get("outcomePrices", "[0.5, 0.5]")
            if isinstance(outcome_prices, str):
                prices = json.loads(outcome_prices)
            else:
                prices = outcome_prices

            up_price = float(prices[0]) if prices else 0.5
            resolution = "UP" if up_price > 0.5 else "DOWN"

            return MarketWindow(
                window_start=window_timestamp,
                window_end=window_timestamp + 900,
                slug=market.get("slug", slug),
                condition_id=market.get("conditionId", ""),
                resolution=resolution,
                volume=float(market.get("volume", 0) or 0),
                liquidity=float(market.get("liquidity", 0) or 0),
            )

    except Exception as e:
        return None


async def fetch_binance_klines(
    session: aiohttp.ClientSession,
    start_time: int,
    end_time: int,
    interval: str = "1s"
) -> list[BinanceOHLCV]:
    """
    Fetch Binance klines for BTC/USDT.

    Uses 1-second klines for maximum granularity during backtest.
    For a 15-minute window, we need ~900 1-second candles.
    """
    url = f"{BINANCE_API}/api/v3/klines"
    all_klines = []

    # Convert to milliseconds
    start_ms = start_time * 1000
    end_ms = end_time * 1000

    # Binance limits to 1000 klines per request
    # For 1s interval and 15 min window (900 seconds), we need 1 request
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000
        }

        try:
            async with session.get(url, params=params, timeout=15) as resp:
                if resp.status != 200:
                    # Try with 1m interval as fallback
                    if interval == "1s":
                        return await fetch_binance_klines(session, start_time, end_time, "1m")
                    return all_klines

                data = await resp.json()
                if not data:
                    break

                for k in data:
                    all_klines.append(BinanceOHLCV(
                        timestamp=k[0] // 1000,  # Convert back to seconds
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                        quote_volume=float(k[7]),
                        num_trades=int(k[8]),
                        taker_buy_volume=float(k[9]),
                        taker_buy_quote_volume=float(k[10]),
                    ))

                # Move to next batch
                if len(data) < 1000:
                    break
                current_start = data[-1][0] + 1

        except Exception as e:
            # Fallback to 1m interval if 1s fails
            if interval == "1s":
                return await fetch_binance_klines(session, start_time, end_time, "1m")
            break

    return all_klines


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_checkpoint(
    klines: list[BinanceOHLCV],
    window_start: int,
    checkpoint_min: int,
    window_end: int
) -> Optional[CheckpointAnalysis]:
    """Analyze a specific checkpoint within the market window"""

    if not klines:
        return None

    # Find the kline at window start (closest to start)
    open_price = None
    for k in klines:
        if k.timestamp >= window_start:
            open_price = k.open
            break

    if open_price is None:
        return None

    # Calculate checkpoint timestamp
    checkpoint_sec = checkpoint_min * 60
    checkpoint_ts = window_start + checkpoint_sec

    # Find the kline at checkpoint (or closest before)
    checkpoint_price = None
    checkpoint_kline = None
    for k in klines:
        if k.timestamp <= checkpoint_ts:
            checkpoint_price = k.close
            checkpoint_kline = k
        else:
            break

    if checkpoint_price is None:
        return None

    # Calculate price change
    price_change_pct = ((checkpoint_price - open_price) / open_price) * 100

    # Calculate time remaining
    time_remaining = window_end - checkpoint_ts

    # Calculate implied probability
    implied_up_prob = calculate_implied_probability(price_change_pct, time_remaining)

    # Calculate edge (assuming Polymarket is at 50% at window start)
    # In a real scenario, we'd have the actual Polymarket price
    # Here we estimate: if price moved, Polymarket lags 30-60 sec
    polymarket_lagged_price = 0.5  # Assume starting at 50%
    edge = implied_up_prob - polymarket_lagged_price

    # Determine predicted direction
    predicted_direction = "NO_SIGNAL"
    if abs(edge) >= MIN_EDGE_PCT / 100:
        predicted_direction = "UP" if edge > 0 else "DOWN"

    # Calculate volume delta (taker buy - taker sell)
    # Use recent klines up to checkpoint
    recent_klines = [k for k in klines if window_start <= k.timestamp <= checkpoint_ts]

    total_taker_buy_volume = sum(k.taker_buy_quote_volume for k in recent_klines)
    total_quote_volume = sum(k.quote_volume for k in recent_klines)
    taker_sell_volume = total_quote_volume - total_taker_buy_volume

    volume_delta = total_taker_buy_volume - taker_sell_volume
    volume_delta_pct = (volume_delta / total_quote_volume * 100) if total_quote_volume > 0 else 0

    return CheckpointAnalysis(
        checkpoint_min=checkpoint_min,
        time_into_window_sec=checkpoint_sec,
        binance_open=open_price,
        binance_current=checkpoint_price,
        price_change_pct=price_change_pct,
        implied_up_prob=implied_up_prob,
        predicted_direction=predicted_direction,
        edge=edge,
        volume_delta=volume_delta,
        volume_delta_pct=volume_delta_pct,
        meets_edge_threshold=abs(edge) >= MIN_EDGE_PCT / 100,
        meets_volume_threshold=abs(volume_delta) >= MIN_VOLUME_DELTA_USD,
        time_remaining_sec=time_remaining,
    )


def backtest_market(
    market: MarketWindow,
    klines: list[BinanceOHLCV],
    checkpoints: list[int]
) -> Optional[MarketBacktest]:
    """Run backtest for a single market"""

    if not klines:
        return None

    # Get open and close prices from klines
    open_price = klines[0].open if klines else None
    close_price = klines[-1].close if klines else None

    if open_price is None or close_price is None:
        return None

    # Analyze each checkpoint
    checkpoint_analyses = []
    for cp_min in checkpoints:
        analysis = analyze_checkpoint(
            klines=klines,
            window_start=market.window_start,
            checkpoint_min=cp_min,
            window_end=market.window_end
        )
        if analysis:
            checkpoint_analyses.append(analysis)

    return MarketBacktest(
        market=market,
        binance_open=open_price,
        binance_close=close_price,
        checkpoints=checkpoint_analyses,
    )


def calculate_results(
    backtests: list[MarketBacktest],
    checkpoints: list[int]
) -> BacktestResults:
    """Calculate aggregate results from all backtests"""

    # Initialize per-checkpoint stats
    by_checkpoint = {}
    for cp in checkpoints:
        by_checkpoint[cp] = {
            "total_signals": 0,
            "correct_signals": 0,
            "incorrect_signals": 0,
            "no_signal": 0,
            "accuracy": 0.0,
            "avg_edge": 0.0,
            "edges_correct": [],
            "edges_incorrect": [],
        }

    markets_analyzed = len(backtests)

    for bt in backtests:
        for cp_analysis in bt.checkpoints:
            cp_stats = by_checkpoint[cp_analysis.checkpoint_min]

            if cp_analysis.predicted_direction == "NO_SIGNAL":
                cp_stats["no_signal"] += 1
            else:
                cp_stats["total_signals"] += 1

                is_correct = cp_analysis.predicted_direction == bt.market.resolution

                if is_correct:
                    cp_stats["correct_signals"] += 1
                    cp_stats["edges_correct"].append(abs(cp_analysis.edge))
                else:
                    cp_stats["incorrect_signals"] += 1
                    cp_stats["edges_incorrect"].append(abs(cp_analysis.edge))

    # Calculate accuracy and averages
    total_signals = 0
    total_correct = 0
    all_edges_correct = []
    all_edges_incorrect = []

    for cp in checkpoints:
        stats = by_checkpoint[cp]
        if stats["total_signals"] > 0:
            stats["accuracy"] = stats["correct_signals"] / stats["total_signals"] * 100
            stats["avg_edge"] = (
                sum(stats["edges_correct"]) + sum(stats["edges_incorrect"])
            ) / stats["total_signals"]

        total_signals += stats["total_signals"]
        total_correct += stats["correct_signals"]
        all_edges_correct.extend(stats["edges_correct"])
        all_edges_incorrect.extend(stats["edges_incorrect"])

        # Clean up temp data
        del stats["edges_correct"]
        del stats["edges_incorrect"]

    overall_accuracy = (total_correct / total_signals * 100) if total_signals > 0 else 0
    avg_edge_correct = sum(all_edges_correct) / len(all_edges_correct) if all_edges_correct else 0
    avg_edge_incorrect = sum(all_edges_incorrect) / len(all_edges_incorrect) if all_edges_incorrect else 0

    return BacktestResults(
        total_markets=markets_analyzed,
        markets_analyzed=markets_analyzed,
        markets_skipped=0,
        by_checkpoint=by_checkpoint,
        overall_accuracy=overall_accuracy,
        avg_edge_when_correct=avg_edge_correct,
        avg_edge_when_wrong=avg_edge_incorrect,
    )


# ============================================================================
# MAIN BACKTEST
# ============================================================================

async def run_backtest(
    days: int = 7,
    max_markets: int = 500,
    checkpoints: list[int] = None
):
    """Run the full backtest"""

    if checkpoints is None:
        checkpoints = DEFAULT_CHECKPOINTS

    print("=" * 70)
    print("LATENCY ARBITRAGE STRATEGY BACKTEST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Period: Last {days} days")
    print(f"  Max markets: {max_markets}")
    print(f"  Checkpoints: {checkpoints} minutes")
    print(f"  Min edge threshold: {MIN_EDGE_PCT}%")
    print(f"  Min volume delta: ${MIN_VOLUME_DELTA_USD:,.0f}")
    print(f"  Min orderbook imbalance: {MIN_ORDERBOOK_IMBALANCE * 100}%")
    print()

    # Calculate time range
    end_timestamp = int(time.time())
    start_timestamp = end_timestamp - (days * 24 * 3600)

    print(f"Time range:")
    print(f"  Start: {datetime.fromtimestamp(start_timestamp)}")
    print(f"  End: {datetime.fromtimestamp(end_timestamp)}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        # Step 1: Fetch resolved markets
        print("Step 1: Fetching resolved BTC markets from Polymarket...")
        markets = await fetch_resolved_markets(
            session, start_timestamp, end_timestamp, max_markets
        )
        print(f"  Found {len(markets)} resolved markets\n")

        if not markets:
            print("No markets found! Exiting.")
            return

        # Step 2: Fetch Binance klines for each market
        print("Step 2: Fetching Binance OHLCV data for each market...")
        backtests = []

        for i, market in enumerate(markets):
            if i % 10 == 0:
                print(f"  Processing market {i+1}/{len(markets)}...")

            klines = await fetch_binance_klines(
                session,
                market.window_start,
                market.window_end
            )

            if klines:
                bt = backtest_market(market, klines, checkpoints)
                if bt:
                    backtests.append(bt)

            # Rate limiting
            await asyncio.sleep(0.1)

        print(f"  Successfully analyzed {len(backtests)} markets\n")

        # Step 3: Calculate results
        print("Step 3: Calculating results...")
        results = calculate_results(backtests, checkpoints)

        # Print results
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        print(f"\nOverview:")
        print(f"  Markets analyzed: {results.markets_analyzed}")
        print(f"  Overall accuracy: {results.overall_accuracy:.1f}%")
        print(f"  Avg edge when correct: {results.avg_edge_when_correct * 100:.2f}%")
        print(f"  Avg edge when wrong: {results.avg_edge_when_wrong * 100:.2f}%")

        print(f"\nResults by checkpoint:")
        print("-" * 70)
        print(f"{'Checkpoint':<12} {'Signals':<10} {'Correct':<10} {'Accuracy':<12} {'Avg Edge':<12}")
        print("-" * 70)

        for cp in checkpoints:
            stats = results.by_checkpoint[cp]
            print(
                f"{cp}m{'':<10} "
                f"{stats['total_signals']:<10} "
                f"{stats['correct_signals']:<10} "
                f"{stats['accuracy']:.1f}%{'':<8} "
                f"{stats['avg_edge'] * 100:.2f}%"
            )

        print("-" * 70)

        # Detailed analysis
        print(f"\nSignal breakdown:")
        for cp in checkpoints:
            stats = results.by_checkpoint[cp]
            total_markets = stats['total_signals'] + stats['no_signal']
            signal_rate = stats['total_signals'] / total_markets * 100 if total_markets > 0 else 0
            print(f"  {cp}m: {stats['total_signals']} signals / {total_markets} markets ({signal_rate:.1f}% signal rate)")

        # Resolution distribution
        if backtests:
            up_count = sum(1 for bt in backtests if bt.market.resolution == "UP")
            down_count = len(backtests) - up_count
            print(f"\nMarket resolutions:")
            print(f"  UP: {up_count} ({up_count/len(backtests)*100:.1f}%)")
            print(f"  DOWN: {down_count} ({down_count/len(backtests)*100:.1f}%)")
        else:
            print("\nNo markets analyzed - check Binance data availability.")

        # Save detailed results
        save_results(backtests, results, checkpoints)


def save_results(
    backtests: list[MarketBacktest],
    results: BacktestResults,
    checkpoints: list[int]
):
    """Save detailed results to files"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary = {
        "run_time": datetime.now().isoformat(),
        "config": {
            "min_edge_pct": MIN_EDGE_PCT,
            "min_volume_delta_usd": MIN_VOLUME_DELTA_USD,
            "min_orderbook_imbalance": MIN_ORDERBOOK_IMBALANCE,
            "checkpoints": checkpoints,
        },
        "results": {
            "total_markets": results.total_markets,
            "markets_analyzed": results.markets_analyzed,
            "overall_accuracy": round(results.overall_accuracy, 2),
            "avg_edge_when_correct": round(results.avg_edge_when_correct, 4),
            "avg_edge_when_wrong": round(results.avg_edge_when_wrong, 4),
            "by_checkpoint": results.by_checkpoint,
        }
    }

    summary_path = OUTPUT_DIR / f"backtest_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {summary_path}")

    # Save detailed per-market results
    detailed = []
    for bt in backtests:
        market_result = {
            "window_start": bt.market.window_start,
            "window_end": bt.market.window_end,
            "resolution": bt.market.resolution,
            "binance_open": bt.binance_open,
            "binance_close": bt.binance_close,
            "binance_change_pct": round((bt.binance_close - bt.binance_open) / bt.binance_open * 100, 4),
            "volume": bt.market.volume,
            "checkpoints": []
        }

        for cp in bt.checkpoints:
            market_result["checkpoints"].append({
                "checkpoint_min": cp.checkpoint_min,
                "price_change_pct": round(cp.price_change_pct, 4),
                "implied_up_prob": round(cp.implied_up_prob, 3),
                "predicted_direction": cp.predicted_direction,
                "edge": round(cp.edge, 4),
                "volume_delta": round(cp.volume_delta, 2),
                "correct": cp.predicted_direction == bt.market.resolution if cp.predicted_direction != "NO_SIGNAL" else None,
            })

        detailed.append(market_result)

    detailed_path = OUTPUT_DIR / f"backtest_detailed_{timestamp}.json"
    with open(detailed_path, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"Detailed results saved to: {detailed_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest latency arbitrage strategy on Polymarket BTC markets"
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Number of days of historical data (default: 7)"
    )
    parser.add_argument(
        "--max-markets", type=int, default=500,
        help="Maximum number of markets to analyze (default: 500)"
    )
    parser.add_argument(
        "--checkpoints", type=str, default="3,6,9,12",
        help="Checkpoint minutes, comma-separated (default: 3,6,9,12)"
    )

    args = parser.parse_args()

    checkpoints = [int(x.strip()) for x in args.checkpoints.split(",")]

    asyncio.run(run_backtest(
        days=args.days,
        max_markets=args.max_markets,
        checkpoints=checkpoints,
    ))


if __name__ == "__main__":
    main()
