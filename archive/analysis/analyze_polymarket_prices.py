#!/usr/bin/env python3
"""
Analyze Polymarket price evolution vs Binance during 15-minute windows.

This script:
1. Fetches recent resolved BTC markets from Polymarket
2. Gets Polymarket trades during each window to reconstruct price evolution
3. Gets Binance 1-second klines for the same windows
4. Compares how Polymarket UP prices moved vs Binance price at each checkpoint

This shows us the actual latency gap between Binance and Polymarket.
"""

import argparse
import asyncio
import aiohttp
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
BINANCE_API = "https://api.binance.com"

OUTPUT_DIR = Path("price_analysis")


@dataclass
class PriceSnapshot:
    """Price state at a specific moment"""
    time_into_window_sec: int
    timestamp: int

    # Binance data
    binance_price: float
    binance_change_from_open_pct: float

    # Polymarket data (from trades)
    polymarket_up_price: Optional[float]
    polymarket_trades_count: int
    polymarket_volume: float

    # Implied fair value from Binance
    implied_up_prob: float

    # Lag/gap metrics
    price_gap: Optional[float]  # implied - polymarket (positive = Polymarket lagging)


@dataclass
class WindowAnalysis:
    """Full analysis of a single market window"""
    window_start: int
    window_end: int
    resolution: str
    binance_open: float
    binance_close: float
    snapshots: list[PriceSnapshot]
    total_polymarket_trades: int
    total_polymarket_volume: float


def calculate_implied_probability(price_change_pct: float, time_remaining_sec: int) -> float:
    """Calculate implied UP probability from Binance price move"""
    base_prob = 0.5
    abs_change = abs(price_change_pct)

    if abs_change < 0.01:
        price_impact = 0
    elif abs_change < 0.05:
        price_impact = abs_change * 2
    elif abs_change < 0.2:
        price_impact = 0.1 + (abs_change - 0.05) * 1.5
    else:
        price_impact = min(0.45, 0.32 + (abs_change - 0.2) * 0.5)

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


async def fetch_market(session: aiohttp.ClientSession, window_ts: int) -> Optional[dict]:
    """Fetch a single market by timestamp"""
    slug = f"btc-updown-15m-{window_ts}"
    url = f"{GAMMA_API}/markets"
    params = {"slug": slug}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with session.get(url, params=params, headers=headers, timeout=10) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            if not data or not data[0].get("closed"):
                return None
            return data[0]
    except:
        return None


async def fetch_polymarket_trades(
    session: aiohttp.ClientSession,
    condition_id: str,
    limit: int = 1000
) -> list[dict]:
    """Fetch trades from Polymarket"""
    url = f"{DATA_API}/trades"
    params = {"market": condition_id, "limit": limit}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with session.get(url, params=params, headers=headers, timeout=15) as resp:
            if resp.status != 200:
                return []
            return await resp.json()
    except:
        return []


async def fetch_binance_klines(
    session: aiohttp.ClientSession,
    start_time: int,
    end_time: int
) -> list[dict]:
    """Fetch Binance 1-second klines"""
    url = f"{BINANCE_API}/api/v3/klines"
    all_klines = []

    start_ms = start_time * 1000
    end_ms = end_time * 1000
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1s",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000
        }

        try:
            async with session.get(url, params=params, timeout=15) as resp:
                if resp.status != 200:
                    break
                data = await resp.json()
                if not data:
                    break
                all_klines.extend(data)
                if len(data) < 1000:
                    break
                current_start = data[-1][0] + 1
        except:
            break

    return all_klines


def analyze_window(
    window_start: int,
    window_end: int,
    resolution: str,
    poly_trades: list[dict],
    binance_klines: list[dict],
    snapshot_interval_sec: int = 30
) -> WindowAnalysis:
    """Analyze price evolution throughout a window"""

    # Parse Binance klines
    binance_prices = {}
    for k in binance_klines:
        ts = k[0] // 1000
        binance_prices[ts] = {
            "open": float(k[1]),
            "close": float(k[4]),
        }

    # Get Binance open price (first kline)
    binance_open = None
    binance_close = None
    if binance_klines:
        binance_open = float(binance_klines[0][1])
        binance_close = float(binance_klines[-1][4])

    # Parse Polymarket trades - get UP trades
    up_trades = [t for t in poly_trades if t.get("outcome") == "Up"]
    up_trades.sort(key=lambda t: t.get("timestamp", 0))

    # Create snapshots at regular intervals
    snapshots = []

    for sec in range(0, 901, snapshot_interval_sec):
        ts = window_start + sec
        time_remaining = window_end - ts

        # Get Binance price at this moment
        binance_price = None
        if ts in binance_prices:
            binance_price = binance_prices[ts]["close"]
        elif binance_prices:
            # Find closest
            closest_ts = min(binance_prices.keys(), key=lambda x: abs(x - ts))
            if abs(closest_ts - ts) <= snapshot_interval_sec:
                binance_price = binance_prices[closest_ts]["close"]

        if binance_price is None or binance_open is None:
            continue

        binance_change_pct = ((binance_price - binance_open) / binance_open) * 100
        implied_up = calculate_implied_probability(binance_change_pct, time_remaining)

        # Get Polymarket UP price from recent trades
        recent_up_trades = [t for t in up_trades
                           if window_start <= t.get("timestamp", 0) <= ts]

        poly_up_price = None
        poly_volume = 0
        if recent_up_trades:
            # Use last trade price
            poly_up_price = recent_up_trades[-1].get("price")
            poly_volume = sum(float(t.get("size", 0) or 0) * float(t.get("price", 0) or 0)
                            for t in recent_up_trades)

        # Calculate gap
        price_gap = None
        if poly_up_price is not None:
            price_gap = implied_up - poly_up_price

        snapshots.append(PriceSnapshot(
            time_into_window_sec=sec,
            timestamp=ts,
            binance_price=binance_price,
            binance_change_from_open_pct=binance_change_pct,
            polymarket_up_price=poly_up_price,
            polymarket_trades_count=len(recent_up_trades),
            polymarket_volume=poly_volume,
            implied_up_prob=implied_up,
            price_gap=price_gap,
        ))

    return WindowAnalysis(
        window_start=window_start,
        window_end=window_end,
        resolution=resolution,
        binance_open=binance_open or 0,
        binance_close=binance_close or 0,
        snapshots=snapshots,
        total_polymarket_trades=len(up_trades),
        total_polymarket_volume=sum(float(t.get("size", 0) or 0) * float(t.get("price", 0) or 0)
                                    for t in up_trades),
    )


async def run_analysis(num_markets: int = 20, snapshot_interval: int = 30):
    """Run the full price analysis"""

    print("=" * 70)
    print("POLYMARKET VS BINANCE PRICE EVOLUTION ANALYSIS")
    print("=" * 70)
    print(f"\nAnalyzing {num_markets} recent BTC markets")
    print(f"Snapshot interval: {snapshot_interval} seconds\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate time range (last 48 hours to ensure we have resolved markets)
    end_ts = int(time.time())
    start_ts = end_ts - (48 * 3600)

    async with aiohttp.ClientSession() as session:
        # Find resolved markets
        print("Finding resolved markets...")
        markets = []
        check_ts = (end_ts // 900) * 900

        while len(markets) < num_markets and check_ts > start_ts:
            market = await fetch_market(session, check_ts)
            if market:
                markets.append({
                    "window_start": check_ts,
                    "window_end": check_ts + 900,
                    "condition_id": market.get("conditionId"),
                    "resolution": "UP" if float(json.loads(market.get("outcomePrices", "[0.5,0.5]"))[0]) > 0.5 else "DOWN",
                })
            check_ts -= 900
            await asyncio.sleep(0.1)

        print(f"Found {len(markets)} resolved markets\n")

        if not markets:
            print("No markets found!")
            return

        # Analyze each market
        analyses = []

        for i, m in enumerate(markets):
            print(f"Analyzing market {i+1}/{len(markets)}: {datetime.fromtimestamp(m['window_start']).strftime('%Y-%m-%d %H:%M')}")

            # Fetch data
            poly_trades = await fetch_polymarket_trades(session, m["condition_id"])
            binance_klines = await fetch_binance_klines(session, m["window_start"], m["window_end"])

            if not binance_klines:
                print("  No Binance data, skipping")
                continue

            analysis = analyze_window(
                m["window_start"],
                m["window_end"],
                m["resolution"],
                poly_trades,
                binance_klines,
                snapshot_interval,
            )
            analyses.append(analysis)

            await asyncio.sleep(0.2)

        # Print summary
        print("\n" + "=" * 70)
        print("PRICE EVOLUTION SUMMARY")
        print("=" * 70)

        # Aggregate by time into window
        time_stats = {}
        for sec in range(0, 901, snapshot_interval):
            time_stats[sec] = {
                "count": 0,
                "binance_changes": [],
                "poly_prices": [],
                "implied_probs": [],
                "gaps": [],
            }

        for analysis in analyses:
            for snap in analysis.snapshots:
                sec = snap.time_into_window_sec
                stats = time_stats[sec]
                stats["count"] += 1
                stats["binance_changes"].append(snap.binance_change_from_open_pct)
                stats["implied_probs"].append(snap.implied_up_prob)
                if snap.polymarket_up_price is not None:
                    stats["poly_prices"].append(snap.polymarket_up_price)
                if snap.price_gap is not None:
                    stats["gaps"].append(snap.price_gap)

        print(f"\n{'Time':<8} {'Binance Δ':<12} {'Implied UP':<12} {'Poly UP':<12} {'Gap':<12} {'Markets w/Poly'}")
        print("-" * 70)

        for sec in sorted(time_stats.keys()):
            stats = time_stats[sec]
            if stats["count"] == 0:
                continue

            avg_binance = sum(stats["binance_changes"]) / len(stats["binance_changes"]) if stats["binance_changes"] else 0
            avg_implied = sum(stats["implied_probs"]) / len(stats["implied_probs"]) if stats["implied_probs"] else 0
            avg_poly = sum(stats["poly_prices"]) / len(stats["poly_prices"]) if stats["poly_prices"] else None
            avg_gap = sum(stats["gaps"]) / len(stats["gaps"]) if stats["gaps"] else None

            poly_str = f"{avg_poly:.3f}" if avg_poly else "-"
            gap_str = f"{avg_gap*100:.1f}%" if avg_gap else "-"
            poly_count = len(stats["poly_prices"])

            print(f"{sec//60}m{sec%60:02d}s    {avg_binance:+.3f}%      {avg_implied:.3f}        {poly_str:<12} {gap_str:<12} {poly_count}/{stats['count']}")

        # Resolution analysis
        print("\n" + "=" * 70)
        print("BY RESOLUTION")
        print("=" * 70)

        for res in ["UP", "DOWN"]:
            res_analyses = [a for a in analyses if a.resolution == res]
            if not res_analyses:
                continue

            print(f"\n{res} markets ({len(res_analyses)}):")

            for sec in [180, 360, 540, 720]:  # 3, 6, 9, 12 min
                gaps = []
                for a in res_analyses:
                    for s in a.snapshots:
                        if s.time_into_window_sec == sec and s.price_gap is not None:
                            gaps.append(s.price_gap)

                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    direction = "Poly LAGGING" if avg_gap > 0 else "Poly LEADING"
                    print(f"  At {sec//60}m: Avg gap = {avg_gap*100:+.1f}% ({direction})")

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"price_evolution_{timestamp}.json"

        output_data = {
            "run_time": datetime.now().isoformat(),
            "num_markets": len(analyses),
            "snapshot_interval_sec": snapshot_interval,
            "markets": [
                {
                    "window_start": a.window_start,
                    "window_end": a.window_end,
                    "resolution": a.resolution,
                    "binance_open": a.binance_open,
                    "binance_close": a.binance_close,
                    "total_polymarket_trades": a.total_polymarket_trades,
                    "snapshots": [asdict(s) for s in a.snapshots],
                }
                for a in analyses
            ]
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Polymarket vs Binance price evolution")
    parser.add_argument("--markets", type=int, default=20, help="Number of markets to analyze")
    parser.add_argument("--interval", type=int, default=30, help="Snapshot interval in seconds")

    args = parser.parse_args()

    asyncio.run(run_analysis(args.markets, args.interval))


if __name__ == "__main__":
    main()
