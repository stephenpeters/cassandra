#!/usr/bin/env python3
"""
Historical Data Collector for Polymarket 15-Minute Crypto Markets

Collects:
1. Polymarket market outcomes and trades for BTC, ETH, SOL
2. Binance trades during the same 15-minute windows

Usage:
    python collect_historical_data.py --hours 24 --symbols BTC,ETH,SOL
    python collect_historical_data.py --start 2026-01-01 --end 2026-01-03
"""

import argparse
import asyncio
import aiohttp
import json
import csv
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
BINANCE_API = "https://api.binance.com"

SYMBOLS = ["BTC", "ETH", "SOL"]

# Output directory
OUTPUT_DIR = Path("historical_data")


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class MarketWindow:
    """A 15-minute market window with its outcome"""
    symbol: str
    window_start: int  # Unix timestamp
    window_end: int
    slug: str
    condition_id: str
    outcome: str  # "UP" or "DOWN"
    up_price_final: float  # Final price (0 or 1)
    down_price_final: float
    volume: float
    liquidity: float
    total_trades: int
    unique_traders: int


@dataclass
class PolymarketTrade:
    """A trade on Polymarket"""
    symbol: str
    window_start: int
    condition_id: str
    timestamp: int
    side: str  # BUY or SELL
    outcome: str  # Up or Down
    size: float
    price: float
    usd_value: float
    trader_wallet: str
    trader_name: str
    tx_hash: str


@dataclass
class BinanceTrade:
    """A trade on Binance"""
    symbol: str
    window_start: int
    timestamp: int
    price: float
    quantity: float
    quote_quantity: float
    is_buyer_maker: bool
    trade_id: int


# ============================================================================
# DATA FETCHERS
# ============================================================================

async def fetch_market_data(
    session: aiohttp.ClientSession,
    symbol: str,
    window_timestamp: int
) -> Optional[dict]:
    """Fetch market data from Polymarket Gamma API"""
    slug = f"{symbol.lower()}-updown-15m-{window_timestamp}"
    url = f"{GAMMA_API}/markets"
    params = {"slug": slug}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with session.get(url, params=params, headers=headers, timeout=10) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data[0] if data else None
    except Exception as e:
        print(f"  Error fetching market {slug}: {e}")
        return None


async def fetch_polymarket_trades(
    session: aiohttp.ClientSession,
    condition_id: str,
    limit: int = 500
) -> list[dict]:
    """Fetch trades from Polymarket Data API"""
    url = f"{DATA_API}/trades"
    params = {"market": condition_id, "limit": limit}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with session.get(url, params=params, headers=headers, timeout=15) as resp:
            if resp.status != 200:
                return []
            return await resp.json()
    except Exception as e:
        print(f"  Error fetching trades: {e}")
        return []


async def fetch_binance_trades(
    session: aiohttp.ClientSession,
    symbol: str,
    start_time: int,
    end_time: int,
    limit: int = 1000
) -> list[dict]:
    """Fetch trades from Binance API for a specific time window"""
    binance_symbol = f"{symbol}USDT"
    url = f"{BINANCE_API}/api/v3/aggTrades"

    all_trades = []
    current_start = start_time * 1000  # Convert to milliseconds
    end_ms = end_time * 1000

    while current_start < end_ms:
        params = {
            "symbol": binance_symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit
        }

        try:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    break
                trades = await resp.json()
                if not trades:
                    break

                all_trades.extend(trades)

                # Move to next batch
                if len(trades) < limit:
                    break
                current_start = trades[-1]["T"] + 1

                # Rate limiting
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"  Error fetching Binance trades: {e}")
            break

    return all_trades


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_market_window(
    symbol: str,
    window_timestamp: int,
    market_data: dict,
    trades: list[dict]
) -> MarketWindow:
    """Process market data into a MarketWindow object"""
    # Parse outcome prices
    outcome_prices = market_data.get("outcomePrices", "[0.5, 0.5]")
    if isinstance(outcome_prices, str):
        prices = json.loads(outcome_prices)
    else:
        prices = outcome_prices

    up_price = float(prices[0]) if prices else 0.5
    down_price = float(prices[1]) if len(prices) > 1 else 0.5

    # Determine outcome
    outcome = "UP" if up_price > 0.5 else "DOWN"

    # Count unique traders
    unique_traders = len(set(t.get("proxyWallet", "") for t in trades))

    return MarketWindow(
        symbol=symbol,
        window_start=window_timestamp,
        window_end=window_timestamp + 900,
        slug=market_data.get("slug", ""),
        condition_id=market_data.get("conditionId", ""),
        outcome=outcome,
        up_price_final=up_price,
        down_price_final=down_price,
        volume=float(market_data.get("volume", 0) or 0),
        liquidity=float(market_data.get("liquidity", 0) or 0),
        total_trades=len(trades),
        unique_traders=unique_traders
    )


def process_polymarket_trades(
    symbol: str,
    window_timestamp: int,
    condition_id: str,
    trades: list[dict]
) -> list[PolymarketTrade]:
    """Process raw trades into PolymarketTrade objects"""
    processed = []
    for t in trades:
        try:
            processed.append(PolymarketTrade(
                symbol=symbol,
                window_start=window_timestamp,
                condition_id=condition_id,
                timestamp=int(t.get("timestamp", 0)),
                side=t.get("side", ""),
                outcome=t.get("outcome", ""),
                size=float(t.get("size", 0) or 0),
                price=float(t.get("price", 0) or 0),
                usd_value=float(t.get("size", 0) or 0) * float(t.get("price", 0) or 0),
                trader_wallet=t.get("proxyWallet", ""),
                trader_name=t.get("name", "") or t.get("pseudonym", ""),
                tx_hash=t.get("transactionHash", "")
            ))
        except Exception as e:
            continue
    return processed


def process_binance_trades(
    symbol: str,
    window_timestamp: int,
    trades: list[dict]
) -> list[BinanceTrade]:
    """Process raw Binance trades into BinanceTrade objects"""
    processed = []
    for t in trades:
        try:
            processed.append(BinanceTrade(
                symbol=symbol,
                window_start=window_timestamp,
                timestamp=int(t.get("T", 0)),
                price=float(t.get("p", 0)),
                quantity=float(t.get("q", 0)),
                quote_quantity=float(t.get("p", 0)) * float(t.get("q", 0)),
                is_buyer_maker=t.get("m", False),
                trade_id=int(t.get("a", 0))
            ))
        except Exception:
            continue
    return processed


# ============================================================================
# DATA COLLECTION
# ============================================================================

async def collect_window_data(
    session: aiohttp.ClientSession,
    symbol: str,
    window_timestamp: int
) -> tuple[Optional[MarketWindow], list[PolymarketTrade], list[BinanceTrade]]:
    """Collect all data for a single 15-minute window"""

    # Fetch Polymarket market data
    market_data = await fetch_market_data(session, symbol, window_timestamp)
    if not market_data:
        return None, [], []

    # Check if market is closed (resolved)
    if not market_data.get("closed", False):
        return None, [], []

    condition_id = market_data.get("conditionId", "")

    # Fetch Polymarket trades
    poly_trades_raw = await fetch_polymarket_trades(session, condition_id)

    # Fetch Binance trades for the same window
    binance_trades_raw = await fetch_binance_trades(
        session, symbol, window_timestamp, window_timestamp + 900
    )

    # Process data
    market_window = process_market_window(symbol, window_timestamp, market_data, poly_trades_raw)
    poly_trades = process_polymarket_trades(symbol, window_timestamp, condition_id, poly_trades_raw)
    binance_trades = process_binance_trades(symbol, window_timestamp, binance_trades_raw)

    return market_window, poly_trades, binance_trades


async def collect_historical_data(
    symbols: list[str],
    start_timestamp: int,
    end_timestamp: int,
    output_dir: Path
):
    """Collect historical data for multiple symbols over a time range"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate windows to fetch
    current_window = (start_timestamp // 900) * 900
    end_window = (end_timestamp // 900) * 900

    total_windows = (end_window - current_window) // 900
    print(f"Collecting data for {len(symbols)} symbols over {total_windows} windows")
    print(f"Start: {datetime.fromtimestamp(current_window)}")
    print(f"End: {datetime.fromtimestamp(end_window)}")

    # Data storage
    all_market_windows: list[MarketWindow] = []
    all_poly_trades: list[PolymarketTrade] = []
    all_binance_trades: list[BinanceTrade] = []

    async with aiohttp.ClientSession() as session:
        window_count = 0

        while current_window < end_window:
            window_count += 1
            window_time = datetime.fromtimestamp(current_window)

            for symbol in symbols:
                print(f"[{window_count}/{total_windows}] {symbol} @ {window_time.strftime('%Y-%m-%d %H:%M')}", end="")

                market, poly_trades, binance_trades = await collect_window_data(
                    session, symbol, current_window
                )

                if market:
                    all_market_windows.append(market)
                    all_poly_trades.extend(poly_trades)
                    all_binance_trades.extend(binance_trades)
                    print(f" - {market.outcome} | {market.total_trades} poly trades | {len(binance_trades)} binance trades")
                else:
                    print(" - not resolved yet")

                # Rate limiting
                await asyncio.sleep(0.2)

            current_window += 900

    # Save data
    print(f"\nSaving data...")
    save_data(output_dir, all_market_windows, all_poly_trades, all_binance_trades)

    print(f"\nCollection complete!")
    print(f"  Market windows: {len(all_market_windows)}")
    print(f"  Polymarket trades: {len(all_poly_trades)}")
    print(f"  Binance trades: {len(all_binance_trades)}")


# ============================================================================
# DATA SAVING
# ============================================================================

def save_data(
    output_dir: Path,
    market_windows: list[MarketWindow],
    poly_trades: list[PolymarketTrade],
    binance_trades: list[BinanceTrade]
):
    """Save collected data to CSV and JSON files"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save market windows
    if market_windows:
        csv_path = output_dir / f"market_windows_{timestamp}.csv"
        json_path = output_dir / f"market_windows_{timestamp}.json"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(market_windows[0]).keys())
            writer.writeheader()
            for mw in market_windows:
                writer.writerow(asdict(mw))

        with open(json_path, "w") as f:
            json.dump([asdict(mw) for mw in market_windows], f, indent=2)

        print(f"  Saved market windows to {csv_path}")

    # Save Polymarket trades
    if poly_trades:
        csv_path = output_dir / f"polymarket_trades_{timestamp}.csv"
        json_path = output_dir / f"polymarket_trades_{timestamp}.json"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(poly_trades[0]).keys())
            writer.writeheader()
            for t in poly_trades:
                writer.writerow(asdict(t))

        with open(json_path, "w") as f:
            json.dump([asdict(t) for t in poly_trades], f, indent=2)

        print(f"  Saved Polymarket trades to {csv_path}")

    # Save Binance trades
    if binance_trades:
        csv_path = output_dir / f"binance_trades_{timestamp}.csv"
        json_path = output_dir / f"binance_trades_{timestamp}.json"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(binance_trades[0]).keys())
            writer.writeheader()
            for t in binance_trades:
                writer.writerow(asdict(t))

        with open(json_path, "w") as f:
            json.dump([asdict(t) for t in binance_trades], f, indent=2)

        print(f"  Saved Binance trades to {csv_path}")

    # Save summary
    summary = {
        "collection_time": datetime.now().isoformat(),
        "symbols": list(set(mw.symbol for mw in market_windows)),
        "total_windows": len(market_windows),
        "total_polymarket_trades": len(poly_trades),
        "total_binance_trades": len(binance_trades),
        "date_range": {
            "start": datetime.fromtimestamp(min(mw.window_start for mw in market_windows)).isoformat() if market_windows else None,
            "end": datetime.fromtimestamp(max(mw.window_end for mw in market_windows)).isoformat() if market_windows else None,
        },
        "by_symbol": {}
    }

    for symbol in summary["symbols"]:
        symbol_windows = [mw for mw in market_windows if mw.symbol == symbol]
        up_wins = sum(1 for mw in symbol_windows if mw.outcome == "UP")
        down_wins = len(symbol_windows) - up_wins

        summary["by_symbol"][symbol] = {
            "windows": len(symbol_windows),
            "up_wins": up_wins,
            "down_wins": down_wins,
            "up_rate": up_wins / len(symbol_windows) if symbol_windows else 0,
            "total_volume": sum(mw.volume for mw in symbol_windows),
            "avg_trades_per_window": sum(mw.total_trades for mw in symbol_windows) / len(symbol_windows) if symbol_windows else 0,
        }

    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved summary to {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect historical Polymarket and Binance data")
    parser.add_argument("--hours", type=int, default=24, help="Hours of history to collect")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM)")
    parser.add_argument("--symbols", type=str, default="BTC,ETH,SOL", help="Comma-separated symbols")
    parser.add_argument("--output", type=str, default="historical_data", help="Output directory")

    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Parse time range
    if args.start and args.end:
        # Parse dates
        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d"]:
            try:
                start_dt = datetime.strptime(args.start, fmt)
                break
            except ValueError:
                continue
        else:
            print(f"Invalid start date format: {args.start}")
            return

        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d"]:
            try:
                end_dt = datetime.strptime(args.end, fmt)
                break
            except ValueError:
                continue
        else:
            print(f"Invalid end date format: {args.end}")
            return

        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())
    else:
        # Use hours from now
        end_timestamp = int(time.time())
        start_timestamp = end_timestamp - (args.hours * 3600)

    output_dir = Path(args.output)

    print("=" * 60)
    print("POLYMARKET + BINANCE HISTORICAL DATA COLLECTOR")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Output: {output_dir}")
    print()

    asyncio.run(collect_historical_data(symbols, start_timestamp, end_timestamp, output_dir))


if __name__ == "__main__":
    main()
