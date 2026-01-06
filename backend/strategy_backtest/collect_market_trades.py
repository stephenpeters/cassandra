"""
Polymarket Trade Data Collector

Collects all gabagool22 trades for BTC 15-min markets.
Saves each market to a separate JSON file using the slug name.
Runs for 20 markets (5 hours) to build a historical dataset.

Usage: python collect_market_trades.py
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path

# Configuration
GABAGOOL_WALLET = "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d"
MARKETS_TO_COLLECT = 20
POLL_INTERVAL_SEC = 10  # How often to poll for new trades
DATA_DIR = Path(__file__).parent / "data" / "markets"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_current_market_window() -> tuple[int, int, str]:
    """Get the current 15-min market window timestamps and slug"""
    now = int(time.time())
    window_start = (now // 900) * 900
    window_end = window_start + 900
    slug = f"btc-updown-15m-{window_start}"
    return window_start, window_end, slug


def get_next_market_window() -> tuple[int, int, str]:
    """Get the next 15-min market window timestamps and slug"""
    now = int(time.time())
    current_start = (now // 900) * 900
    next_start = current_start + 900
    next_end = next_start + 900
    slug = f"btc-updown-15m-{next_start}"
    return next_start, next_end, slug


def fetch_trades_for_market(slug: str, window_start: int) -> list[dict]:
    """Fetch all trades for a specific market slug"""
    all_trades = []
    seen_ids = set()
    offset = 0
    max_offset = 100000  # Safety limit

    while offset < max_offset:
        try:
            resp = requests.get(
                "https://data-api.polymarket.com/trades",
                params={
                    "maker": GABAGOOL_WALLET,
                    "limit": 500,
                    "offset": offset,
                },
                timeout=30,
            )

            if resp.status_code != 200:
                print(f"  API error: {resp.status_code}")
                break

            trades = resp.json()
            if not trades:
                break

            # Filter for this specific market
            for t in trades:
                if t.get("eventSlug", "") == slug:
                    # Deduplicate by transaction hash + timestamp
                    trade_id = f"{t.get('transactionHash', '')}_{t.get('timestamp', '')}"
                    if trade_id not in seen_ids:
                        seen_ids.add(trade_id)
                        all_trades.append(t)

            # Check if we've gone past this market's time range
            timestamps = [t.get("timestamp", 0) for t in trades if isinstance(t.get("timestamp"), int)]
            if timestamps:
                oldest = min(timestamps)
                if oldest < window_start - 300:  # 5 min buffer
                    break

            offset += len(trades)

        except Exception as e:
            print(f"  Error fetching trades: {e}")
            break

    return all_trades


def calculate_position(trades: list[dict]) -> dict:
    """Calculate net position from trades"""
    net_up = 0.0
    net_down = 0.0
    total_volume = 0.0

    for t in trades:
        size = float(t.get("size", 0))
        price = float(t.get("price", 0))
        outcome = t.get("outcome", "")
        side = t.get("side", "")

        total_volume += size * price

        if outcome == "Up":
            if side == "BUY":
                net_up += size
            else:
                net_up -= size
        elif outcome == "Down":
            if side == "BUY":
                net_down += size
            else:
                net_down -= size

    # Determine bias
    if net_up > net_down * 1.2:
        bias = "UP"
    elif net_down > net_up * 1.2:
        bias = "DOWN"
    else:
        bias = "NEUTRAL"

    confidence = abs(net_up - net_down) / (net_up + net_down) * 100 if (net_up + net_down) > 0 else 0

    return {
        "net_up": net_up,
        "net_down": net_down,
        "total_volume_usd": total_volume,
        "bias": bias,
        "bias_confidence": confidence,
    }


def save_market_data(slug: str, window_start: int, window_end: int, trades: list[dict], position: dict):
    """Save market data to JSON file"""
    filepath = DATA_DIR / f"{slug}.json"

    # Get timestamps from trades
    timestamps = [t.get("timestamp", 0) for t in trades if isinstance(t.get("timestamp"), int)]
    first_trade = min(timestamps) if timestamps else None
    last_trade = max(timestamps) if timestamps else None

    data = {
        "slug": slug,
        "window_start": window_start,
        "window_end": window_end,
        "window_start_human": datetime.fromtimestamp(window_start).isoformat(),
        "window_end_human": datetime.fromtimestamp(window_end).isoformat(),
        "collected_at": int(time.time()),
        "total_trades": len(trades),
        "first_trade_ts": first_trade,
        "last_trade_ts": last_trade,
        **position,
        "trades": trades,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def collect_single_market(slug: str, window_start: int, window_end: int) -> dict:
    """Collect all trades for a single market from start to end"""
    print(f"\n{'='*60}")
    print(f"Collecting: {slug}")
    print(f"Window: {datetime.fromtimestamp(window_start)} to {datetime.fromtimestamp(window_end)}")
    print("="*60)

    all_trades = []
    seen_ids = set()
    last_save = time.time()

    # Wait for market to start
    now = time.time()
    if now < window_start:
        wait_time = window_start - now
        print(f"Waiting {wait_time:.0f}s for market to start...")
        time.sleep(wait_time)

    # Collect trades throughout the market
    while time.time() < window_end + 60:  # 60s buffer after market end
        trades = fetch_trades_for_market(slug, window_start)

        # Add new trades
        new_count = 0
        for t in trades:
            trade_id = f"{t.get('transactionHash', '')}_{t.get('timestamp', '')}"
            if trade_id not in seen_ids:
                seen_ids.add(trade_id)
                all_trades.append(t)
                new_count += 1

        remaining = window_end - time.time()
        position = calculate_position(all_trades)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Trades: {len(all_trades):,} (+{new_count}) | "
              f"Bias: {position['bias']} | "
              f"Remaining: {max(0, remaining):.0f}s")

        # Save checkpoint every 2 minutes
        if time.time() - last_save > 120:
            save_market_data(slug, window_start, window_end, all_trades, position)
            print(f"  [Checkpoint saved: {len(all_trades)} trades]")
            last_save = time.time()

        # Exit if market ended
        if remaining < -60:
            break

        time.sleep(POLL_INTERVAL_SEC)

    # Final collection and save
    trades = fetch_trades_for_market(slug, window_start)
    for t in trades:
        trade_id = f"{t.get('transactionHash', '')}_{t.get('timestamp', '')}"
        if trade_id not in seen_ids:
            seen_ids.add(trade_id)
            all_trades.append(t)

    position = calculate_position(all_trades)
    filepath = save_market_data(slug, window_start, window_end, all_trades, position)

    print(f"\n--- Market Complete ---")
    print(f"Total trades: {len(all_trades):,}")
    print(f"Net UP: {position['net_up']:,.2f}")
    print(f"Net DOWN: {position['net_down']:,.2f}")
    print(f"Bias: {position['bias']} ({position['bias_confidence']:.1f}%)")
    print(f"Saved to: {filepath}")

    return {
        "slug": slug,
        "trades": len(all_trades),
        "bias": position["bias"],
        "net_up": position["net_up"],
        "net_down": position["net_down"],
    }


def main():
    print("="*60)
    print("POLYMARKET TRADE DATA COLLECTOR")
    print("="*60)
    print(f"Whale: gabagool22 ({GABAGOOL_WALLET})")
    print(f"Markets to collect: {MARKETS_TO_COLLECT}")
    print(f"Data directory: {DATA_DIR}")
    print()

    collected_markets = []

    # Check for existing market files
    existing = list(DATA_DIR.glob("btc-updown-15m-*.json"))
    print(f"Existing market files: {len(existing)}")

    for i in range(MARKETS_TO_COLLECT):
        # Get next market window
        next_start, next_end, next_slug = get_next_market_window()

        # Skip if we already have this market
        if (DATA_DIR / f"{next_slug}.json").exists():
            print(f"Skipping {next_slug} (already collected)")
            time.sleep(900)  # Wait for next market
            continue

        print(f"\n[Market {i+1}/{MARKETS_TO_COLLECT}]")

        result = collect_single_market(next_slug, next_start, next_end)
        collected_markets.append(result)

        # Summary
        print(f"\n{'='*60}")
        print(f"PROGRESS: {len(collected_markets)}/{MARKETS_TO_COLLECT} markets collected")
        print("="*60)
        for m in collected_markets:
            print(f"  {m['slug']}: {m['trades']:,} trades, bias={m['bias']}")

    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Total markets: {len(collected_markets)}")
    print(f"Data saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
