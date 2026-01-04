#!/usr/bin/env python3
"""
Historical Data Analyzer for Polymarket 15-Minute Crypto Markets

Analyzes:
1. Trader behavior patterns - who trades, when, what sizes
2. Price momentum correlations - Binance price movements vs market outcomes
3. Volume patterns - buy/sell imbalances, timing patterns
4. Whale activity - large traders and their win rates

Usage:
    python analyze_historical_data.py --data historical_data
    python analyze_historical_data.py --data historical_data --symbol BTC
"""

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import statistics


# ============================================================================
# DATA LOADING
# ============================================================================

def load_csv(filepath: Path) -> list[dict]:
    """Load CSV file into list of dicts"""
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def find_latest_files(data_dir: Path) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find the most recent data files in the directory"""
    market_files = list(data_dir.glob("market_windows_*.csv"))
    poly_files = list(data_dir.glob("polymarket_trades_*.csv"))
    binance_files = list(data_dir.glob("binance_trades_*.csv"))

    market_file = max(market_files, key=lambda p: p.stat().st_mtime) if market_files else None
    poly_file = max(poly_files, key=lambda p: p.stat().st_mtime) if poly_files else None
    binance_file = max(binance_files, key=lambda p: p.stat().st_mtime) if binance_files else None

    return market_file, poly_file, binance_file


# ============================================================================
# ANALYSIS CLASSES
# ============================================================================

@dataclass
class TraderStats:
    """Statistics for a single trader"""
    wallet: str
    name: str
    total_trades: int = 0
    total_volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    up_trades: int = 0
    down_trades: int = 0
    wins: int = 0
    losses: int = 0
    symbols_traded: set = None
    windows_traded: set = None
    avg_trade_size: float = 0.0

    def __post_init__(self):
        if self.symbols_traded is None:
            self.symbols_traded = set()
        if self.windows_traded is None:
            self.windows_traded = set()

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def up_bias(self) -> float:
        total = self.up_trades + self.down_trades
        return self.up_trades / total if total > 0 else 0.5


@dataclass
class WindowStats:
    """Statistics for a single 15-minute window"""
    symbol: str
    window_start: int
    outcome: str
    total_trades: int = 0
    up_volume: float = 0.0
    down_volume: float = 0.0
    unique_traders: int = 0
    binance_open: float = 0.0
    binance_close: float = 0.0
    binance_high: float = 0.0
    binance_low: float = 0.0
    binance_volume: float = 0.0
    binance_buy_volume: float = 0.0
    binance_sell_volume: float = 0.0
    binance_trade_count: int = 0

    @property
    def volume_imbalance(self) -> float:
        """Positive = more UP volume, negative = more DOWN volume"""
        total = self.up_volume + self.down_volume
        if total == 0:
            return 0.0
        return (self.up_volume - self.down_volume) / total

    @property
    def binance_price_change(self) -> float:
        if self.binance_open == 0:
            return 0.0
        return (self.binance_close - self.binance_open) / self.binance_open * 100

    @property
    def binance_buy_pressure(self) -> float:
        """Positive = more buying, negative = more selling"""
        total = self.binance_buy_volume + self.binance_sell_volume
        if total == 0:
            return 0.0
        return (self.binance_buy_volume - self.binance_sell_volume) / total


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_traders(
    poly_trades: list[dict],
    market_windows: list[dict]
) -> dict[str, TraderStats]:
    """Analyze trader behavior patterns"""

    # Create outcome lookup
    outcomes = {}
    for mw in market_windows:
        key = (mw["symbol"], int(mw["window_start"]))
        outcomes[key] = mw["outcome"]

    # Aggregate trader stats
    traders: dict[str, TraderStats] = {}

    for trade in poly_trades:
        wallet = trade["trader_wallet"]
        if not wallet:
            continue

        if wallet not in traders:
            traders[wallet] = TraderStats(
                wallet=wallet,
                name=trade.get("trader_name", "") or "Anonymous"
            )

        t = traders[wallet]
        t.total_trades += 1
        size = float(trade.get("size", 0) or 0)
        price = float(trade.get("price", 0) or 0)
        usd_value = size * price
        t.total_volume += usd_value

        side = trade.get("side", "")
        if side == "BUY":
            t.buy_volume += usd_value
        else:
            t.sell_volume += usd_value

        outcome = trade.get("outcome", "")
        if "Up" in outcome or "UP" in outcome:
            t.up_trades += 1
        elif "Down" in outcome or "DOWN" in outcome:
            t.down_trades += 1

        t.symbols_traded.add(trade["symbol"])
        t.windows_traded.add((trade["symbol"], int(trade["window_start"])))

        # Check if trade was correct
        window_key = (trade["symbol"], int(trade["window_start"]))
        market_outcome = outcomes.get(window_key, "")

        if side == "BUY":
            # Bought outcome, wins if outcome matches market result
            trade_outcome = "UP" if "Up" in outcome else "DOWN"
            if trade_outcome == market_outcome:
                t.wins += 1
            else:
                t.losses += 1

    # Calculate averages
    for t in traders.values():
        if t.total_trades > 0:
            t.avg_trade_size = t.total_volume / t.total_trades

    return traders


def analyze_windows(
    market_windows: list[dict],
    poly_trades: list[dict],
    binance_trades: list[dict]
) -> list[WindowStats]:
    """Analyze each 15-minute window"""

    # Group poly trades by window
    poly_by_window = defaultdict(list)
    for t in poly_trades:
        key = (t["symbol"], int(t["window_start"]))
        poly_by_window[key].append(t)

    # Group binance trades by window
    binance_by_window = defaultdict(list)
    for t in binance_trades:
        key = (t["symbol"], int(t["window_start"]))
        binance_by_window[key].append(t)

    # Analyze each window
    windows = []
    for mw in market_windows:
        key = (mw["symbol"], int(mw["window_start"]))

        ws = WindowStats(
            symbol=mw["symbol"],
            window_start=int(mw["window_start"]),
            outcome=mw["outcome"]
        )

        # Polymarket trade analysis
        poly = poly_by_window.get(key, [])
        ws.total_trades = len(poly)
        ws.unique_traders = len(set(t["trader_wallet"] for t in poly if t["trader_wallet"]))

        for t in poly:
            size = float(t.get("size", 0) or 0)
            price = float(t.get("price", 0) or 0)
            usd = size * price
            outcome = t.get("outcome", "")
            if "Up" in outcome or "UP" in outcome:
                ws.up_volume += usd
            else:
                ws.down_volume += usd

        # Binance trade analysis
        binance = binance_by_window.get(key, [])
        ws.binance_trade_count = len(binance)

        if binance:
            prices = [float(t["price"]) for t in binance]
            quantities = [float(t["quantity"]) for t in binance]

            # Sort by timestamp to get open/close
            sorted_binance = sorted(binance, key=lambda x: int(x["timestamp"]))
            ws.binance_open = float(sorted_binance[0]["price"])
            ws.binance_close = float(sorted_binance[-1]["price"])
            ws.binance_high = max(prices)
            ws.binance_low = min(prices)
            ws.binance_volume = sum(float(t["quote_quantity"]) for t in binance)

            for t in binance:
                quote_vol = float(t["quote_quantity"])
                is_buyer_maker = t["is_buyer_maker"] in [True, "True", "true", "1"]
                if is_buyer_maker:
                    ws.binance_sell_volume += quote_vol
                else:
                    ws.binance_buy_volume += quote_vol

        windows.append(ws)

    return windows


def analyze_momentum_correlation(windows: list[WindowStats]) -> dict:
    """Analyze correlation between Binance momentum and market outcomes"""

    results = {
        "total_windows": len(windows),
        "up_wins": 0,
        "down_wins": 0,
        "price_predicted_outcome": 0,  # Binance price direction matched outcome
        "volume_predicted_outcome": 0,  # Polymarket volume imbalance matched outcome
        "buy_pressure_predicted_outcome": 0,  # Binance buy pressure matched outcome
        "by_symbol": {}
    }

    for ws in windows:
        if ws.outcome == "UP":
            results["up_wins"] += 1
        else:
            results["down_wins"] += 1

        # Check if Binance price direction predicted outcome
        price_up = ws.binance_price_change > 0
        outcome_up = ws.outcome == "UP"
        if price_up == outcome_up:
            results["price_predicted_outcome"] += 1

        # Check if Polymarket volume imbalance predicted outcome
        volume_up = ws.volume_imbalance > 0
        if volume_up == outcome_up:
            results["volume_predicted_outcome"] += 1

        # Check if Binance buy pressure predicted outcome
        buy_pressure_up = ws.binance_buy_pressure > 0
        if buy_pressure_up == outcome_up:
            results["buy_pressure_predicted_outcome"] += 1

    # Calculate prediction rates
    n = len(windows) if windows else 1
    results["price_prediction_rate"] = results["price_predicted_outcome"] / n
    results["volume_prediction_rate"] = results["volume_predicted_outcome"] / n
    results["buy_pressure_prediction_rate"] = results["buy_pressure_predicted_outcome"] / n

    # By symbol analysis
    by_symbol = defaultdict(lambda: {
        "windows": 0, "up_wins": 0, "price_matches": 0,
        "volume_matches": 0, "buy_pressure_matches": 0
    })

    for ws in windows:
        s = by_symbol[ws.symbol]
        s["windows"] += 1
        if ws.outcome == "UP":
            s["up_wins"] += 1

        outcome_up = ws.outcome == "UP"
        if (ws.binance_price_change > 0) == outcome_up:
            s["price_matches"] += 1
        if (ws.volume_imbalance > 0) == outcome_up:
            s["volume_matches"] += 1
        if (ws.binance_buy_pressure > 0) == outcome_up:
            s["buy_pressure_matches"] += 1

    for symbol, s in by_symbol.items():
        n = s["windows"] if s["windows"] else 1
        results["by_symbol"][symbol] = {
            "windows": s["windows"],
            "up_rate": s["up_wins"] / n,
            "price_prediction_rate": s["price_matches"] / n,
            "volume_prediction_rate": s["volume_matches"] / n,
            "buy_pressure_prediction_rate": s["buy_pressure_matches"] / n
        }

    return results


def find_whales(traders: dict[str, TraderStats], min_volume: float = 10000) -> list[TraderStats]:
    """Find high-volume traders (whales)"""
    whales = [t for t in traders.values() if t.total_volume >= min_volume]
    return sorted(whales, key=lambda x: x.total_volume, reverse=True)


def find_smart_money(traders: dict[str, TraderStats], min_trades: int = 10) -> list[TraderStats]:
    """Find traders with high win rates (potential smart money)"""
    eligible = [t for t in traders.values() if (t.wins + t.losses) >= min_trades]
    return sorted(eligible, key=lambda x: x.win_rate, reverse=True)


# ============================================================================
# REPORTING
# ============================================================================

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def format_usd(value: float) -> str:
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.2f}"


def generate_report(
    market_windows: list[dict],
    traders: dict[str, TraderStats],
    windows: list[WindowStats],
    momentum: dict,
    symbol_filter: Optional[str] = None
):
    """Generate analysis report"""

    title = f"POLYMARKET 15-MIN MARKET ANALYSIS"
    if symbol_filter:
        title += f" - {symbol_filter}"

    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

    # Filter data if symbol specified
    if symbol_filter:
        windows = [w for w in windows if w.symbol == symbol_filter]
        traders = {k: v for k, v in traders.items() if symbol_filter in v.symbols_traded}

    # === OVERVIEW ===
    print_section("OVERVIEW")
    print(f"Total windows analyzed: {len(windows)}")
    print(f"Total unique traders: {len(traders)}")
    print(f"Total trading volume: {format_usd(sum(t.total_volume for t in traders.values()))}")

    up_wins = sum(1 for w in windows if w.outcome == "UP")
    down_wins = len(windows) - up_wins
    print(f"\nOutcome distribution:")
    print(f"  UP wins:   {up_wins} ({up_wins/len(windows)*100:.1f}%)" if windows else "")
    print(f"  DOWN wins: {down_wins} ({down_wins/len(windows)*100:.1f}%)" if windows else "")

    # === MOMENTUM ANALYSIS ===
    print_section("MOMENTUM PREDICTION RATES")
    print(f"\nDo these factors predict the correct outcome?")
    print(f"  Binance price direction:     {momentum['price_prediction_rate']*100:.1f}%")
    print(f"  Polymarket volume imbalance: {momentum['volume_prediction_rate']*100:.1f}%")
    print(f"  Binance buy pressure:        {momentum['buy_pressure_prediction_rate']*100:.1f}%")

    print(f"\nBy symbol:")
    for symbol, stats in sorted(momentum["by_symbol"].items()):
        print(f"  {symbol}:")
        print(f"    Windows: {stats['windows']} | UP rate: {stats['up_rate']*100:.1f}%")
        print(f"    Price prediction: {stats['price_prediction_rate']*100:.1f}%")
        print(f"    Volume prediction: {stats['volume_prediction_rate']*100:.1f}%")

    # === TOP TRADERS (WHALES) ===
    print_section("TOP TRADERS BY VOLUME")
    whales = find_whales(traders, min_volume=1000)[:15]

    if whales:
        print(f"\n{'Rank':<5} {'Name':<20} {'Volume':>12} {'Trades':>8} {'Win Rate':>10} {'UP Bias':>9}")
        print("-" * 70)
        for i, t in enumerate(whales, 1):
            name = t.name[:18] if t.name else t.wallet[:8] + "..."
            print(f"{i:<5} {name:<20} {format_usd(t.total_volume):>12} {t.total_trades:>8} {t.win_rate*100:>9.1f}% {t.up_bias*100:>8.1f}%")
    else:
        print("  No traders with volume >= $1,000")

    # === SMART MONEY ===
    print_section("SMART MONEY (High Win Rate Traders)")
    smart = find_smart_money(traders, min_trades=5)[:10]

    if smart:
        print(f"\n{'Rank':<5} {'Name':<20} {'Win Rate':>10} {'Wins':>6} {'Losses':>8} {'Volume':>12}")
        print("-" * 70)
        for i, t in enumerate(smart, 1):
            name = t.name[:18] if t.name else t.wallet[:8] + "..."
            print(f"{i:<5} {name:<20} {t.win_rate*100:>9.1f}% {t.wins:>6} {t.losses:>8} {format_usd(t.total_volume):>12}")
    else:
        print("  No traders with >= 5 trades")

    # === WORST PERFORMERS ===
    print_section("WORST PERFORMERS (Potential Fade Candidates)")
    losers = [t for t in traders.values() if (t.wins + t.losses) >= 5]
    losers = sorted(losers, key=lambda x: x.win_rate)[:10]

    if losers:
        print(f"\n{'Rank':<5} {'Name':<20} {'Win Rate':>10} {'Wins':>6} {'Losses':>8} {'Volume':>12}")
        print("-" * 70)
        for i, t in enumerate(losers, 1):
            name = t.name[:18] if t.name else t.wallet[:8] + "..."
            print(f"{i:<5} {name:<20} {t.win_rate*100:>9.1f}% {t.wins:>6} {t.losses:>8} {format_usd(t.total_volume):>12}")
    else:
        print("  No traders with >= 5 trades")

    # === TIMING PATTERNS ===
    print_section("TRADING PATTERNS")

    # Volume by hour of day
    volume_by_hour = defaultdict(float)
    trades_by_hour = defaultdict(int)
    for w in windows:
        hour = datetime.fromtimestamp(w.window_start).hour
        volume_by_hour[hour] += w.up_volume + w.down_volume
        trades_by_hour[hour] += w.total_trades

    print("\nVolume by hour (UTC):")
    sorted_hours = sorted(volume_by_hour.items(), key=lambda x: x[1], reverse=True)[:6]
    for hour, vol in sorted_hours:
        print(f"  {hour:02d}:00 - {format_usd(vol)}")

    # === WINDOW ANALYSIS ===
    print_section("WINDOW STATISTICS")

    if windows:
        avg_traders = statistics.mean(w.unique_traders for w in windows)
        avg_poly_vol = statistics.mean(w.up_volume + w.down_volume for w in windows)
        avg_binance_vol = statistics.mean(w.binance_volume for w in windows)
        avg_binance_trades = statistics.mean(w.binance_trade_count for w in windows)

        print(f"\nAverages per 15-min window:")
        print(f"  Unique traders:      {avg_traders:.1f}")
        print(f"  Polymarket volume:   {format_usd(avg_poly_vol)}")
        print(f"  Binance volume:      {format_usd(avg_binance_vol)}")
        print(f"  Binance trades:      {avg_binance_trades:.0f}")

        # Windows with highest volume
        print("\nHighest volume windows:")
        sorted_windows = sorted(windows, key=lambda x: x.up_volume + x.down_volume, reverse=True)[:5]
        for w in sorted_windows:
            ts = datetime.fromtimestamp(w.window_start).strftime("%Y-%m-%d %H:%M")
            vol = w.up_volume + w.down_volume
            print(f"  {w.symbol} @ {ts} - {format_usd(vol)} ({w.outcome})")

    # === ACTIONABLE INSIGHTS ===
    print_section("ACTIONABLE INSIGHTS")

    print("\n1. MOMENTUM SIGNALS")
    best_predictor = max(
        [("Price direction", momentum["price_prediction_rate"]),
         ("Volume imbalance", momentum["volume_prediction_rate"]),
         ("Buy pressure", momentum["buy_pressure_prediction_rate"])],
        key=lambda x: x[1]
    )
    print(f"   Best predictor: {best_predictor[0]} ({best_predictor[1]*100:.1f}% accuracy)")

    print("\n2. FOLLOW/FADE CANDIDATES")
    if smart:
        best_trader = smart[0]
        print(f"   Follow: {best_trader.name or best_trader.wallet[:10]} (Win rate: {best_trader.win_rate*100:.1f}%)")
    if losers:
        worst_trader = losers[0]
        print(f"   Fade:   {worst_trader.name or worst_trader.wallet[:10]} (Win rate: {worst_trader.win_rate*100:.1f}%)")

    print("\n3. MARKET BIAS")
    if windows:
        up_rate = up_wins / len(windows)
        if up_rate > 0.55:
            print(f"   Market has UP bias ({up_rate*100:.1f}%) - consider bias in signals")
        elif up_rate < 0.45:
            print(f"   Market has DOWN bias ({up_rate*100:.1f}%) - consider bias in signals")
        else:
            print(f"   Market is balanced ({up_rate*100:.1f}% UP)")

    print("\n" + "=" * 60)
    print(" END OF REPORT")
    print("=" * 60 + "\n")


def save_analysis(
    output_path: Path,
    traders: dict[str, TraderStats],
    windows: list[WindowStats],
    momentum: dict
):
    """Save analysis results to JSON"""

    # Convert to serializable format
    traders_data = []
    for t in traders.values():
        traders_data.append({
            "wallet": t.wallet,
            "name": t.name,
            "total_trades": t.total_trades,
            "total_volume": t.total_volume,
            "buy_volume": t.buy_volume,
            "sell_volume": t.sell_volume,
            "up_trades": t.up_trades,
            "down_trades": t.down_trades,
            "wins": t.wins,
            "losses": t.losses,
            "win_rate": t.win_rate,
            "up_bias": t.up_bias,
            "avg_trade_size": t.avg_trade_size,
            "symbols_traded": list(t.symbols_traded),
            "windows_traded": len(t.windows_traded)
        })

    windows_data = []
    for w in windows:
        windows_data.append({
            "symbol": w.symbol,
            "window_start": w.window_start,
            "outcome": w.outcome,
            "total_trades": w.total_trades,
            "up_volume": w.up_volume,
            "down_volume": w.down_volume,
            "volume_imbalance": w.volume_imbalance,
            "unique_traders": w.unique_traders,
            "binance_open": w.binance_open,
            "binance_close": w.binance_close,
            "binance_price_change_pct": w.binance_price_change,
            "binance_buy_pressure": w.binance_buy_pressure,
            "binance_volume": w.binance_volume,
            "binance_trade_count": w.binance_trade_count
        })

    output = {
        "analysis_time": datetime.now().isoformat(),
        "momentum_analysis": momentum,
        "traders": sorted(traders_data, key=lambda x: x["total_volume"], reverse=True),
        "windows": windows_data
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved analysis to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze historical Polymarket data")
    parser.add_argument("--data", type=str, default="historical_data", help="Data directory")
    parser.add_argument("--symbol", type=str, help="Filter by symbol (BTC, ETH, SOL)")
    parser.add_argument("--output", type=str, help="Output JSON file for analysis results")

    args = parser.parse_args()
    data_dir = Path(args.data)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    # Find latest data files
    market_file, poly_file, binance_file = find_latest_files(data_dir)

    if not all([market_file, poly_file, binance_file]):
        print("Could not find all required data files in", data_dir)
        print("  Market windows:", market_file)
        print("  Polymarket trades:", poly_file)
        print("  Binance trades:", binance_file)
        return

    print(f"Loading data from {data_dir}...")
    print(f"  Market windows: {market_file.name}")
    print(f"  Polymarket trades: {poly_file.name}")
    print(f"  Binance trades: {binance_file.name}")

    # Load data
    market_windows = load_csv(market_file)
    poly_trades = load_csv(poly_file)
    binance_trades = load_csv(binance_file)

    print(f"\nLoaded:")
    print(f"  {len(market_windows)} market windows")
    print(f"  {len(poly_trades)} Polymarket trades")
    print(f"  {len(binance_trades)} Binance trades")

    # Run analysis
    print("\nAnalyzing trader behavior...")
    traders = analyze_traders(poly_trades, market_windows)

    print("Analyzing window statistics...")
    windows = analyze_windows(market_windows, poly_trades, binance_trades)

    print("Analyzing momentum correlations...")
    momentum = analyze_momentum_correlation(windows)

    # Generate report
    generate_report(
        market_windows,
        traders,
        windows,
        momentum,
        symbol_filter=args.symbol.upper() if args.symbol else None
    )

    # Save results if output specified
    if args.output:
        save_analysis(Path(args.output), traders, windows, momentum)
    else:
        # Auto-save to data directory
        output_path = data_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_analysis(output_path, traders, windows, momentum)


if __name__ == "__main__":
    main()
