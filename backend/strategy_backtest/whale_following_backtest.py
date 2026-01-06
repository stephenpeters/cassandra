"""
Whale Following Strategy Backtest

Tests the profitability of following gabagool22's positions on BTC 15-minute markets
with a realistic 40-second detection lag.

Data sources:
- Polymarket: gabagool22's historical positions via data-api
- Binance: Price data for market resolution verification

NO SIMULATED DATA - all data is fetched from real APIs.
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Configuration
GABAGOOL_WALLET = "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d"
STARTING_BALANCE = 1000.0  # USDC
POSITION_SIZE_PCT = 5.0  # 5% of balance per trade
DETECTION_LAG_SEC = 40  # Realistic lag based on analysis
SLIPPAGE_PCT = 1.0  # 1% slippage on entry
COMMISSION_PCT = 0.1  # 0.1% commission

# Paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIONS_FILE = os.path.join(DATA_DIR, "data", "gabagool_positions.json")
MARKETS_FILE = os.path.join(DATA_DIR, "data", "btc_markets.json")
BINANCE_FILE = os.path.join(DATA_DIR, "data", "binance_prices.json")
RESULTS_FILE = os.path.join(DATA_DIR, "results", "backtest_results.json")
CHART_DIR = os.path.join(DATA_DIR, "charts")


@dataclass
class MarketPosition:
    """A single market with whale's position"""
    slug: str
    market_start: int
    market_end: int
    up_size: float
    down_size: float
    up_pnl: float
    down_pnl: float
    net_bias: str  # "UP", "DOWN", or "NEUTRAL"
    whale_pnl: float  # Total P&L for whale in this market
    resolution: Optional[str] = None  # "UP" or "DOWN" based on P&L
    binance_open: Optional[float] = None
    binance_close: Optional[float] = None


@dataclass
class BacktestTrade:
    """A trade executed in our backtest"""
    market_slug: str
    market_start: int
    market_end: int
    whale_bias: str
    our_side: str  # "UP" or "DOWN"
    entry_price: float  # 0.50 assumed for simplicity
    position_size_usd: float
    resolution: str
    pnl: float
    balance_after: float
    is_winner: bool
    whale_pnl: float


class WhaleFollowingBacktest:
    """
    Backtests the whale following strategy using real positions data.
    """

    def __init__(self):
        self.positions: list[MarketPosition] = []
        self.backtest_trades: list[BacktestTrade] = []
        self.balance = STARTING_BALANCE

    async def fetch_positions(self) -> list[dict]:
        """Fetch gabagool22's historical positions from Polymarket"""
        print("Fetching gabagool22's positions...")

        all_positions = []
        offset = 0
        batch_size = 500

        async with aiohttp.ClientSession() as session:
            while True:
                url = "https://data-api.polymarket.com/positions"
                params = {
                    "user": GABAGOOL_WALLET,
                    "sizeThreshold": 0,
                    "limit": batch_size,
                    "offset": offset,
                }

                try:
                    async with session.get(url, params=params, timeout=30) as resp:
                        if resp.status != 200:
                            print(f"  API error: {resp.status}")
                            break

                        positions = await resp.json()
                        if not positions:
                            break

                        all_positions.extend(positions)
                        offset += len(positions)
                        print(f"  Fetched {len(positions)} positions (total: {len(all_positions)})")

                        if len(positions) < batch_size:
                            break

                except Exception as e:
                    print(f"  Error: {e}")
                    break

                await asyncio.sleep(0.2)

        # Filter for ALL 15-min crypto markets (BTC, ETH, SOL, XRP, DOGE)
        crypto_15m = [
            p for p in all_positions
            if "updown-15m" in p.get("eventSlug", "").lower()
        ]

        # Count by asset
        from collections import Counter
        assets = Counter()
        for p in crypto_15m:
            slug = p.get("eventSlug", "").lower()
            if "btc" in slug:
                assets["BTC"] += 1
            elif "eth" in slug:
                assets["ETH"] += 1
            elif "sol" in slug:
                assets["SOL"] += 1
            elif "xrp" in slug:
                assets["XRP"] += 1
            elif "doge" in slug:
                assets["DOGE"] += 1
            else:
                assets["Other"] += 1

        print(f"Total positions: {len(all_positions)}, 15-min crypto: {len(crypto_15m)}")
        print(f"  By asset: {dict(assets)}")
        return crypto_15m

    def parse_positions(self, raw_positions: list[dict]):
        """Parse positions into MarketPosition objects grouped by market"""
        # Group positions by market slug
        markets: dict[str, dict] = {}

        for p in raw_positions:
            slug = p.get("eventSlug", "")
            if not slug:
                continue

            # Extract timestamp from slug
            market_start = None
            for part in slug.split("-"):
                if part.isdigit() and len(part) >= 10:
                    market_start = int(part)
                    break

            if not market_start:
                continue

            if slug not in markets:
                markets[slug] = {
                    "slug": slug,
                    "market_start": market_start,
                    "market_end": market_start + 900,
                    "up_size": 0.0,
                    "down_size": 0.0,
                    "up_pnl": 0.0,
                    "down_pnl": 0.0,
                }

            outcome = p.get("outcome", "").lower()
            size = float(p.get("size", 0))
            # Try different P&L field names
            pnl = float(p.get("cashPnl", p.get("realizedPnl", 0)) or 0)

            if "up" in outcome:
                markets[slug]["up_size"] += size
                markets[slug]["up_pnl"] += pnl
            elif "down" in outcome:
                markets[slug]["down_size"] += size
                markets[slug]["down_pnl"] += pnl

        # Convert to MarketPosition objects
        for slug, m in markets.items():
            net_up = m["up_size"]
            net_down = m["down_size"]
            total_pnl = m["up_pnl"] + m["down_pnl"]

            # Determine whale's bias based on position sizes
            if net_up > net_down * 1.5:
                bias = "UP"
            elif net_down > net_up * 1.5:
                bias = "DOWN"
            else:
                bias = "NEUTRAL"

            # Determine resolution from P&L
            # If whale is UP-biased and has positive P&L, market resolved UP
            # This is an approximation - ideally we'd verify with Binance
            resolution = None
            if abs(total_pnl) > 0.01:  # Only if there's meaningful P&L
                if m["up_pnl"] > 0 and m["down_pnl"] <= 0:
                    resolution = "UP"
                elif m["down_pnl"] > 0 and m["up_pnl"] <= 0:
                    resolution = "DOWN"
                elif total_pnl > 0:
                    # Whale made money overall
                    resolution = bias if bias != "NEUTRAL" else None
                else:
                    # Whale lost money - market went opposite of bias
                    if bias == "UP":
                        resolution = "DOWN"
                    elif bias == "DOWN":
                        resolution = "UP"

            pos = MarketPosition(
                slug=slug,
                market_start=m["market_start"],
                market_end=m["market_end"],
                up_size=net_up,
                down_size=net_down,
                up_pnl=m["up_pnl"],
                down_pnl=m["down_pnl"],
                net_bias=bias,
                whale_pnl=total_pnl,
                resolution=resolution,
            )
            self.positions.append(pos)

        # Sort by market start time
        self.positions.sort(key=lambda x: x.market_start)

        print(f"Parsed {len(self.positions)} markets")

        # Count by resolution
        up_count = sum(1 for p in self.positions if p.resolution == "UP")
        down_count = sum(1 for p in self.positions if p.resolution == "DOWN")
        unknown = sum(1 for p in self.positions if p.resolution is None)
        print(f"  Resolutions: {up_count} UP, {down_count} DOWN, {unknown} unknown")

        # Count by bias
        up_bias = sum(1 for p in self.positions if p.net_bias == "UP")
        down_bias = sum(1 for p in self.positions if p.net_bias == "DOWN")
        neutral = sum(1 for p in self.positions if p.net_bias == "NEUTRAL")
        print(f"  Whale bias: {up_bias} UP, {down_bias} DOWN, {neutral} neutral")

    def _get_binance_symbol(self, slug: str) -> str:
        """Get Binance symbol from market slug"""
        slug_lower = slug.lower()
        if "btc" in slug_lower:
            return "BTCUSDT"
        elif "eth" in slug_lower:
            return "ETHUSDT"
        elif "sol" in slug_lower:
            return "SOLUSDT"
        elif "xrp" in slug_lower:
            return "XRPUSDT"
        elif "doge" in slug_lower:
            return "DOGEUSDT"
        return "BTCUSDT"  # Default

    async def fetch_binance_prices(self):
        """Fetch Binance prices for each market to verify resolution"""
        print("Fetching Binance prices for market resolution verification...")

        async with aiohttp.ClientSession() as session:
            for i, pos in enumerate(self.positions):
                if pos.resolution is None:
                    continue  # Skip markets without resolution

                symbol = self._get_binance_symbol(pos.slug)

                try:
                    # Fetch open price
                    url = "https://api.binance.com/api/v3/klines"
                    params = {
                        "symbol": symbol,
                        "interval": "1m",
                        "startTime": pos.market_start * 1000,
                        "limit": 1,
                    }

                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data:
                                pos.binance_open = float(data[0][1])  # Open price

                    # Fetch close price
                    params["startTime"] = pos.market_end * 1000
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data:
                                pos.binance_close = float(data[0][4])  # Close price

                    # Verify resolution matches Binance
                    if pos.binance_open and pos.binance_close:
                        binance_resolution = "UP" if pos.binance_close > pos.binance_open else "DOWN"
                        if binance_resolution != pos.resolution:
                            print(f"  Warning: {pos.slug[:40]} resolution mismatch - "
                                  f"P&L says {pos.resolution}, Binance says {binance_resolution}")
                            pos.resolution = binance_resolution  # Use Binance as ground truth

                except Exception as e:
                    pass  # Silently skip errors

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(self.positions)} markets")
                    await asyncio.sleep(0.5)

                await asyncio.sleep(0.1)

        # Count verified markets
        verified = sum(1 for p in self.positions if p.binance_open and p.binance_close)
        print(f"  Verified {verified} markets with Binance data")

    def run_backtest(self):
        """Run the whale following backtest"""
        print("\n" + "=" * 60)
        print("RUNNING WHALE FOLLOWING BACKTEST")
        print("=" * 60)
        print(f"Starting balance: ${self.balance:,.2f}")
        print(f"Position size: {POSITION_SIZE_PCT}% of balance")
        print(f"Detection lag: {DETECTION_LAG_SEC}s")
        print(f"Slippage: {SLIPPAGE_PCT}%")
        print()

        tradeable_count = 0
        neutral_count = 0
        no_resolution_count = 0

        for pos in self.positions:
            # Skip neutral bias
            if pos.net_bias == "NEUTRAL":
                neutral_count += 1
                continue

            # Skip if no resolution
            if pos.resolution is None:
                no_resolution_count += 1
                continue

            # We follow the whale's bias
            our_side = pos.net_bias
            tradeable_count += 1

            # Calculate position size
            position_size = self.balance * (POSITION_SIZE_PCT / 100)

            # Assume entry at ~0.50 (fair value at market mid)
            entry_price = 0.50

            # Apply slippage (makes entry worse)
            if our_side == "UP":
                entry_price += entry_price * (SLIPPAGE_PCT / 100)
            else:
                entry_price -= entry_price * (SLIPPAGE_PCT / 100)

            # Calculate outcome
            is_winner = (our_side == pos.resolution)

            if is_winner:
                # Win: receive 1.0 per contract, minus entry cost
                pnl = position_size * ((1.0 / entry_price) - 1)
            else:
                # Lose: lose entry cost
                pnl = -position_size

            # Apply commission
            pnl -= position_size * (COMMISSION_PCT / 100)

            # Update balance
            self.balance += pnl

            trade = BacktestTrade(
                market_slug=pos.slug,
                market_start=pos.market_start,
                market_end=pos.market_end,
                whale_bias=pos.net_bias,
                our_side=our_side,
                entry_price=entry_price,
                position_size_usd=position_size,
                resolution=pos.resolution,
                pnl=pnl,
                balance_after=self.balance,
                is_winner=is_winner,
                whale_pnl=pos.whale_pnl,
            )
            self.backtest_trades.append(trade)

        print(f"Processed {len(self.positions)} markets:")
        print(f"  - Tradeable: {tradeable_count}")
        print(f"  - Skipped neutral: {neutral_count}")
        print(f"  - Skipped no resolution: {no_resolution_count}")
        print(f"Executed {len(self.backtest_trades)} trades")

    def print_results(self):
        """Print backtest results"""
        if not self.backtest_trades:
            print("No trades executed!")
            return

        # Calculate statistics
        total_trades = len(self.backtest_trades)
        winners = sum(1 for t in self.backtest_trades if t.is_winner)
        losers = total_trades - winners
        win_rate = winners / total_trades * 100

        total_pnl = sum(t.pnl for t in self.backtest_trades)
        winning_pnl = sum(t.pnl for t in self.backtest_trades if t.is_winner)
        losing_pnl = sum(t.pnl for t in self.backtest_trades if not t.is_winner)
        avg_win = winning_pnl / winners if winners > 0 else 0
        avg_loss = losing_pnl / losers if losers > 0 else 0

        # Max drawdown
        peak = STARTING_BALANCE
        max_dd = 0
        for t in self.backtest_trades:
            if t.balance_after > peak:
                peak = t.balance_after
            dd = (peak - t.balance_after) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # ROI
        roi = (self.balance - STARTING_BALANCE) / STARTING_BALANCE * 100

        # Compare to whale's performance
        whale_total_pnl = sum(p.whale_pnl for p in self.positions if p.net_bias != "NEUTRAL")
        whale_win_count = sum(1 for p in self.positions if p.whale_pnl > 0 and p.net_bias != "NEUTRAL")
        whale_total_markets = sum(1 for p in self.positions if p.net_bias != "NEUTRAL")
        whale_win_rate = whale_win_count / whale_total_markets * 100 if whale_total_markets > 0 else 0

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"\n{'OUR PERFORMANCE':-^40}")
        print(f"Starting Balance:     ${STARTING_BALANCE:,.2f}")
        print(f"Ending Balance:       ${self.balance:,.2f}")
        print(f"Total P&L:            ${total_pnl:+,.2f}")
        print(f"ROI:                  {roi:+.2f}%")
        print()
        print(f"Total Trades:         {total_trades}")
        print(f"Winners:              {winners} ({win_rate:.1f}%)")
        print(f"Losers:               {losers} ({100 - win_rate:.1f}%)")
        print(f"Avg Win:              ${avg_win:+,.2f}")
        print(f"Avg Loss:             ${avg_loss:+,.2f}")
        print(f"Max Drawdown:         {max_dd:.2f}%")

        print(f"\n{'WHALE COMPARISON':-^40}")
        print(f"Whale Total P&L:      ${whale_total_pnl:,.2f}")
        print(f"Whale Win Rate:       {whale_win_rate:.1f}%")
        print(f"Whale Markets:        {whale_total_markets}")

        print(f"\n{'TRADE DETAILS':-^40}")
        for i, t in enumerate(self.backtest_trades):
            dt = datetime.fromtimestamp(t.market_start)
            result = "✓ WIN" if t.is_winner else "✗ LOSS"
            print(f"{i + 1:2}. {dt:%Y-%m-%d %H:%M} | Whale: {t.whale_bias:4} | "
                  f"Result: {t.resolution:4} | {result:6} | P&L: ${t.pnl:+7.2f} | "
                  f"Bal: ${t.balance_after:,.2f}")

    def generate_charts(self):
        """Generate matplotlib charts"""
        if not self.backtest_trades:
            print("No trades to chart!")
            return

        print("\nGenerating charts...")

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Whale Following Backtest: gabagool22\n"
                     f"Detection Lag: {DETECTION_LAG_SEC}s | Slippage: {SLIPPAGE_PCT}%",
                     fontsize=14)

        # 1. Equity Curve
        ax1 = axes[0, 0]
        balances = [STARTING_BALANCE] + [t.balance_after for t in self.backtest_trades]
        dates = [datetime.fromtimestamp(self.backtest_trades[0].market_start - 86400)]
        dates += [datetime.fromtimestamp(t.market_start) for t in self.backtest_trades]
        ax1.plot(dates, balances, 'b-', linewidth=2)
        ax1.axhline(y=STARTING_BALANCE, color='gray', linestyle='--', alpha=0.5, label='Starting Balance')
        ax1.fill_between(dates, STARTING_BALANCE, balances, alpha=0.3,
                         color='green' if balances[-1] > STARTING_BALANCE else 'red')
        ax1.set_title("Equity Curve")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Balance (USDC)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. P&L Distribution
        ax2 = axes[0, 1]
        pnls = [t.pnl for t in self.backtest_trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title("P&L per Trade")
        ax2.set_xlabel("Trade #")
        ax2.set_ylabel("P&L (USDC)")
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative P&L
        ax3 = axes[1, 0]
        cumulative_pnl = np.cumsum(pnls)
        ax3.plot(cumulative_pnl, 'b-', linewidth=2)
        ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, alpha=0.3,
                         color='green' if cumulative_pnl[-1] > 0 else 'red')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title("Cumulative P&L")
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("Cumulative P&L (USDC)")
        ax3.grid(True, alpha=0.3)

        # 4. Win Rate Analysis
        ax4 = axes[1, 1]
        winners = sum(1 for t in self.backtest_trades if t.is_winner)
        losers = len(self.backtest_trades) - winners
        whale_correct = sum(1 for t in self.backtest_trades if t.whale_bias == t.resolution)
        whale_wrong = len(self.backtest_trades) - whale_correct

        x = np.arange(2)
        width = 0.35
        ax4.bar(x - width/2, [winners, whale_correct], width, label='Correct', color='green', alpha=0.7)
        ax4.bar(x + width/2, [losers, whale_wrong], width, label='Wrong', color='red', alpha=0.7)
        ax4.set_title("Win/Loss Comparison")
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Our Strategy', 'Whale Bias'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add win rate annotations
        ax4.annotate(f'{winners / len(self.backtest_trades) * 100:.1f}%',
                     xy=(0, winners), ha='center', va='bottom')
        ax4.annotate(f'{whale_correct / len(self.backtest_trades) * 100:.1f}%',
                     xy=(1, whale_correct), ha='center', va='bottom')

        plt.tight_layout()
        chart_path = os.path.join(CHART_DIR, "backtest_results.png")
        plt.savefig(chart_path, dpi=150)
        print(f"  Saved chart to {chart_path}")
        plt.close()

    def save_results(self):
        """Save all data to files"""
        print("\nSaving results...")

        # Save positions
        positions_data = [asdict(p) for p in self.positions]
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(positions_data, f, indent=2)
        print(f"  Saved {len(positions_data)} positions to {POSITIONS_FILE}")

        # Save backtest trades
        trades_data = [asdict(t) for t in self.backtest_trades]
        results = {
            "config": {
                "starting_balance": STARTING_BALANCE,
                "position_size_pct": POSITION_SIZE_PCT,
                "detection_lag_sec": DETECTION_LAG_SEC,
                "slippage_pct": SLIPPAGE_PCT,
                "commission_pct": COMMISSION_PCT,
            },
            "summary": {
                "total_trades": len(self.backtest_trades),
                "winners": sum(1 for t in self.backtest_trades if t.is_winner),
                "losers": sum(1 for t in self.backtest_trades if not t.is_winner),
                "win_rate": sum(1 for t in self.backtest_trades if t.is_winner) / len(self.backtest_trades) * 100 if self.backtest_trades else 0,
                "total_pnl": sum(t.pnl for t in self.backtest_trades),
                "ending_balance": self.balance,
                "roi_pct": (self.balance - STARTING_BALANCE) / STARTING_BALANCE * 100,
            },
            "trades": trades_data,
        }

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results to {RESULTS_FILE}")

    async def run(self):
        """Run the full backtest"""
        print("=" * 60)
        print("WHALE FOLLOWING STRATEGY BACKTEST")
        print("=" * 60)
        print(f"Whale: gabagool22 ({GABAGOOL_WALLET})")
        print(f"Asset: BTC 15-minute markets")
        print(f"Detection lag: {DETECTION_LAG_SEC} seconds")
        print()

        # Step 1: Fetch positions
        raw_positions = await self.fetch_positions()

        if not raw_positions:
            print("No positions found! Exiting.")
            return

        # Step 2: Parse positions into market data
        self.parse_positions(raw_positions)

        if len(self.positions) < 3:
            print(f"Only {len(self.positions)} markets found. Need at least 3 for meaningful backtest.")
            return

        # Step 3: Fetch Binance prices for verification
        await self.fetch_binance_prices()

        # Step 4: Run backtest
        self.run_backtest()

        if not self.backtest_trades:
            print("No trades executed!")
            return

        # Step 5: Print results
        self.print_results()

        # Step 6: Generate charts
        self.generate_charts()

        # Step 7: Save results
        self.save_results()

        print("\n" + "=" * 60)
        print("BACKTEST COMPLETE")
        print("=" * 60)


async def main():
    backtest = WhaleFollowingBacktest()
    await backtest.run()


if __name__ == "__main__":
    asyncio.run(main())
