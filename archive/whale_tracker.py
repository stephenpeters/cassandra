#!/usr/bin/env python3
"""
Polymarket Whale Tracker for Crypto Up/Down Markets
Tracks whale trades on BTC/ETH/SOL/XRP 15-minute prediction markets.

Supports multiple strategies via --strategy flag:
  - crypto (default): Track crypto up/down markets (BTC, ETH, SOL, etc.)
  - politics: Track political/election markets
  - sports: Track sports prediction markets
  - all: Track all whale trades (no filtering)
"""

import requests
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Callable
from abc import ABC, abstractmethod
import json
import argparse

# Rich for terminal UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
except ImportError:
    print("Installing rich library...")
    import subprocess
    subprocess.check_call(["pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text

# ============================================================================
# CONFIGURATION
# ============================================================================

# Known whale wallets focused on crypto markets
WHALE_WALLETS = {
    "gabagool22": "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d",
    # Add more whales here as you discover them
    # "whale_name": "0x...",
}

# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

class Strategy(ABC):
    """Base class for market filtering strategies"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def keywords(self) -> list[str]:
        pass

    def matches(self, market_question: str) -> bool:
        """Check if market matches this strategy"""
        question_lower = market_question.lower()
        return any(kw in question_lower for kw in self.keywords)

    def get_signal(self, trade: "WhaleTrade") -> str:
        """Determine the signal for a trade. Override for custom logic."""
        return "[green]BULL[/]" if trade.is_bullish else "[red]BEAR[/]"


class CryptoStrategy(Strategy):
    """Track crypto price prediction markets (BTC, ETH, SOL, etc.)"""

    @property
    def name(self) -> str:
        return "crypto"

    @property
    def description(self) -> str:
        return "Crypto Up/Down Markets (BTC, ETH, SOL, XRP)"

    @property
    def keywords(self) -> list[str]:
        return [
            "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "ripple",
            "crypto", "doge", "dogecoin", "above", "below", "price"
        ]


class PoliticsStrategy(Strategy):
    """Track political and election markets"""

    @property
    def name(self) -> str:
        return "politics"

    @property
    def description(self) -> str:
        return "Political & Election Markets"

    @property
    def keywords(self) -> list[str]:
        return [
            "trump", "biden", "election", "president", "congress", "senate",
            "democrat", "republican", "vote", "poll", "governor", "mayor",
            "political", "party", "nominee", "cabinet", "impeach"
        ]

    def get_signal(self, trade: "WhaleTrade") -> str:
        """Political signal based on outcome"""
        outcome_lower = trade.outcome.lower()
        if "yes" in outcome_lower:
            return "[cyan]YES[/]" if trade.side == "BUY" else "[magenta]NO[/]"
        return "[magenta]NO[/]" if trade.side == "BUY" else "[cyan]YES[/]"


class SportsStrategy(Strategy):
    """Track sports prediction markets"""

    @property
    def name(self) -> str:
        return "sports"

    @property
    def description(self) -> str:
        return "Sports Prediction Markets"

    @property
    def keywords(self) -> list[str]:
        return [
            "nfl", "nba", "mlb", "nhl", "football", "basketball", "baseball",
            "hockey", "soccer", "tennis", "golf", "ufc", "boxing", "super bowl",
            "world series", "championship", "playoff", "win", "score", "game"
        ]

    def get_signal(self, trade: "WhaleTrade") -> str:
        """Sports just shows BET direction"""
        return f"[yellow]{trade.outcome[:10]}[/]"


class AllMarketsStrategy(Strategy):
    """Track all markets without filtering"""

    @property
    def name(self) -> str:
        return "all"

    @property
    def description(self) -> str:
        return "All Markets (No Filter)"

    @property
    def keywords(self) -> list[str]:
        return []  # Empty = matches everything

    def matches(self, market_question: str) -> bool:
        return True  # Always matches

    def get_signal(self, trade: "WhaleTrade") -> str:
        return f"[dim]{trade.side}[/]"


# Strategy registry
STRATEGIES: dict[str, Strategy] = {
    "crypto": CryptoStrategy(),
    "politics": PoliticsStrategy(),
    "sports": SportsStrategy(),
    "all": AllMarketsStrategy(),
}

# Default strategy
DEFAULT_STRATEGY = "crypto"

# Polymarket Data API
POLYMARKET_API = "https://data-api.polymarket.com"

# Polling interval (seconds)
POLL_INTERVAL = 5

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class WhaleTrade:
    """Represents a whale trade on Polymarket"""
    whale_name: str
    wallet: str
    market: str
    outcome: str
    side: str  # BUY or SELL
    size: float
    price: float
    timestamp: datetime
    transaction_hash: str

    @property
    def position_value(self) -> float:
        return self.size * self.price

    @property
    def is_bullish(self) -> bool:
        """Determine if trade is bullish on the underlying crypto"""
        outcome_lower = self.outcome.lower()
        # Buying YES on "above/up" or SELL on "below/down" = bullish
        if self.side == "BUY":
            return "above" in outcome_lower or "up" in outcome_lower or "yes" in outcome_lower
        else:
            return "below" in outcome_lower or "down" in outcome_lower

# ============================================================================
# API FUNCTIONS
# ============================================================================

def fetch_whale_trades(wallet: str, limit: int = 50) -> list[dict]:
    """Fetch recent trades for a wallet from Polymarket Data API"""
    try:
        url = f"{POLYMARKET_API}/trades"
        params = {
            "user": wallet.lower(),
            "limit": limit,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return []

def fetch_market_info(condition_id: str) -> Optional[dict]:
    """Fetch market details by condition ID"""
    try:
        url = f"{POLYMARKET_API}/markets"
        params = {"condition_id": condition_id}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        markets = response.json()
        return markets[0] if markets else None
    except:
        return None

def market_matches_strategy(market_question: str, strategy: Strategy) -> bool:
    """Check if market matches the current strategy"""
    return strategy.matches(market_question)

def parse_trade(trade_data: dict, whale_name: str, wallet: str, strategy: Optional[Strategy] = None) -> Optional[WhaleTrade]:
    """Parse raw trade data into WhaleTrade object"""
    try:
        # Extract market question from the trade data
        market = trade_data.get("market", {})
        question = market.get("question", "") if isinstance(market, dict) else str(market)

        # Filter by strategy if provided
        if strategy and not strategy.matches(question):
            return None

        # Parse timestamp
        ts = trade_data.get("timestamp", trade_data.get("created_at", ""))
        if isinstance(ts, str):
            try:
                timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                timestamp = datetime.now()
        else:
            timestamp = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts)

        return WhaleTrade(
            whale_name=whale_name,
            wallet=wallet,
            market=question[:80] + "..." if len(question) > 80 else question,
            outcome=trade_data.get("outcome", trade_data.get("asset", {}).get("outcome", "Unknown")),
            side=trade_data.get("side", "BUY").upper(),
            size=float(trade_data.get("size", trade_data.get("amount", 0))),
            price=float(trade_data.get("price", 0)),
            timestamp=timestamp,
            transaction_hash=trade_data.get("transactionHash", trade_data.get("tx_hash", ""))[:16],
        )
    except Exception as e:
        return None

# ============================================================================
# UI COMPONENTS
# ============================================================================

class WhaleTracker:
    """Main tracker with Rich UI"""

    def __init__(self, strategy: Strategy = None):
        self.console = Console()
        self.strategy = strategy or STRATEGIES[DEFAULT_STRATEGY]
        self.trades: list[WhaleTrade] = []
        self.last_seen: dict[str, str] = {}  # wallet -> last tx hash
        self.stats = {
            "total_trades": 0,
            "bullish_trades": 0,
            "bearish_trades": 0,
            "total_volume": 0.0,
        }
        self.alerts: list[str] = []

    def fetch_all_whale_trades(self) -> list[WhaleTrade]:
        """Fetch trades from all tracked whales"""
        new_trades = []

        for name, wallet in WHALE_WALLETS.items():
            raw_trades = fetch_whale_trades(wallet)

            for trade_data in raw_trades:
                tx_hash = trade_data.get("transactionHash", trade_data.get("tx_hash", ""))

                # Skip if we've seen this trade
                if self.last_seen.get(wallet) == tx_hash:
                    break

                trade = parse_trade(trade_data, name, wallet, self.strategy)
                if trade:
                    new_trades.append(trade)

            # Update last seen
            if raw_trades:
                self.last_seen[wallet] = raw_trades[0].get("transactionHash", raw_trades[0].get("tx_hash", ""))

        return new_trades

    def update_stats(self, new_trades: list[WhaleTrade]):
        """Update statistics with new trades"""
        for trade in new_trades:
            self.stats["total_trades"] += 1
            self.stats["total_volume"] += trade.position_value
            if trade.is_bullish:
                self.stats["bullish_trades"] += 1
            else:
                self.stats["bearish_trades"] += 1

            # Add alert for significant trades
            if trade.position_value > 1000:
                self.alerts.append(
                    f"[bold yellow]ALERT:[/] {trade.whale_name} {'BOUGHT' if trade.side == 'BUY' else 'SOLD'} "
                    f"${trade.position_value:,.0f} on {trade.market[:40]}..."
                )
                if len(self.alerts) > 5:
                    self.alerts.pop(0)

    def build_trades_table(self) -> Table:
        """Build the main trades table"""
        table = Table(title=f"Recent Whale Trades - {self.strategy.description}", expand=True)

        table.add_column("Time", style="dim", width=8)
        table.add_column("Whale", style="cyan", width=12)
        table.add_column("Market", style="white", width=40)
        table.add_column("Side", width=6)
        table.add_column("Size", justify="right", width=10)
        table.add_column("Price", justify="right", width=8)
        table.add_column("Signal", width=8)

        # Show last 15 trades
        for trade in self.trades[:15]:
            side_style = "green" if trade.side == "BUY" else "red"
            signal = self.strategy.get_signal(trade)

            table.add_row(
                trade.timestamp.strftime("%H:%M:%S"),
                trade.whale_name,
                trade.market,
                f"[{side_style}]{trade.side}[/]",
                f"${trade.size:,.0f}",
                f"{trade.price:.2f}",
                signal,
            )

        if not self.trades:
            table.add_row("--", "--", f"Waiting for {self.strategy.name} trades...", "--", "--", "--", "--")

        return table

    def build_stats_panel(self) -> Panel:
        """Build statistics panel"""
        bullish_pct = (
            self.stats["bullish_trades"] / self.stats["total_trades"] * 100
            if self.stats["total_trades"] > 0 else 50
        )

        stats_text = Text()
        stats_text.append(f"Total Trades: {self.stats['total_trades']}  ", style="white")
        stats_text.append(f"Volume: ${self.stats['total_volume']:,.0f}  ", style="yellow")
        stats_text.append(f"Bull: {self.stats['bullish_trades']} ", style="green")
        stats_text.append(f"({bullish_pct:.0f}%)  ", style="green")
        stats_text.append(f"Bear: {self.stats['bearish_trades']}", style="red")

        return Panel(stats_text, title="Session Stats", border_style="blue")

    def build_alerts_panel(self) -> Panel:
        """Build alerts panel"""
        if self.alerts:
            alert_text = "\n".join(self.alerts)
        else:
            alert_text = "[dim]No significant trades yet...[/]"

        return Panel(alert_text, title="Alerts (>$1k trades)", border_style="yellow")

    def build_whales_panel(self) -> Panel:
        """Build tracked whales panel"""
        whale_text = Text()
        for name, wallet in WHALE_WALLETS.items():
            whale_text.append(f"{name}: ", style="cyan")
            whale_text.append(f"{wallet[:10]}...{wallet[-6:]}\n", style="dim")

        return Panel(whale_text, title="Tracked Whales", border_style="green")

    def build_layout(self) -> Layout:
        """Build the full UI layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="stats", size=3),
            Layout(name="main", ratio=2),
            Layout(name="footer", size=8),
        )

        # Header
        header_text = Text("POLYMARKET WHALE TRACKER", style="bold white on blue", justify="center")
        layout["header"].update(Panel(header_text, border_style="blue"))

        # Stats
        layout["stats"].update(self.build_stats_panel())

        # Main trades table
        layout["main"].update(self.build_trades_table())

        # Footer split into alerts and whales
        layout["footer"].split_row(
            Layout(self.build_alerts_panel(), name="alerts"),
            Layout(self.build_whales_panel(), name="whales", size=40),
        )

        return layout

    def run(self):
        """Main run loop with live updating UI"""
        self.console.print("[bold blue]Starting Polymarket Whale Tracker...[/]")
        self.console.print(f"[dim]Strategy: {self.strategy.description}[/]")
        self.console.print(f"[dim]Tracking {len(WHALE_WALLETS)} whale(s). Polling every {POLL_INTERVAL}s. Ctrl+C to exit.[/]")
        self.console.print()

        # Initial fetch
        initial_trades = self.fetch_all_whale_trades()
        self.trades = initial_trades
        self.update_stats(initial_trades)

        try:
            with Live(self.build_layout(), console=self.console, refresh_per_second=1) as live:
                while True:
                    time.sleep(POLL_INTERVAL)

                    # Fetch new trades
                    new_trades = self.fetch_all_whale_trades()

                    if new_trades:
                        # Prepend new trades to list
                        self.trades = new_trades + self.trades
                        self.trades = self.trades[:50]  # Keep last 50
                        self.update_stats(new_trades)

                    # Update display
                    live.update(self.build_layout())

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Tracker stopped.[/]")

# ============================================================================
# MOCK MODE FOR TESTING
# ============================================================================

class MockWhaleTracker(WhaleTracker):
    """Mock tracker for testing without live API"""

    def __init__(self, strategy: Strategy = None):
        super().__init__(strategy)
        self.mock_trades = self._generate_mock_trades()
        self.mock_index = 0

    def _generate_mock_trades(self) -> list[WhaleTrade]:
        """Generate fake trades for testing based on strategy"""
        import random

        # Markets by strategy
        market_sets = {
            "crypto": [
                "Will Bitcoin be above $100,000 at 12:00 PM ET?",
                "Will ETH be above $4,000 at 1:00 PM ET?",
                "Will BTC be below $99,000 at 2:00 PM ET?",
                "Will Solana be above $200 at 3:00 PM ET?",
                "Will XRP price be above $2.50 at 4:00 PM ET?",
            ],
            "politics": [
                "Will Trump win the 2028 election?",
                "Will Democrats control the Senate?",
                "Will Biden announce re-election bid?",
                "Will Republicans win House majority?",
                "Will the cabinet nominee be confirmed?",
            ],
            "sports": [
                "Will the Chiefs win the Super Bowl?",
                "Will Lakers win NBA Championship?",
                "Will Yankees win World Series?",
                "Will Connor McDavid score tonight?",
                "Will Mahomes throw 300+ yards?",
            ],
            "all": [
                "Will Bitcoin be above $100,000?",
                "Will Trump win 2028?",
                "Will Lakers win championship?",
                "Will AI replace programmers by 2030?",
                "Will SpaceX land on Mars by 2026?",
            ],
        }

        markets = market_sets.get(self.strategy.name, market_sets["crypto"])

        trades = []
        for i in range(20):
            trade = WhaleTrade(
                whale_name=random.choice(list(WHALE_WALLETS.keys())),
                wallet=list(WHALE_WALLETS.values())[0],
                market=random.choice(markets),
                outcome="Yes" if random.random() > 0.5 else "No",
                side="BUY" if random.random() > 0.3 else "SELL",
                size=random.uniform(100, 10000),
                price=random.uniform(0.3, 0.8),
                timestamp=datetime.now(),
                transaction_hash=f"0x{random.randbytes(8).hex()}",
            )
            trades.append(trade)

        return trades

    def fetch_all_whale_trades(self) -> list[WhaleTrade]:
        """Return mock trades periodically"""
        import random

        # 30% chance of new trade each poll
        if random.random() < 0.3 and self.mock_index < len(self.mock_trades):
            trade = self.mock_trades[self.mock_index]
            trade.timestamp = datetime.now()  # Fresh timestamp
            self.mock_index += 1
            return [trade]
        return []

# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Polymarket Whale Tracker - Track whale trades on prediction markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python whale_tracker.py                    # Track crypto markets (default)
  python whale_tracker.py -s politics        # Track political markets
  python whale_tracker.py -s sports          # Track sports markets
  python whale_tracker.py -s all             # Track all markets
  python whale_tracker.py --mock             # Run with simulated trades
  python whale_tracker.py -s politics --mock # Mock mode with politics strategy
        """
    )

    parser.add_argument(
        "-s", "--strategy",
        choices=list(STRATEGIES.keys()),
        default=DEFAULT_STRATEGY,
        help=f"Market strategy to track (default: {DEFAULT_STRATEGY})"
    )

    parser.add_argument(
        "-m", "--mock",
        action="store_true",
        help="Run in mock mode with simulated trades"
    )

    parser.add_argument(
        "-l", "--list-strategies",
        action="store_true",
        help="List available strategies and exit"
    )

    return parser.parse_args()


def list_strategies():
    """Print available strategies"""
    print("\nAvailable Strategies:")
    print("-" * 50)
    for name, strategy in STRATEGIES.items():
        print(f"  {name:12} - {strategy.description}")
        if strategy.keywords:
            keywords_preview = ", ".join(strategy.keywords[:5])
            if len(strategy.keywords) > 5:
                keywords_preview += "..."
            print(f"                 Keywords: {keywords_preview}")
    print()


def main():
    args = parse_args()

    # List strategies if requested
    if args.list_strategies:
        list_strategies()
        return

    # Get selected strategy
    strategy = STRATEGIES[args.strategy]

    # Create tracker
    if args.mock:
        print(f"Running in MOCK mode (simulated {args.strategy} trades)")
        tracker = MockWhaleTracker(strategy)
    else:
        tracker = WhaleTracker(strategy)

    tracker.run()


if __name__ == "__main__":
    main()
