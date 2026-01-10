"""
Strategy Manager - Loads and manages trading strategy configurations.

Provides:
- Strategy enable/disable per market
- Copy trading trader management
- Position sizing based on strategy settings
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TraderConfig:
    """Configuration for a trader to copy"""
    name: str
    address: str
    enabled: bool = True
    min_trade_size: float = 500
    trust_score: float = 0.5

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "address": self.address,
            "enabled": self.enabled,
            "min_trade_size": self.min_trade_size,
            "trust_score": self.trust_score,
        }


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    enabled: bool = True
    description: str = ""
    markets: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    traders: list[TraderConfig] = field(default_factory=list)

    def is_enabled_for_market(self, market: str) -> bool:
        """Check if strategy is enabled for a specific market"""
        return self.enabled and market.upper() in [m.upper() for m in self.markets]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "description": self.description,
            "markets": self.markets,
            "settings": self.settings,
            "traders": [t.to_dict() for t in self.traders],
        }


class StrategyManager:
    """
    Manages trading strategy configurations.

    Single source of truth for:
    - Which strategies are enabled
    - Which markets each strategy applies to
    - Copy trading trader settings
    - Position sizing rules
    """

    def __init__(self, config_path: str = "strategy_config.json"):
        self.config_path = config_path
        self.strategies: dict[str, StrategyConfig] = {}
        self.global_settings: dict = {}
        self._load_config()

    def _load_config(self):
        """Load strategy configuration from JSON file"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Strategy config not found at {self.config_path}, using defaults")
                self._init_defaults()
                return

            with open(self.config_path, "r") as f:
                data = json.load(f)

            # Load strategies
            for name, config in data.get("strategies", {}).items():
                traders = []
                for t in config.get("traders", []):
                    traders.append(TraderConfig(**t))

                self.strategies[name] = StrategyConfig(
                    name=name,
                    enabled=config.get("enabled", True),
                    description=config.get("description", ""),
                    markets=config.get("markets", []),
                    settings=config.get("settings", {}),
                    traders=traders,
                )

            # Load global settings
            self.global_settings = data.get("global_settings", {})

            logger.info(f"Loaded {len(self.strategies)} strategies from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load strategy config: {e}")
            self._init_defaults()

    def _init_defaults(self):
        """Initialize with default configuration"""
        self.strategies = {
            "latency_gap": StrategyConfig(
                name="latency_gap",
                enabled=True,
                description="Trade based on Binance price leading Polymarket odds",
                markets=["BTC", "ETH"],
                settings={
                    "min_edge_pct": 5.0,
                    "checkpoints": ["7m30s", "9m"],
                    "min_confidence": 0.6,
                },
            ),
            "copy_trading": StrategyConfig(
                name="copy_trading",
                enabled=True,
                description="Copy trades from successful traders",
                markets=["BTC"],
                settings={
                    "scale_to_account": True,
                    "reference_balance": 10000,
                    "position_multiplier": 0.5,
                },
                traders=[
                    TraderConfig(
                        name="gabagool22",
                        address="0x8758FE5b2e8cdCc4dE988E91D9F215E1A6d0f5E2",
                        enabled=True,
                    ),
                ],
            ),
            "dip_arb": StrategyConfig(
                name="dip_arb",
                enabled=False,  # Disabled by default - needs backtesting
                description="Two-leg flash crash arbitrage: buy dip, then hedge opposite",
                markets=["BTC"],
                settings={
                    "dip_threshold": 0.15,  # 15% drop triggers Leg1
                    "surge_threshold": 0.15,  # 15% surge also triggers
                    "sliding_window_sec": 3,  # Detect flash moves in 3s window
                    "sum_target": 0.95,  # Hedge when leg1_entry + opposite_ask < 0.95
                    "window_minutes": 2,  # Only trigger in first 2 minutes
                    "enable_surge": True,
                    "min_profit_rate": 0.05,  # Minimum 5% profit for Leg2
                    "position_size_pct": 2.0,
                    "max_position_usd": 100.0,
                    "shares_per_leg": 10,  # Same shares on both legs
                },
            ),
            "latency_arb": StrategyConfig(
                name="latency_arb",
                enabled=False,  # Disabled by default - needs backtesting
                description="Exploit Binance→Polymarket lag: buy when price already moved",
                markets=["BTC", "ETH", "SOL"],
                settings={
                    "min_move_pct": 0.3,  # Trigger on 0.3% Binance move
                    "take_profit_pct": 6.0,  # Take profit at 6% return
                    "max_entry_price": 0.65,  # Don't chase above 65c
                    "min_time_remaining_sec": 300,  # Need 5+ min remaining
                    "hold_if_confirmed": True,  # Hold 90%+ winners to expiry
                    "confirmed_threshold": 0.90,
                    "position_size_pct": 2.0,
                    "max_position_usd": 100.0,
                    "slippage_buffer_pct": 1.5,
                    "cooldown_sec": 30,
                },
            ),
        }
        self.global_settings = {
            "enabled_markets": ["BTC", "ETH", "SOL", "XRP"],
            "max_concurrent_positions": 3,
            "max_daily_trades": 20,
        }

    def save_config(self):
        """Save current configuration to JSON file"""
        data = {
            "strategies": {},
            "global_settings": self.global_settings,
        }

        for name, strategy in self.strategies.items():
            data["strategies"][name] = {
                "enabled": strategy.enabled,
                "description": strategy.description,
                "markets": strategy.markets,
                "settings": strategy.settings,
                "traders": [t.to_dict() for t in strategy.traders],
            }

        try:
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved strategy config to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save strategy config: {e}")

    # -------------------------------------------------------------------------
    # Strategy Queries
    # -------------------------------------------------------------------------

    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """Get a strategy by name"""
        return self.strategies.get(name)

    def is_strategy_enabled(self, strategy: str, market: str) -> bool:
        """Check if a strategy is enabled for a market"""
        strat = self.strategies.get(strategy)
        if not strat:
            return False
        return strat.is_enabled_for_market(market)

    def get_enabled_strategies(self, market: str) -> list[str]:
        """Get all strategies enabled for a market"""
        return [
            name for name, strat in self.strategies.items()
            if strat.is_enabled_for_market(market)
        ]

    def get_enabled_markets(self) -> list[str]:
        """Get all globally enabled markets"""
        return self.global_settings.get("enabled_markets", ["BTC"])

    # -------------------------------------------------------------------------
    # Strategy Mutations
    # -------------------------------------------------------------------------

    def enable_strategy(self, name: str, enabled: bool = True):
        """Enable or disable a strategy"""
        if name in self.strategies:
            self.strategies[name].enabled = enabled
            self.save_config()
            logger.info(f"Strategy {name} {'enabled' if enabled else 'disabled'}")

    def set_strategy_markets(self, name: str, markets: list[str]):
        """Set which markets a strategy applies to"""
        if name in self.strategies:
            self.strategies[name].markets = [m.upper() for m in markets]
            self.save_config()
            logger.info(f"Strategy {name} markets set to: {markets}")

    def update_strategy_settings(self, name: str, settings: dict):
        """Update strategy settings"""
        if name in self.strategies:
            self.strategies[name].settings.update(settings)
            self.save_config()
            logger.info(f"Strategy {name} settings updated")

    # -------------------------------------------------------------------------
    # Copy Trading Specific
    # -------------------------------------------------------------------------

    def get_copy_traders(self) -> list[TraderConfig]:
        """Get all configured copy trading traders"""
        copy_strat = self.strategies.get("copy_trading")
        if not copy_strat:
            return []
        return copy_strat.traders

    def get_enabled_traders(self) -> list[TraderConfig]:
        """Get only enabled copy trading traders"""
        return [t for t in self.get_copy_traders() if t.enabled]

    def enable_trader(self, name: str, enabled: bool = True):
        """Enable or disable a specific trader"""
        copy_strat = self.strategies.get("copy_trading")
        if not copy_strat:
            return

        for trader in copy_strat.traders:
            if trader.name == name:
                trader.enabled = enabled
                self.save_config()
                logger.info(f"Trader {name} {'enabled' if enabled else 'disabled'}")
                return

    def add_trader(self, trader: TraderConfig):
        """Add a new trader to copy"""
        copy_strat = self.strategies.get("copy_trading")
        if not copy_strat:
            return

        # Check if trader already exists
        for t in copy_strat.traders:
            if t.address.lower() == trader.address.lower():
                logger.warning(f"Trader {trader.address} already exists")
                return

        copy_strat.traders.append(trader)
        self.save_config()
        logger.info(f"Added trader {trader.name}")

    def remove_trader(self, name: str):
        """Remove a trader from copy list"""
        copy_strat = self.strategies.get("copy_trading")
        if not copy_strat:
            return

        copy_strat.traders = [t for t in copy_strat.traders if t.name != name]
        self.save_config()
        logger.info(f"Removed trader {name}")

    # -------------------------------------------------------------------------
    # Position Sizing
    # -------------------------------------------------------------------------

    def calculate_position_size(
        self,
        strategy: str,
        whale_bet_size: float,
        account_balance: float,
        max_position: float,
    ) -> float:
        """
        Calculate position size for a trade based on strategy settings.

        For copy trading, scales the whale's bet size relative to account balance.
        For latency gap, uses a percentage of account balance.

        Args:
            strategy: Strategy name
            whale_bet_size: Size of the whale's bet (for copy trading)
            account_balance: Current account balance
            max_position: Maximum position size allowed

        Returns:
            Position size in USD
        """
        strat = self.strategies.get(strategy)
        if not strat:
            return min(max_position, 10.0)  # Default small position

        settings = strat.settings

        if strategy == "copy_trading":
            # Scale position based on account size relative to reference
            scale_to_account = settings.get("scale_to_account", True)
            if scale_to_account:
                reference_balance = settings.get("reference_balance", 10000)
                multiplier = settings.get("position_multiplier", 0.5)
                scale_factor = min(1.0, account_balance / reference_balance)

                position_size = whale_bet_size * scale_factor * multiplier
            else:
                # Use fixed multiplier of whale bet
                multiplier = settings.get("position_multiplier", 0.5)
                position_size = whale_bet_size * multiplier

            # Apply max position cap
            return min(position_size, max_position, settings.get("max_position_usd", max_position))

        elif strategy == "latency_gap":
            # Use percentage of account balance
            size_pct = settings.get("position_size_pct", 5.0) / 100
            position_size = account_balance * size_pct
            return min(position_size, max_position)

        elif strategy == "sniper":
            # Use percentage of account balance with max cap
            size_pct = settings.get("position_size_pct", 2.0) / 100
            max_usd = settings.get("max_position_usd", 100)
            position_size = account_balance * size_pct
            return min(position_size, max_position, max_usd)

        return min(max_position, 10.0)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "strategies": {
                name: strat.to_dict()
                for name, strat in self.strategies.items()
            },
            "global_settings": self.global_settings,
        }


# Singleton instance
_strategy_manager: Optional[StrategyManager] = None


def get_strategy_manager(config_path: str = "strategy_config.json") -> StrategyManager:
    """Get or create the strategy manager singleton"""
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = StrategyManager(config_path)
    return _strategy_manager
