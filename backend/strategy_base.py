"""
Strategy Base Classes

Provides a standardized interface for all trading strategies.
Strategies ONLY produce signals - they never execute trades directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Literal, Optional, Any
import time


class MarketType(Enum):
    """Market timeframe types"""
    FIFTEEN_MIN = "15m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    DAILY = "1d"


@dataclass
class MarketEvent:
    """Standardized event passed to all strategies."""
    symbol: str                    # "BTC", "ETH", etc.
    market_type: MarketType        # FIFTEEN_MIN, ONE_HOUR, etc.
    market_start: int              # Unix timestamp
    market_end: int                # Unix timestamp
    elapsed_sec: int
    remaining_sec: int
    up_price: float                # Polymarket UP token price
    down_price: float              # Polymarket DOWN token price (usually 1 - up_price)
    binance_price: float           # Current Binance spot price
    binance_open: Optional[float]  # Binance price at market open
    timestamp: int

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "market_type": self.market_type.value,
        }


@dataclass
class Signal:
    """Output from strategy - pure data, no execution logic."""
    strategy: str                  # Strategy name (e.g., "sniper")
    symbol: str                    # "BTC", "ETH", etc.
    side: Literal["UP", "DOWN"]    # Which outcome to buy
    entry_price: float             # Price to buy at
    reason: str                    # Human-readable reason
    confidence: float              # 0.0 - 1.0
    metadata: dict                 # Strategy-specific data (e.g., EV, win rate)
    market_start: int
    market_end: int
    timestamp: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StrategyRecommendation:
    """Why a strategy is/isn't recommended for a market type."""
    market_type: MarketType
    recommended: bool
    reason: str  # Human-readable explanation

    def to_dict(self) -> dict:
        return {
            "market_type": self.market_type.value,
            "recommended": self.recommended,
            "reason": self.reason,
        }


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Strategies ONLY produce signals, never execute.
    Execution is handled by the Executor layer.
    """

    name: str = "base"
    description: str = "Base strategy class"

    def __init__(self, config: dict, market_type: Optional[MarketType] = None):
        self.config = config
        self.market_type = market_type or MarketType.FIFTEEN_MIN
        self._positions_taken: set[str] = set()  # "SYMBOL_market_start" keys

    @classmethod
    @abstractmethod
    def get_recommendations(cls) -> list[StrategyRecommendation]:
        """Return recommendations for each market type."""
        pass

    @abstractmethod
    def on_tick(self, event: MarketEvent) -> Optional[Signal]:
        """
        Called every tick. Return Signal or None.

        This is the main entry point - called every second.
        """
        pass

    @abstractmethod
    def get_status(self, event: MarketEvent) -> dict:
        """
        Return current status for UI display.

        Called frequently to update the UI with strategy state.
        """
        pass

    def on_market_open(self, event: MarketEvent) -> None:
        """Called when a new market window opens."""
        pass

    def on_market_close(self, event: MarketEvent, resolution: str) -> None:
        """Called when a market window closes/resolves."""
        pass

    def on_position_filled(self, symbol: str, side: str, market_start: int) -> None:
        """Called when our signal resulted in a filled position."""
        key = f"{symbol}_{market_start}"
        self._positions_taken.add(key)

    def has_position(self, symbol: str, market_start: int) -> bool:
        """Check if we already have a position for this market window."""
        key = f"{symbol}_{market_start}"
        return key in self._positions_taken

    def reset_positions(self) -> None:
        """Clear all position tracking."""
        self._positions_taken.clear()
