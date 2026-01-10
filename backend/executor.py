"""
Executor Module - Handles trade execution for Paper and Live modes.

Executors receive Signals from strategies and execute them.
Paper mode simulates trades, Live mode executes real trades on Polymarket.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Any

from strategy_base import Signal
from mode_controller import get_mode_controller, TradingMode

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of attempting to execute a signal."""
    success: bool
    order_id: Optional[str]
    error: Optional[str]
    mode: str  # "paper" or "live"
    size_usd: float
    filled_price: Optional[float]
    timestamp: int

    def to_dict(self) -> dict:
        return asdict(self)


class Executor(ABC):
    """Base class for trade executors."""

    @abstractmethod
    async def execute(self, signal: Signal) -> ExecutionResult:
        """Execute a trading signal. Returns result."""
        pass

    @abstractmethod
    def get_position_size(self, symbol: str) -> float:
        """Get the position size in USD for a symbol."""
        pass


class PaperExecutor(Executor):
    """
    Simulated execution - no real trades.

    Uses the TradingEngine for paper trading simulation.
    """

    def __init__(self, trading_engine: Any = None):
        """
        Initialize paper executor.

        Args:
            trading_engine: The TradingEngine instance for paper trading.
                           If None, will try to import and get global instance.
        """
        self._engine = trading_engine
        self._default_position_size = 10.0  # $10 default

    def set_engine(self, engine: Any) -> None:
        """Set the trading engine after construction."""
        self._engine = engine

    def get_position_size(self, symbol: str) -> float:
        """Get position size from engine config or use default."""
        if self._engine and hasattr(self._engine, 'position_size'):
            return self._engine.position_size
        return self._default_position_size

    async def execute(self, signal: Signal) -> ExecutionResult:
        """Execute signal in paper mode."""
        if not self._engine:
            logger.error("PaperExecutor: No trading engine configured")
            return ExecutionResult(
                success=False,
                order_id=None,
                error="No trading engine configured",
                mode="paper",
                size_usd=0,
                filled_price=None,
                timestamp=int(time.time()),
            )

        try:
            # Use the trading engine to open a paper position
            position = self._engine.open_position(
                symbol=signal.symbol,
                side=signal.side,
                entry_price=signal.entry_price,
                checkpoint=signal.strategy,
                market_start=signal.market_start,
                market_end=signal.market_end,
            )

            if position:
                logger.info(
                    f"Paper trade executed: {signal.side} {signal.symbol} @ {signal.entry_price:.4f}"
                )
                return ExecutionResult(
                    success=True,
                    order_id=position.id if hasattr(position, 'id') else str(uuid.uuid4()),
                    error=None,
                    mode="paper",
                    size_usd=position.cost_basis if hasattr(position, 'cost_basis') else self._default_position_size,
                    filled_price=signal.entry_price,
                    timestamp=int(time.time()),
                )
            else:
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    error="Failed to open paper position",
                    mode="paper",
                    size_usd=0,
                    filled_price=None,
                    timestamp=int(time.time()),
                )

        except Exception as e:
            logger.error(f"Paper execution error: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                error=str(e),
                mode="paper",
                size_usd=0,
                filled_price=None,
                timestamp=int(time.time()),
            )


class LiveExecutor(Executor):
    """
    Real execution on Polymarket.

    IMPORTANT: Always checks mode controller before executing.
    OFF mode = instant rejection (kill switch).
    """

    def __init__(self, live_engine: Any = None):
        """
        Initialize live executor.

        Args:
            live_engine: The LiveTradingEngine instance for real trades.
        """
        self._engine = live_engine
        self._default_position_size = 10.0  # $10 default

    def set_engine(self, engine: Any) -> None:
        """Set the live trading engine after construction."""
        self._engine = engine

    def get_position_size(self, symbol: str) -> float:
        """Get position size from engine config or use default."""
        if self._engine and hasattr(self._engine, 'position_size'):
            return self._engine.position_size
        return self._default_position_size

    async def execute(self, signal: Signal) -> ExecutionResult:
        """
        Execute signal in live mode.

        CRITICAL: Always checks mode controller first.
        """
        mode_controller = get_mode_controller()

        # Check kill switch FIRST
        if mode_controller.is_off:
            logger.warning("Live execution blocked: Kill switch active (mode=OFF)")
            return ExecutionResult(
                success=False,
                order_id=None,
                error="Kill switch active (mode=OFF)",
                mode="live",
                size_usd=0,
                filled_price=None,
                timestamp=int(time.time()),
            )

        # Double-check we're in LIVE mode
        if not mode_controller.is_live:
            logger.warning(f"Live execution blocked: Mode is {mode_controller.mode.value}, not LIVE")
            return ExecutionResult(
                success=False,
                order_id=None,
                error=f"Mode is {mode_controller.mode.value}, not LIVE",
                mode="live",
                size_usd=0,
                filled_price=None,
                timestamp=int(time.time()),
            )

        if not self._engine:
            logger.error("LiveExecutor: No live trading engine configured")
            return ExecutionResult(
                success=False,
                order_id=None,
                error="No live trading engine configured",
                mode="live",
                size_usd=0,
                filled_price=None,
                timestamp=int(time.time()),
            )

        try:
            # Execute real order through the live engine
            # The live engine interface should match this pattern
            order = await self._engine.place_order(
                symbol=signal.symbol,
                side=signal.side,
                price=signal.entry_price,
                market_start=signal.market_start,
                market_end=signal.market_end,
                reason=signal.reason,
            )

            if order and order.get("status") == "filled":
                logger.info(
                    f"LIVE trade executed: {signal.side} {signal.symbol} @ {order.get('filled_price', signal.entry_price):.4f}"
                )
                return ExecutionResult(
                    success=True,
                    order_id=order.get("id"),
                    error=None,
                    mode="live",
                    size_usd=order.get("size_usd", self._default_position_size),
                    filled_price=order.get("filled_price"),
                    timestamp=int(time.time()),
                )
            else:
                error = order.get("error", "Order not filled") if order else "No order response"
                return ExecutionResult(
                    success=False,
                    order_id=order.get("id") if order else None,
                    error=error,
                    mode="live",
                    size_usd=0,
                    filled_price=None,
                    timestamp=int(time.time()),
                )

        except Exception as e:
            logger.error(f"Live execution error: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                error=str(e),
                mode="live",
                size_usd=0,
                filled_price=None,
                timestamp=int(time.time()),
            )


# Factory function for creating executors
def create_executors(paper_engine: Any = None, live_engine: Any = None) -> tuple[PaperExecutor, LiveExecutor]:
    """
    Create both paper and live executors.

    Args:
        paper_engine: TradingEngine instance for paper trading
        live_engine: LiveTradingEngine instance for live trading

    Returns:
        Tuple of (PaperExecutor, LiveExecutor)
    """
    paper = PaperExecutor(paper_engine)
    live = LiveExecutor(live_engine)
    return paper, live
