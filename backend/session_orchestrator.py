"""
Session Orchestrator - Manages trading sessions and routes events to strategies.

This is the central coordinator that:
1. Manages one active trading session at a time
2. Routes MarketEvents to the active strategy
3. Handles signal execution through the appropriate executor
4. Broadcasts status updates to the frontend
"""

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Awaitable, Any

from strategy_base import Strategy, MarketEvent, Signal, MarketType
from mode_controller import get_mode_controller, TradingMode

logger = logging.getLogger(__name__)


@dataclass
class TradingSession:
    """User's active trading configuration."""
    id: str                        # Unique session ID
    market_type: MarketType        # FIFTEEN_MIN, ONE_HOUR, etc.
    currency: str                  # "BTC", "ETH", "SOL"
    strategy_name: str             # "sniper", "dip_arb", etc.
    created_at: int
    active: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "market_type": self.market_type.value,
            "currency": self.currency,
            "strategy_name": self.strategy_name,
            "created_at": self.created_at,
            "active": self.active,
        }


@dataclass
class ExecutionResult:
    """Result of attempting to execute a signal."""
    success: bool
    order_id: Optional[str]
    error: Optional[str]
    mode: str  # "paper" or "live"
    size_usd: float
    filled_price: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


class SessionOrchestrator:
    """
    Manages ONE active trading session.
    No hidden logic - explicit control flow.
    """

    def __init__(self):
        self.session: Optional[TradingSession] = None
        self.strategy: Optional[Strategy] = None
        self._position_taken_this_window: bool = False
        self._current_market_start: Optional[int] = None
        self._on_signal: Optional[Callable[[Signal, ExecutionResult], Awaitable[None]]] = None
        self._on_status: Optional[Callable[[dict], Awaitable[None]]] = None

        # Executor will be set externally (dependency injection)
        self._paper_executor: Optional[Any] = None
        self._live_executor: Optional[Any] = None

    def set_executors(self, paper_executor: Any, live_executor: Any) -> None:
        """Set the executors for paper and live trading."""
        self._paper_executor = paper_executor
        self._live_executor = live_executor

    def set_callbacks(
        self,
        on_signal: Optional[Callable[[Signal, ExecutionResult], Awaitable[None]]] = None,
        on_status: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> None:
        """Set callbacks for signal execution and status updates."""
        self._on_signal = on_signal
        self._on_status = on_status

    def start_session(
        self,
        session_id: str,
        market_type: MarketType,
        currency: str,
        strategy: Strategy,
    ) -> TradingSession:
        """
        Start a new trading session.

        Args:
            session_id: Unique identifier for this session
            market_type: Which market timeframe to trade
            currency: Which currency to trade (BTC, ETH, SOL)
            strategy: Strategy instance to use

        Returns:
            The created TradingSession
        """
        # Stop any existing session
        if self.session:
            self.stop_session()

        self.session = TradingSession(
            id=session_id,
            market_type=market_type,
            currency=currency,
            strategy_name=strategy.name,
            created_at=int(time.time()),
            active=True,
        )
        self.strategy = strategy
        self._position_taken_this_window = False
        self._current_market_start = None

        logger.info(
            f"Session started: {session_id} - {currency} {market_type.value} with {strategy.name}"
        )
        return self.session

    def stop_session(self) -> None:
        """Stop the current session."""
        if self.session:
            logger.info(f"Session stopped: {self.session.id}")
            self.session.active = False
            self.session = None
        self.strategy = None
        self._position_taken_this_window = False
        self._current_market_start = None

    async def process_tick(self, event: MarketEvent) -> Optional[Signal]:
        """
        Process one market tick - called from main loop.

        Returns the signal if one was generated and executed.
        """
        mode_controller = get_mode_controller()

        # No active session?
        if not self.session or not self.session.active:
            return None

        # Trading disabled (OFF mode)?
        if not mode_controller.is_trading_enabled:
            return None

        # Wrong currency?
        if event.symbol != self.session.currency:
            return None

        # Wrong market type?
        if event.market_type != self.session.market_type:
            return None

        # Detect new market window
        if self._current_market_start != event.market_start:
            self._current_market_start = event.market_start
            self._position_taken_this_window = False
            self.strategy.on_market_open(event)

        # Already have position for this window?
        if self._position_taken_this_window:
            return None

        # Get signal from strategy
        signal = self.strategy.on_tick(event)

        if signal:
            # Execute the signal
            result = await self._execute_signal(signal)

            if result.success:
                self._position_taken_this_window = True
                self.strategy.on_position_filled(
                    signal.symbol,
                    signal.side,
                    signal.market_start
                )

            # Notify callback if set
            if self._on_signal:
                try:
                    await self._on_signal(signal, result)
                except Exception as e:
                    logger.error(f"Error in signal callback: {e}")

            return signal

        return None

    async def _execute_signal(self, signal: Signal) -> ExecutionResult:
        """Execute a signal through the appropriate executor."""
        mode_controller = get_mode_controller()

        # Select executor based on mode
        if mode_controller.is_live:
            executor = self._live_executor
            mode = "live"
        else:
            executor = self._paper_executor
            mode = "paper"

        if not executor:
            logger.error(f"No {mode} executor configured")
            return ExecutionResult(
                success=False,
                order_id=None,
                error=f"No {mode} executor configured",
                mode=mode,
                size_usd=0,
                filled_price=None,
            )

        try:
            result = await executor.execute(signal)
            return result
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                error=str(e),
                mode=mode,
                size_usd=0,
                filled_price=None,
            )

    def on_market_close(self, event: MarketEvent, resolution: str) -> None:
        """Market window closed."""
        if not self.session or event.symbol != self.session.currency:
            return
        if self.strategy:
            self.strategy.on_market_close(event, resolution)

    def get_status(self, event: Optional[MarketEvent] = None) -> dict:
        """Get current orchestrator status."""
        mode_controller = get_mode_controller()

        status = {
            "has_session": self.session is not None,
            "mode": mode_controller.mode.value,
            "is_trading_enabled": mode_controller.is_trading_enabled,
            "position_taken_this_window": self._position_taken_this_window,
        }

        if self.session:
            status["session"] = self.session.to_dict()

        if self.strategy and event:
            status["strategy_status"] = self.strategy.get_status(event)

        return status


# Global singleton instance
_orchestrator: Optional[SessionOrchestrator] = None


def get_orchestrator() -> SessionOrchestrator:
    """Get the global session orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SessionOrchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset the global orchestrator (for testing)."""
    global _orchestrator
    _orchestrator = None
