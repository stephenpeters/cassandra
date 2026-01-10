"""
Mode Controller - Unified LIVE/PAPER/OFF mode management

Single source of truth for trading mode.
OFF mode = kill switch (stops all trading instantly).
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading execution modes"""
    LIVE = "live"      # Real trades on Polymarket
    PAPER = "paper"    # Simulated trades
    OFF = "off"        # Kill switch - no trading


class ModeController:
    """
    Single source of truth for trading mode.
    OFF = kill switch (stops all trading instantly).
    """

    def __init__(self):
        self._mode = TradingMode.OFF  # Safe default
        self._changed_at: int = int(time.time())
        self._changed_by: str = "system"
        self._on_mode_change: Optional[Callable[[TradingMode, TradingMode, str], Awaitable[None]]] = None

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def is_trading_enabled(self) -> bool:
        """Returns True if trading is allowed (LIVE or PAPER)"""
        return self._mode != TradingMode.OFF

    @property
    def is_live(self) -> bool:
        """Returns True if in live trading mode"""
        return self._mode == TradingMode.LIVE

    @property
    def is_paper(self) -> bool:
        """Returns True if in paper trading mode"""
        return self._mode == TradingMode.PAPER

    @property
    def is_off(self) -> bool:
        """Returns True if trading is disabled (kill switch)"""
        return self._mode == TradingMode.OFF

    def set_on_mode_change(self, callback: Callable[[TradingMode, TradingMode, str], Awaitable[None]]) -> None:
        """Set callback for mode changes. Callback receives (old_mode, new_mode, changed_by)."""
        self._on_mode_change = callback

    async def set_mode(self, mode: TradingMode, changed_by: str = "user") -> bool:
        """
        Set the trading mode.

        Args:
            mode: The new mode to set
            changed_by: Who initiated the change (user, telegram, circuit_breaker, etc.)

        Returns:
            True if mode was changed, False if already in that mode
        """
        old_mode = self._mode

        if old_mode == mode:
            return False

        self._mode = mode
        self._changed_at = int(time.time())
        self._changed_by = changed_by

        logger.info(f"Mode changed: {old_mode.value} -> {mode.value} by {changed_by}")

        # Notify callback if set
        if self._on_mode_change:
            try:
                await self._on_mode_change(old_mode, mode, changed_by)
            except Exception as e:
                logger.error(f"Error in mode change callback: {e}")

        return True

    async def kill(self, reason: str = "manual") -> None:
        """
        Emergency stop - set mode to OFF.

        Args:
            reason: Why the kill switch was activated
        """
        await self.set_mode(TradingMode.OFF, changed_by=f"kill:{reason}")

    def get_status(self) -> dict:
        """Get current mode status as dict."""
        return {
            "mode": self._mode.value,
            "is_trading_enabled": self.is_trading_enabled,
            "is_live": self.is_live,
            "changed_at": self._changed_at,
            "changed_by": self._changed_by,
        }

    @classmethod
    def from_string(cls, mode_str: str) -> TradingMode:
        """Convert string to TradingMode enum."""
        mode_str = mode_str.lower().strip()
        if mode_str == "live":
            return TradingMode.LIVE
        elif mode_str == "paper":
            return TradingMode.PAPER
        else:
            return TradingMode.OFF


# Global singleton instance
_mode_controller: Optional[ModeController] = None


def get_mode_controller() -> ModeController:
    """Get the global mode controller instance."""
    global _mode_controller
    if _mode_controller is None:
        _mode_controller = ModeController()
    return _mode_controller


def reset_mode_controller() -> None:
    """Reset the global mode controller (for testing)."""
    global _mode_controller
    _mode_controller = None
