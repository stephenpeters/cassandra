"""
Tests for kill switch and mode resolution.

Tests cover:
- get_effective_mode() returns correct mode
- Kill switch overrides everything
- Circuit breaker halts trading
- Alerts respect mode defaults
- Telegram commands work
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading import (
    LiveTradingEngine,
    LiveTradingConfig,
    TradingMode,
    CircuitBreaker,
)


class TestEffectiveMode:
    """Test get_effective_mode() returns correct state"""

    def test_paper_mode_returns_paper(self):
        """Paper mode with no overrides returns 'paper'"""
        config = LiveTradingConfig(mode=TradingMode.PAPER)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = False
        engine.circuit_breaker.triggered = False

        assert engine.get_effective_mode() == "paper"

    def test_live_mode_returns_live(self):
        """Live mode with no overrides returns 'live'"""
        config = LiveTradingConfig(mode=TradingMode.LIVE)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = False
        engine.circuit_breaker.triggered = False

        assert engine.get_effective_mode() == "live"

    def test_kill_switch_overrides_live(self):
        """Kill switch active returns 'killed' even in live mode"""
        config = LiveTradingConfig(mode=TradingMode.LIVE)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = True

        assert engine.get_effective_mode() == "killed"

    def test_kill_switch_overrides_paper(self):
        """Kill switch active returns 'killed' even in paper mode"""
        config = LiveTradingConfig(mode=TradingMode.PAPER)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = True

        assert engine.get_effective_mode() == "killed"

    def test_circuit_breaker_halts_trading(self):
        """Circuit breaker triggered returns 'halted'"""
        config = LiveTradingConfig(mode=TradingMode.LIVE)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = False
        engine.circuit_breaker.triggered = True

        assert engine.get_effective_mode() == "halted"

    def test_kill_switch_overrides_circuit_breaker(self):
        """Kill switch takes precedence over circuit breaker"""
        config = LiveTradingConfig(mode=TradingMode.LIVE)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = True
        engine.circuit_breaker.triggered = True

        # Kill switch wins
        assert engine.get_effective_mode() == "killed"


class TestCanExecuteTrade:
    """Test can_execute_trade() only returns True when safe to trade"""

    def test_can_trade_in_live_mode(self):
        """Can trade when in live mode with no overrides"""
        config = LiveTradingConfig(mode=TradingMode.LIVE)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = False
        engine.circuit_breaker.triggered = False

        assert engine.can_execute_trade() is True

    def test_cannot_trade_in_paper_mode(self):
        """Cannot execute real trades in paper mode"""
        config = LiveTradingConfig(mode=TradingMode.PAPER)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")

        assert engine.can_execute_trade() is False

    def test_cannot_trade_with_kill_switch(self):
        """Cannot trade when kill switch is active"""
        config = LiveTradingConfig(mode=TradingMode.LIVE)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = True

        assert engine.can_execute_trade() is False

    def test_cannot_trade_with_circuit_breaker(self):
        """Cannot trade when circuit breaker is triggered"""
        config = LiveTradingConfig(mode=TradingMode.LIVE)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.circuit_breaker.triggered = True

        assert engine.can_execute_trade() is False


class TestAlertsEnabled:
    """Test alert defaults by mode"""

    def test_paper_mode_alerts_off_by_default(self):
        """Paper mode has alerts OFF by default"""
        config = LiveTradingConfig(mode=TradingMode.PAPER, alerts_enabled=None)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")

        assert engine.are_alerts_enabled() is False

    def test_live_mode_alerts_on_by_default(self):
        """Live mode has alerts ON by default"""
        config = LiveTradingConfig(mode=TradingMode.LIVE, alerts_enabled=None)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")

        assert engine.are_alerts_enabled() is True

    def test_explicit_enable_overrides_paper_default(self):
        """Explicit True overrides paper mode default"""
        config = LiveTradingConfig(mode=TradingMode.PAPER, alerts_enabled=True)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")

        assert engine.are_alerts_enabled() is True

    def test_explicit_disable_overrides_live_default(self):
        """Explicit False overrides live mode default"""
        config = LiveTradingConfig(mode=TradingMode.LIVE, alerts_enabled=False)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")

        assert engine.are_alerts_enabled() is False


class TestKillSwitchActivation:
    """Test kill switch activation/deactivation"""

    @pytest.mark.asyncio
    async def test_activate_kill_switch(self):
        """Activating kill switch sets correct state"""
        config = LiveTradingConfig(mode=TradingMode.LIVE)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")

        # Mock _save_state to avoid file I/O
        engine._save_state = MagicMock()

        await engine.activate_kill_switch("Test activation")

        assert engine.kill_switch_active is True
        assert engine.circuit_breaker.triggered is True
        assert "KILL SWITCH" in engine.circuit_breaker.reason
        assert engine.get_effective_mode() == "killed"

    @pytest.mark.asyncio
    async def test_deactivate_kill_switch(self):
        """Deactivating kill switch clears state"""
        config = LiveTradingConfig(mode=TradingMode.LIVE)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = True
        engine.circuit_breaker.triggered = True

        # Mock _save_state to avoid file I/O
        engine._save_state = MagicMock()

        await engine.deactivate_kill_switch()

        assert engine.kill_switch_active is False
        assert engine.circuit_breaker.triggered is False
        assert engine.get_effective_mode() == "live"


class TestTelegramCommands:
    """Test Telegram command handling"""

    @pytest.mark.asyncio
    async def test_kill_command(self):
        """Test /kill command activates kill switch"""
        config = LiveTradingConfig(
            mode=TradingMode.LIVE,
            telegram_chat_id="12345"
        )
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine._save_state = MagicMock()

        response = await engine.handle_telegram_command("/kill", "12345")

        assert "KILL SWITCH ACTIVATED" in response
        assert engine.kill_switch_active is True

    @pytest.mark.asyncio
    async def test_status_command(self):
        """Test /status command returns status"""
        config = LiveTradingConfig(
            mode=TradingMode.PAPER,
            telegram_chat_id="12345"
        )
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.get_wallet_balance = MagicMock(return_value={"usdc_balance": 100.0})

        response = await engine.handle_telegram_command("/status", "12345")

        assert "Status:" in response
        assert "paper" in response.lower() or "PAPER" in response

    @pytest.mark.asyncio
    async def test_resume_command(self):
        """Test /resume command deactivates kill switch"""
        config = LiveTradingConfig(
            mode=TradingMode.LIVE,
            telegram_chat_id="12345"
        )
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = True
        engine._save_state = MagicMock()

        response = await engine.handle_telegram_command("/resume", "12345")

        assert "TRADING RESUMED" in response
        assert engine.kill_switch_active is False

    @pytest.mark.asyncio
    async def test_unauthorized_chat_rejected(self):
        """Commands from unauthorized chat are rejected"""
        config = LiveTradingConfig(
            mode=TradingMode.LIVE,
            telegram_chat_id="12345"
        )
        engine = LiveTradingEngine(config=config, data_dir="/tmp")

        response = await engine.handle_telegram_command("/kill", "99999")

        assert "Unauthorized" in response
        assert engine.kill_switch_active is False

    @pytest.mark.asyncio
    async def test_alerts_commands(self):
        """Test /alerts on and /alerts off commands"""
        config = LiveTradingConfig(
            mode=TradingMode.PAPER,
            telegram_chat_id="12345"
        )
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine._save_state = MagicMock()

        # Enable alerts
        response = await engine.handle_telegram_command("/alerts on", "12345")
        assert "ENABLED" in response
        assert engine.are_alerts_enabled() is True

        # Disable alerts
        response = await engine.handle_telegram_command("/alerts off", "12345")
        assert "DISABLED" in response
        assert engine.are_alerts_enabled() is False


class TestModeTransitions:
    """Test mode transitions are safe"""

    @patch("asyncio.create_task")
    def test_set_mode_updates_config(self, mock_create_task):
        """set_mode updates the config mode when credentials are available"""
        config = LiveTradingConfig(mode=TradingMode.PAPER)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine._save_state = MagicMock()
        # Mock credentials so live mode can be enabled
        engine.private_key = "0x" + "1" * 64
        engine._init_clob_client = MagicMock()

        engine.set_mode("live")

        assert engine.config.mode == TradingMode.LIVE

    def test_set_mode_blocked_without_credentials(self):
        """set_mode to live blocked without private key"""
        config = LiveTradingConfig(mode=TradingMode.PAPER)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine._save_state = MagicMock()
        # No private key - should block live mode

        engine.set_mode("live")

        # Mode should remain paper since credentials are missing
        assert engine.config.mode == TradingMode.PAPER

    @patch("asyncio.create_task")
    def test_mode_change_with_kill_switch_still_killed(self, mock_create_task):
        """Changing mode doesn't override kill switch"""
        config = LiveTradingConfig(mode=TradingMode.PAPER)
        engine = LiveTradingEngine(config=config, data_dir="/tmp")
        engine.kill_switch_active = True
        engine._save_state = MagicMock()
        # Mock credentials so live mode can be enabled
        engine.private_key = "0x" + "1" * 64
        engine._init_clob_client = MagicMock()

        engine.set_mode("live")

        # Mode is live but effective mode is still killed
        assert engine.config.mode == TradingMode.LIVE
        assert engine.get_effective_mode() == "killed"
        assert engine.can_execute_trade() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
