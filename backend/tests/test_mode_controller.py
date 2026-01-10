"""
Tests for Mode Controller - unified LIVE/PAPER/OFF mode management.

Tests cover:
- Default mode initialization
- Mode transitions (OFF -> PAPER -> LIVE)
- Mode properties (is_trading_enabled, is_live, is_paper, is_off)
- Mode change callbacks
- Kill switch functionality
- Status reporting
"""
import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mode_controller import (
    ModeController,
    TradingMode,
    get_mode_controller,
    reset_mode_controller,
)


class TestTradingMode:
    """Tests for TradingMode enum."""

    def test_mode_values(self):
        """Test mode enum values."""
        assert TradingMode.LIVE.value == "live"
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.OFF.value == "off"

    def test_mode_from_string(self):
        """Test converting strings to TradingMode."""
        assert ModeController.from_string("live") == TradingMode.LIVE
        assert ModeController.from_string("LIVE") == TradingMode.LIVE
        assert ModeController.from_string("paper") == TradingMode.PAPER
        assert ModeController.from_string("PAPER") == TradingMode.PAPER
        assert ModeController.from_string("off") == TradingMode.OFF
        assert ModeController.from_string("OFF") == TradingMode.OFF
        assert ModeController.from_string("invalid") == TradingMode.OFF  # Safe default


class TestModeControllerInit:
    """Tests for ModeController initialization."""

    def test_default_mode_is_off(self):
        """Default mode should be OFF for safety."""
        controller = ModeController()
        assert controller.mode == TradingMode.OFF
        assert controller.is_off is True
        assert controller.is_trading_enabled is False

    def test_changed_by_defaults_to_system(self):
        """Initial changed_by should be 'system'."""
        controller = ModeController()
        status = controller.get_status()
        assert status["changed_by"] == "system"


class TestModeControllerProperties:
    """Tests for ModeController properties."""

    @pytest.fixture
    def controller(self):
        """Create a fresh controller for each test."""
        return ModeController()

    @pytest.mark.asyncio
    async def test_is_trading_enabled_off(self, controller):
        """is_trading_enabled should be False when OFF."""
        assert controller.is_trading_enabled is False

    @pytest.mark.asyncio
    async def test_is_trading_enabled_paper(self, controller):
        """is_trading_enabled should be True when PAPER."""
        await controller.set_mode(TradingMode.PAPER)
        assert controller.is_trading_enabled is True

    @pytest.mark.asyncio
    async def test_is_trading_enabled_live(self, controller):
        """is_trading_enabled should be True when LIVE."""
        await controller.set_mode(TradingMode.LIVE)
        assert controller.is_trading_enabled is True

    @pytest.mark.asyncio
    async def test_is_live_property(self, controller):
        """is_live should only be True when LIVE."""
        assert controller.is_live is False
        await controller.set_mode(TradingMode.PAPER)
        assert controller.is_live is False
        await controller.set_mode(TradingMode.LIVE)
        assert controller.is_live is True

    @pytest.mark.asyncio
    async def test_is_paper_property(self, controller):
        """is_paper should only be True when PAPER."""
        assert controller.is_paper is False
        await controller.set_mode(TradingMode.PAPER)
        assert controller.is_paper is True
        await controller.set_mode(TradingMode.LIVE)
        assert controller.is_paper is False

    @pytest.mark.asyncio
    async def test_is_off_property(self, controller):
        """is_off should only be True when OFF."""
        assert controller.is_off is True
        await controller.set_mode(TradingMode.PAPER)
        assert controller.is_off is False
        await controller.set_mode(TradingMode.LIVE)
        assert controller.is_off is False


class TestModeControllerTransitions:
    """Tests for mode transitions."""

    @pytest.fixture
    def controller(self):
        """Create a fresh controller for each test."""
        return ModeController()

    @pytest.mark.asyncio
    async def test_set_mode_returns_true_on_change(self, controller):
        """set_mode should return True when mode actually changes."""
        result = await controller.set_mode(TradingMode.PAPER)
        assert result is True

    @pytest.mark.asyncio
    async def test_set_mode_returns_false_when_same(self, controller):
        """set_mode should return False when mode is already set."""
        await controller.set_mode(TradingMode.PAPER)
        result = await controller.set_mode(TradingMode.PAPER)
        assert result is False

    @pytest.mark.asyncio
    async def test_off_to_paper_transition(self, controller):
        """Should transition from OFF to PAPER."""
        assert controller.mode == TradingMode.OFF
        await controller.set_mode(TradingMode.PAPER)
        assert controller.mode == TradingMode.PAPER

    @pytest.mark.asyncio
    async def test_paper_to_live_transition(self, controller):
        """Should transition from PAPER to LIVE."""
        await controller.set_mode(TradingMode.PAPER)
        await controller.set_mode(TradingMode.LIVE)
        assert controller.mode == TradingMode.LIVE

    @pytest.mark.asyncio
    async def test_live_to_off_transition(self, controller):
        """Should transition from LIVE to OFF (kill switch)."""
        await controller.set_mode(TradingMode.LIVE)
        await controller.set_mode(TradingMode.OFF)
        assert controller.mode == TradingMode.OFF

    @pytest.mark.asyncio
    async def test_changed_by_is_recorded(self, controller):
        """changed_by should be recorded on mode change."""
        await controller.set_mode(TradingMode.PAPER, changed_by="test_user")
        status = controller.get_status()
        assert status["changed_by"] == "test_user"

    @pytest.mark.asyncio
    async def test_changed_at_is_updated(self, controller):
        """changed_at timestamp should be updated on mode change."""
        status_before = controller.get_status()
        await asyncio.sleep(0.01)  # Small delay
        await controller.set_mode(TradingMode.PAPER)
        status_after = controller.get_status()
        assert status_after["changed_at"] >= status_before["changed_at"]


class TestModeControllerKillSwitch:
    """Tests for kill switch functionality."""

    @pytest.fixture
    def controller(self):
        """Create a fresh controller for each test."""
        return ModeController()

    @pytest.mark.asyncio
    async def test_kill_sets_mode_to_off(self, controller):
        """kill() should set mode to OFF."""
        await controller.set_mode(TradingMode.LIVE)
        await controller.kill()
        assert controller.mode == TradingMode.OFF

    @pytest.mark.asyncio
    async def test_kill_records_reason(self, controller):
        """kill() should record the reason in changed_by."""
        await controller.set_mode(TradingMode.LIVE)
        await controller.kill(reason="emergency")
        status = controller.get_status()
        assert "kill:emergency" in status["changed_by"]

    @pytest.mark.asyncio
    async def test_kill_from_paper_mode(self, controller):
        """kill() should work from PAPER mode too."""
        await controller.set_mode(TradingMode.PAPER)
        await controller.kill(reason="user_request")
        assert controller.mode == TradingMode.OFF


class TestModeControllerCallback:
    """Tests for mode change callbacks."""

    @pytest.fixture
    def controller(self):
        """Create a fresh controller for each test."""
        return ModeController()

    @pytest.mark.asyncio
    async def test_callback_is_called_on_mode_change(self, controller):
        """Callback should be called when mode changes."""
        callback_called = False
        callback_args = {}

        async def test_callback(old_mode, new_mode, changed_by):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = {
                "old_mode": old_mode,
                "new_mode": new_mode,
                "changed_by": changed_by,
            }

        controller.set_on_mode_change(test_callback)
        await controller.set_mode(TradingMode.PAPER, changed_by="test")

        assert callback_called is True
        assert callback_args["old_mode"] == TradingMode.OFF
        assert callback_args["new_mode"] == TradingMode.PAPER
        assert callback_args["changed_by"] == "test"

    @pytest.mark.asyncio
    async def test_callback_not_called_when_mode_unchanged(self, controller):
        """Callback should NOT be called when mode is already set."""
        callback_called = False

        async def test_callback(old_mode, new_mode, changed_by):
            nonlocal callback_called
            callback_called = True

        controller.set_on_mode_change(test_callback)
        await controller.set_mode(TradingMode.PAPER)
        callback_called = False  # Reset after first call

        await controller.set_mode(TradingMode.PAPER)  # Same mode
        assert callback_called is False


class TestModeControllerStatus:
    """Tests for status reporting."""

    @pytest.fixture
    def controller(self):
        """Create a fresh controller for each test."""
        return ModeController()

    def test_get_status_returns_dict(self, controller):
        """get_status should return a dict with required fields."""
        status = controller.get_status()
        assert isinstance(status, dict)
        assert "mode" in status
        assert "is_trading_enabled" in status
        assert "is_live" in status
        assert "changed_at" in status
        assert "changed_by" in status

    @pytest.mark.asyncio
    async def test_status_reflects_current_mode(self, controller):
        """Status should reflect current mode."""
        status_off = controller.get_status()
        assert status_off["mode"] == "off"
        assert status_off["is_trading_enabled"] is False

        await controller.set_mode(TradingMode.PAPER)
        status_paper = controller.get_status()
        assert status_paper["mode"] == "paper"
        assert status_paper["is_trading_enabled"] is True
        assert status_paper["is_live"] is False

        await controller.set_mode(TradingMode.LIVE)
        status_live = controller.get_status()
        assert status_live["mode"] == "live"
        assert status_live["is_live"] is True


class TestModeControllerSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_mode_controller()

    def test_get_mode_controller_returns_same_instance(self):
        """get_mode_controller should return the same instance."""
        controller1 = get_mode_controller()
        controller2 = get_mode_controller()
        assert controller1 is controller2

    @pytest.mark.asyncio
    async def test_singleton_preserves_state(self):
        """Singleton should preserve state across calls."""
        controller1 = get_mode_controller()
        await controller1.set_mode(TradingMode.PAPER)

        controller2 = get_mode_controller()
        assert controller2.mode == TradingMode.PAPER

    def test_reset_clears_singleton(self):
        """reset_mode_controller should clear the singleton."""
        controller1 = get_mode_controller()
        reset_mode_controller()
        controller2 = get_mode_controller()
        assert controller1 is not controller2


class TestModeControllerIntegration:
    """Integration tests for realistic scenarios."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_mode_controller()

    @pytest.mark.asyncio
    async def test_startup_flow(self):
        """Test typical startup flow: OFF -> PAPER."""
        controller = get_mode_controller()

        # Default is OFF
        assert controller.mode == TradingMode.OFF

        # Server startup sets to PAPER from env
        await controller.set_mode(TradingMode.PAPER, changed_by="startup")
        assert controller.mode == TradingMode.PAPER

        status = controller.get_status()
        assert status["changed_by"] == "startup"

    @pytest.mark.asyncio
    async def test_user_enables_live_trading(self):
        """Test user enabling live trading via UI."""
        controller = get_mode_controller()
        await controller.set_mode(TradingMode.PAPER, changed_by="startup")

        # User clicks LIVE in UI
        await controller.set_mode(TradingMode.LIVE, changed_by="api")
        assert controller.is_live is True

    @pytest.mark.asyncio
    async def test_emergency_kill_switch(self):
        """Test emergency kill switch activation."""
        controller = get_mode_controller()
        await controller.set_mode(TradingMode.LIVE, changed_by="startup")

        # Emergency! Kill trading
        await controller.kill(reason="circuit_breaker")
        assert controller.mode == TradingMode.OFF
        assert controller.is_trading_enabled is False

    @pytest.mark.asyncio
    async def test_telegram_kill_command(self):
        """Test Telegram /kill command."""
        controller = get_mode_controller()
        await controller.set_mode(TradingMode.LIVE, changed_by="startup")

        # Telegram bot sends /kill
        await controller.kill(reason="telegram")
        status = controller.get_status()
        assert status["mode"] == "off"
        assert "telegram" in status["changed_by"]
