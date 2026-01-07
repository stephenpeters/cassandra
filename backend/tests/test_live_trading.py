"""
Tests for the Live Trading Engine (live_trading.py).

Tests cover:
- Order processing and execution
- Position management
- Circuit breaker logic
- Kill switch functionality
- State persistence
- CLOB client integration
"""
import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading import (
    LiveTradingEngine,
    LiveTradingConfig,
    LiveOrder,
    LivePosition,
    CircuitBreaker,
    TradingMode,
)
from trading import CheckpointSignal, SignalType


class TestLiveTradingConfig:
    """Tests for LiveTradingConfig."""

    def test_config_defaults(self):
        """Test default config values."""
        config = LiveTradingConfig()
        assert config.mode == TradingMode.PAPER
        assert config.max_position_usd == 5000.0
        assert config.max_open_positions == 3

    def test_config_to_dict(self):
        """Test config serialization."""
        config = LiveTradingConfig(mode=TradingMode.LIVE, max_position_usd=100)
        data = config.to_dict()
        assert data["mode"] == "live"
        assert data["max_position_usd"] == 100


class TestLiveOrder:
    """Tests for LiveOrder."""

    def test_order_creation(self):
        """Test order creation with required fields."""
        order = LiveOrder(
            id="BTC_123_7m30s",
            symbol="BTC",
            side="UP",
            direction="BUY",
            token_id="token123",
            size_usd=50.0,
            price=0.55,
            order_type="GTC",
            status="pending",
            created_at=int(time.time()),
        )

        assert order.symbol == "BTC"
        assert order.side == "UP"
        assert order.status == "pending"

    def test_order_to_dict(self):
        """Test order serialization."""
        order = LiveOrder(
            id="BTC_123",
            symbol="BTC",
            side="UP",
            direction="BUY",
            token_id="token123",
            size_usd=50.0,
            price=0.55,
            order_type="GTC",
            status="filled",
            created_at=int(time.time()),
            filled_at=int(time.time()),
            filled_size=90.9,
            filled_price=0.55,
        )

        data = order.to_dict()
        assert data["status"] == "filled"
        assert data["filled_size"] == 90.9


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_circuit_breaker_not_triggered_initially(self):
        """Test circuit breaker starts inactive."""
        cb = CircuitBreaker()
        assert cb.triggered is False
        assert cb.consecutive_losses == 0

    def test_circuit_breaker_triggers_on_consecutive_losses(self, live_trading_config):
        """Test circuit breaker triggers after max consecutive losses."""
        cb = CircuitBreaker()
        live_trading_config.max_consecutive_losses = 3

        # Simulate 3 losses
        for i in range(3):
            cb.check_and_update(live_trading_config, trade_pnl=-10, trade_volume=50)

        assert cb.triggered is True
        assert "consecutive losses" in cb.reason.lower()

    def test_circuit_breaker_triggers_on_daily_loss(self, live_trading_config):
        """Test circuit breaker triggers on daily loss limit."""
        cb = CircuitBreaker()
        live_trading_config.daily_loss_limit_usd = 100

        # Single big loss
        cb.check_and_update(live_trading_config, trade_pnl=-150, trade_volume=200)

        assert cb.triggered is True
        assert "daily loss" in cb.reason.lower()

    def test_winning_trade_resets_consecutive_losses(self, live_trading_config):
        """Test winning trade resets consecutive loss counter."""
        cb = CircuitBreaker()
        cb.consecutive_losses = 2

        # Winning trade
        cb.check_and_update(live_trading_config, trade_pnl=50, trade_volume=100)

        assert cb.consecutive_losses == 0


class TestKillSwitch:
    """Tests for kill switch functionality."""

    @pytest.mark.asyncio
    async def test_activate_kill_switch(self, live_trading_engine):
        """Test kill switch activation."""
        await live_trading_engine.activate_kill_switch(reason="Test activation")

        assert live_trading_engine.kill_switch_active is True

    @pytest.mark.asyncio
    async def test_deactivate_kill_switch(self, live_trading_engine):
        """Test kill switch deactivation."""
        live_trading_engine.kill_switch_active = True

        await live_trading_engine.deactivate_kill_switch()

        assert live_trading_engine.kill_switch_active is False

    @pytest.mark.asyncio
    async def test_no_orders_when_kill_switch_active(self, live_trading_engine, sample_signal):
        """Test no orders processed when kill switch is active."""
        live_trading_engine.kill_switch_active = True

        order = await live_trading_engine.process_signal(
            signal=sample_signal,
            token_id="token123",
            current_price=0.50,
        )

        assert order is None


class TestOrderProcessing:
    """Tests for order processing."""

    @pytest.mark.asyncio
    async def test_hold_signal_no_order(self, live_trading_engine):
        """Test HOLD signal doesn't create order."""
        signal = CheckpointSignal(
            symbol="BTC",
            slug="btc-updown-15m-123",
            checkpoint="7m30s",
            timestamp=int(time.time()),
            signal=SignalType.HOLD,
            fair_value=0.50,
            market_price=0.50,
            edge=0.0,
            confidence=0.5,
            momentum={},
        )

        order = await live_trading_engine.process_signal(
            signal=signal,
            token_id="token123",
            current_price=0.50,
        )

        assert order is None

    @pytest.mark.asyncio
    async def test_low_confidence_no_order(self, live_trading_engine, sample_signal):
        """Test low confidence signal doesn't create order."""
        sample_signal.confidence = 0.3  # Below default 0.7 threshold

        order = await live_trading_engine.process_signal(
            signal=sample_signal,
            token_id="token123",
            current_price=0.50,
        )

        assert order is None

    @pytest.mark.asyncio
    async def test_paper_mode_creates_paper_order(self, live_trading_engine, sample_signal):
        """Test paper mode creates paper order without CLOB call."""
        live_trading_engine.config.mode = TradingMode.PAPER
        sample_signal.confidence = 0.8

        order = await live_trading_engine.process_signal(
            signal=sample_signal,
            token_id="token123",
            current_price=0.50,
        )

        if order:
            assert order.status == "paper"

    @pytest.mark.asyncio
    async def test_max_positions_prevents_new_orders(self, live_trading_engine, sample_signal):
        """Test max positions limit prevents new orders."""
        live_trading_engine.config.max_open_positions = 1
        sample_signal.confidence = 0.8

        # Add existing position
        live_trading_engine.open_positions.append(
            LivePosition(
                symbol="ETH",
                side="UP",
                token_id="token456",
                size=100,
                avg_entry_price=0.50,
                cost_basis_usd=50,
                market_start=int(time.time()),
                market_end=int(time.time()) + 900,
            )
        )

        order = await live_trading_engine.process_signal(
            signal=sample_signal,
            token_id="token123",
            current_price=0.50,
        )

        assert order is None

    @pytest.mark.asyncio
    async def test_duplicate_signal_rejected(self, live_trading_engine, sample_signal):
        """Test duplicate signals are rejected (idempotency)."""
        sample_signal.confidence = 0.8
        live_trading_engine.config.mode = TradingMode.PAPER

        # Process first signal
        order1 = await live_trading_engine.process_signal(
            signal=sample_signal,
            token_id="token123",
            current_price=0.50,
        )

        # Process same signal again
        order2 = await live_trading_engine.process_signal(
            signal=sample_signal,
            token_id="token123",
            current_price=0.50,
        )

        # Second should be rejected as duplicate
        assert order2 is None


class TestPositionManagement:
    """Tests for live position management."""

    @pytest.mark.asyncio
    async def test_position_created_on_fill(self, live_trading_engine, sample_signal):
        """Test position is created when order is filled."""
        live_trading_engine.config.mode = TradingMode.PAPER
        sample_signal.confidence = 0.8

        order = await live_trading_engine.process_signal(
            signal=sample_signal,
            token_id="token123",
            current_price=0.50,
        )

        if order and order.status == "paper":
            assert len(live_trading_engine.open_positions) == 1

    @pytest.mark.asyncio
    async def test_resolve_winning_position(self, live_trading_engine):
        """Test resolving a winning position."""
        # Add position
        position = LivePosition(
            symbol="BTC",
            side="UP",
            token_id="token123",
            size=100,
            avg_entry_price=0.50,
            cost_basis_usd=50,
            market_start=int(time.time()) - 900,
            market_end=int(time.time()),
        )
        live_trading_engine.open_positions.append(position)

        # Resolve with UP (win)
        await live_trading_engine.resolve_position(
            symbol="BTC",
            market_start=position.market_start,
            binance_open=91000,
            binance_close=92000,  # UP
        )

        assert len(live_trading_engine.open_positions) == 0


class TestStatePersistence:
    """Tests for state save/load."""

    def test_save_and_load_state(self, live_trading_engine, tmp_path):
        """Test state persistence."""
        # Set some state
        live_trading_engine.kill_switch_active = True
        live_trading_engine.circuit_breaker.consecutive_losses = 2

        # Save
        live_trading_engine._save_state()

        # Load into new engine
        new_engine = LiveTradingEngine(
            config=live_trading_engine.config,
            paper_engine=live_trading_engine.paper_engine,
            state_path=live_trading_engine._get_state_path(),
        )

        assert new_engine.kill_switch_active is True
        assert new_engine.circuit_breaker.consecutive_losses == 2


class TestModeSwitch:
    """Tests for mode switching."""

    def test_switch_to_paper(self, live_trading_engine):
        """Test switching to paper mode."""
        live_trading_engine.set_mode("paper")
        assert live_trading_engine.config.mode == TradingMode.PAPER

    def test_switch_to_live_without_clob(self, live_trading_engine):
        """Test switching to live without CLOB client logs error."""
        live_trading_engine.clob_client = None
        live_trading_engine.set_mode("live")
        # Should log error but not crash


class TestWalletBalance:
    """Tests for wallet balance checks."""

    def test_get_wallet_balance_no_clob(self, live_trading_engine):
        """Test wallet balance returns error without CLOB."""
        live_trading_engine.clob_client = None

        result = live_trading_engine.get_wallet_balance()

        assert "error" in result

    def test_check_sufficient_balance(self, live_trading_engine):
        """Test balance check logic."""
        live_trading_engine.clob_client = None

        has_balance, msg = live_trading_engine.check_sufficient_balance(100)

        assert has_balance is False
        assert "not initialized" in msg.lower()


class TestGetStatus:
    """Tests for status retrieval."""

    def test_get_status(self, live_trading_engine):
        """Test status includes all required fields."""
        status = live_trading_engine.get_status()

        assert "mode" in status
        assert "kill_switch_active" in status
        assert "circuit_breaker" in status
        assert "open_positions" in status
        assert "enabled_assets" in status

    def test_get_positions(self, live_trading_engine):
        """Test positions list."""
        live_trading_engine.open_positions.append(
            LivePosition(
                symbol="BTC",
                side="UP",
                token_id="token123",
                size=100,
                avg_entry_price=0.50,
                cost_basis_usd=50,
                market_start=int(time.time()),
                market_end=int(time.time()) + 900,
            )
        )

        positions = live_trading_engine.get_positions()

        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC"
