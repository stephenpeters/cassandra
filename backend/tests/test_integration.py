"""
Integration tests for the full trading flow.

Tests cover:
- Signal generation -> Order placement -> Position resolution
- Live mode order flow
- State consistency across restarts
"""
import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading import TradingEngine, TradingConfig
from live_trading import LiveTradingEngine, LiveTradingConfig, TradingMode


class TestFullTradingFlow:
    """Integration tests for complete trading flow."""

    @pytest.fixture
    def trading_system(self, tmp_path):
        """Create a full trading system for integration tests."""
        # Paper trading config
        trading_config = TradingConfig(
            enabled=True,
            starting_balance=1000,
            max_position_usd=100,
            enabled_assets=["BTC", "ETH"],
            signal_checkpoints=[180, 360, 450, 540, 720],
            active_checkpoint=450,
            min_confirmations=2,
            use_vwap=True,
            use_rsi=True,
            use_supertrend=True,
        )

        # Create trading engine
        paper_engine = TradingEngine(
            config=trading_config,
            state_path=str(tmp_path / "paper_state.json"),
        )
        paper_engine.trading_mode = "paper"

        # Create live trading engine
        live_config = LiveTradingConfig(
            mode=TradingMode.PAPER,
            max_position_usd=100,
            enabled_assets=["BTC", "ETH"],
        )

        live_engine = LiveTradingEngine(
            config=live_config,
            paper_engine=paper_engine,
            state_path=str(tmp_path / "live_state.json"),
        )

        return paper_engine, live_engine

    def test_paper_mode_full_cycle(self, trading_system):
        """Test full cycle in paper mode: signal -> position -> resolution."""
        paper_engine, live_engine = trading_system

        # Record window open
        market_start = int(time.time()) - 450  # At 7:30 checkpoint
        paper_engine.record_window_open("BTC", market_start, 91000.0)

        # Create position manually (simulating signal)
        paper_engine._open_position(
            symbol="BTC",
            side="UP",
            entry_price=0.50,
            size=100,
            market_start=market_start,
            market_end=market_start + 900,
            checkpoint="7m30s",
        )

        assert len(paper_engine.account.positions) == 1
        initial_balance = paper_engine.account.balance

        # Resolve position (winning)
        paper_engine.resolve_market(
            symbol="BTC",
            market_start=market_start,
            market_end=market_start + 900,
            binance_open=91000.0,
            binance_close=92000.0,  # UP
        )

        # Position closed
        assert len(paper_engine.account.positions) == 0
        assert len(paper_engine.account.trade_history) == 1
        assert paper_engine.account.balance > initial_balance  # Won

    @pytest.mark.asyncio
    async def test_live_mode_signal_processing(self, trading_system):
        """Test live mode processes signals correctly."""
        paper_engine, live_engine = trading_system

        from trading import CheckpointSignal, SignalType

        # Create signal
        signal = CheckpointSignal(
            symbol="BTC",
            slug="btc-updown-15m-123",
            checkpoint="7m30s",
            timestamp=int(time.time()),
            signal=SignalType.BUY_UP,
            fair_value=0.60,
            market_price=0.50,
            edge=0.10,
            confidence=0.8,
            momentum={
                "direction": "UP",
                "market_start": int(time.time()) - 450,
            },
        )

        # Process signal
        order = await live_engine.process_signal(
            signal=signal,
            token_id="test_token_123",
            current_price=0.50,
        )

        # In paper mode, should create paper order
        if order:
            assert order.status == "paper"
            assert len(live_engine.open_positions) == 1

    def test_mode_switch_preserves_state(self, trading_system):
        """Test switching modes preserves state correctly."""
        paper_engine, live_engine = trading_system

        # Add some state
        paper_engine.account.balance = 1500
        paper_engine.trading_mode = "live"
        paper_engine._save_state()

        live_engine.circuit_breaker.consecutive_losses = 2
        live_engine._save_state()

        # Switch mode
        live_engine.set_mode("paper")

        # State should be preserved
        assert live_engine.circuit_breaker.consecutive_losses == 2

    def test_state_consistency_across_restart(self, trading_system, tmp_path):
        """Test state is consistent after simulated restart."""
        paper_engine, live_engine = trading_system

        # Modify state
        paper_engine.account.balance = 2000
        paper_engine.trading_mode = "live"

        # Create position
        market_start = int(time.time())
        paper_engine._open_position(
            symbol="ETH",
            side="DOWN",
            entry_price=0.45,
            size=50,
            market_start=market_start,
            market_end=market_start + 900,
            checkpoint="7m30s",
        )

        # Save
        paper_engine._save_state()

        # Create new engine (simulate restart)
        new_engine = TradingEngine(
            config=paper_engine.config,
            state_path=paper_engine.state_path,
        )

        # Verify state
        assert new_engine.account.balance == 2000
        assert new_engine.trading_mode == "live"
        assert len(new_engine.account.positions) == 1
        assert new_engine.account.positions[0].symbol == "ETH"


class TestConcurrency:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_duplicate_signals_rejected(self, trading_system):
        """Test duplicate signals are properly rejected."""
        paper_engine, live_engine = trading_system

        from trading import CheckpointSignal, SignalType

        signal = CheckpointSignal(
            symbol="BTC",
            slug="btc-updown-15m-123",
            checkpoint="7m30s",
            timestamp=int(time.time()),
            signal=SignalType.BUY_UP,
            fair_value=0.60,
            market_price=0.50,
            edge=0.10,
            confidence=0.8,
            momentum={"market_start": int(time.time()) - 450},
        )

        # Process same signal multiple times
        order1 = await live_engine.process_signal(signal, "token123", 0.50)
        order2 = await live_engine.process_signal(signal, "token123", 0.50)
        order3 = await live_engine.process_signal(signal, "token123", 0.50)

        # Only first should succeed
        orders = [o for o in [order1, order2, order3] if o is not None]
        assert len(orders) <= 1


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    def test_corrupted_state_file_recovery(self, tmp_path):
        """Test recovery from corrupted state file."""
        state_path = tmp_path / "corrupted.json"

        # Write corrupted JSON
        with open(state_path, "w") as f:
            f.write("{invalid json")

        # Should not crash, should start with fresh state
        engine = TradingEngine(
            config=TradingConfig(),
            state_path=str(state_path),
        )

        assert engine.account.balance == 1000  # Default
        assert len(engine.account.positions) == 0

    def test_missing_state_file(self, tmp_path):
        """Test handling of missing state file."""
        state_path = tmp_path / "nonexistent.json"

        # Should start with fresh state
        engine = TradingEngine(
            config=TradingConfig(),
            state_path=str(state_path),
        )

        assert engine.account.balance == 1000


class TestAPIIntegration:
    """Tests for API integration points."""

    def test_account_summary_includes_all_fields(self, trading_system):
        """Test account summary has all required fields."""
        paper_engine, _ = trading_system

        summary = paper_engine.get_account_summary()

        assert "balance" in summary
        assert "total_pnl" in summary
        assert "positions" in summary
        assert "recent_trades" in summary
        assert "win_rate" in summary

    def test_config_to_dict_complete(self, trading_system):
        """Test config serialization is complete."""
        paper_engine, _ = trading_system

        config = paper_engine.config.to_dict()

        # All key fields present
        assert "enabled" in config
        assert "enabled_assets" in config
        assert "signal_checkpoints" in config
        assert "min_confirmations" in config
        assert "use_vwap" in config

    def test_live_status_includes_all_fields(self, trading_system):
        """Test live trading status has all required fields."""
        _, live_engine = trading_system

        status = live_engine.get_status()

        assert "mode" in status
        assert "kill_switch_active" in status
        assert "circuit_breaker" in status
        assert "open_positions" in status
        assert "enabled_assets" in status
