"""
Pytest fixtures for the test suite.
"""
import pytest
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading import TradingEngine, TradingConfig, TradingAccount, Position, Trade, CheckpointSignal, SignalType
from live_trading import LiveTradingEngine, LiveTradingConfig, LiveOrder, LivePosition, TradingMode


@pytest.fixture
def trading_config():
    """Create a test trading config."""
    return TradingConfig(
        enabled=True,
        starting_balance=1000,
        max_position_pct=5,
        max_position_usd=100,
        enabled_assets=["BTC", "ETH"],
        signal_checkpoints=[180, 360, 450, 540, 720],
        active_checkpoint=450,
        min_confirmations=2,
        use_vwap=True,
        use_rsi=True,
        use_adx=True,
        use_supertrend=True,
    )


@pytest.fixture
def trading_engine(trading_config, tmp_path):
    """Create a test trading engine with temp state file."""
    # TradingEngine constructor takes data_dir and ledger
    engine = TradingEngine(data_dir=str(tmp_path), ledger=None)
    # Override config with test config
    engine.config = trading_config
    engine.account = TradingAccount(
        balance=trading_config.starting_balance,
        starting_balance=trading_config.starting_balance,
    )
    engine.trading_mode = "paper"
    # Store state path for tests
    engine.state_path = str(tmp_path / "paper_trading_state.json")
    return engine


@pytest.fixture
def live_trading_config():
    """Create a test live trading config."""
    return LiveTradingConfig(
        mode=TradingMode.PAPER,
        max_position_usd=100,
        max_daily_volume_usd=1000,
        max_open_positions=3,
        enabled_assets=["BTC", "ETH"],
    )


@pytest.fixture
def mock_paper_engine(trading_config):
    """Create a mock paper trading engine for live trading tests."""
    engine = MagicMock()
    engine.config = trading_config
    engine._calculate_position_size = MagicMock(return_value=50.0)
    engine._binance_opens = {}
    return engine


@pytest.fixture
def live_trading_engine(live_trading_config, tmp_path):
    """Create a test live trading engine."""
    # LiveTradingEngine creates its own paper_engine internally
    engine = LiveTradingEngine(
        private_key=None,  # No real trades
        config=live_trading_config,
        data_dir=str(tmp_path),
        ledger=None,
    )
    return engine


@pytest.fixture
def sample_signal():
    """Create a sample trading signal."""
    # CheckpointSignal doesn't have 'slug' field - generates slug in to_dict() from market_start
    return CheckpointSignal(
        symbol="BTC",
        checkpoint="7m30s",
        timestamp=1767812400,
        signal=SignalType.BUY_UP,
        fair_value=0.55,
        market_price=0.50,
        edge=0.05,
        confidence=0.75,
        momentum={
            "direction": "UP",
            "confidence": 0.7,
            "volume_delta": 15000,
            "price_change_pct": 0.05,
            "orderbook_imbalance": 0.15,
            "binance_current": 91500.0,
        },
        market_start=1767811500,  # Used to generate slug in to_dict()
    )


@pytest.fixture
def sample_position():
    """Create a sample position."""
    # Position doesn't have 'slug' field - generates slug in to_dict() from market_start
    return Position(
        id="BTC_1767811500_1767811947",
        symbol="BTC",
        side="UP",
        entry_price=0.50,
        size=100,
        cost_basis=50.0,
        entry_time=1767811947,
        market_start=1767811500,
        market_end=1767812400,
        checkpoint="7m30s",
    )


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
