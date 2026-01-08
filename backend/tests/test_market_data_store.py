"""
Tests for the MarketDataStore - rolling 24-hour SQLite storage.

Tests cover:
- Price snapshot recording and retrieval
- Market trade recording and retrieval
- Automatic cleanup of old data
- Query filtering
- Analytics methods
"""
import pytest
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_data_store import (
    MarketDataStore,
    PriceSnapshot,
    MarketTradeRecord,
    RETENTION_PERIOD_SEC,
    CLEANUP_INTERVAL_SEC,
)


class TestPriceSnapshots:
    """Tests for price snapshot operations."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a fresh store for each test."""
        db_path = str(tmp_path / "test_market_data.db")
        return MarketDataStore(db_path=db_path)

    def test_record_and_retrieve_snapshot(self, store):
        """Test basic snapshot recording and retrieval."""
        now = int(time.time())
        market_start = (now // 900) * 900

        snapshot = PriceSnapshot(
            id=f"snap_{now}_BTC",
            timestamp=now,
            symbol="BTC",
            binance_price=91000.0,
            pm_up_price=0.55,
            pm_down_price=0.45,
            market_start=market_start,
            market_end=market_start + 900,
            elapsed_sec=300,
            volume_delta_usd=10000.0,
            orderbook_imbalance=0.6,
        )

        # Record
        result = store.record_snapshot(snapshot)
        assert result is True

        # Retrieve
        snapshots = store.get_snapshots(symbol="BTC", limit=10)
        assert len(snapshots) == 1
        assert snapshots[0].symbol == "BTC"
        assert snapshots[0].binance_price == 91000.0
        assert snapshots[0].pm_up_price == 0.55

    def test_snapshot_filters(self, store):
        """Test snapshot query filters."""
        now = int(time.time())
        market_start = (now // 900) * 900

        # Create snapshots for multiple symbols
        for symbol in ["BTC", "ETH", "SOL"]:
            snapshot = PriceSnapshot(
                id=f"snap_{now}_{symbol}",
                timestamp=now,
                symbol=symbol,
                binance_price=91000.0 if symbol == "BTC" else 3000.0 if symbol == "ETH" else 100.0,
                pm_up_price=0.50,
                pm_down_price=0.50,
                market_start=market_start,
                market_end=market_start + 900,
                elapsed_sec=300,
            )
            store.record_snapshot(snapshot)

        # Filter by symbol
        btc_snapshots = store.get_snapshots(symbol="BTC")
        assert len(btc_snapshots) == 1
        assert btc_snapshots[0].symbol == "BTC"

        eth_snapshots = store.get_snapshots(symbol="ETH")
        assert len(eth_snapshots) == 1
        assert eth_snapshots[0].symbol == "ETH"

        # Get all
        all_snapshots = store.get_snapshots()
        assert len(all_snapshots) == 3

    def test_snapshot_time_filters(self, store):
        """Test snapshot time-based filters."""
        now = int(time.time())
        market_start = (now // 900) * 900

        # Create snapshots at different times
        for i, offset in enumerate([0, 60, 120]):
            snapshot = PriceSnapshot(
                id=f"snap_{now}_{i}",
                timestamp=now - offset,
                symbol="BTC",
                binance_price=91000.0 + i * 100,
                pm_up_price=0.50,
                pm_down_price=0.50,
                market_start=market_start,
                market_end=market_start + 900,
                elapsed_sec=300,
            )
            store.record_snapshot(snapshot)

        # Filter by since
        recent = store.get_snapshots(since=now - 90)
        assert len(recent) == 2  # Only snapshots within last 90s

        # Filter by until
        older = store.get_snapshots(until=now - 90)
        assert len(older) == 1  # Only snapshot older than 90s

    def test_snapshot_to_dict(self, store):
        """Test snapshot serialization."""
        now = int(time.time())
        snapshot = PriceSnapshot(
            id="test_snap",
            timestamp=now,
            symbol="BTC",
            binance_price=91000.0,
            pm_up_price=0.55,
            pm_down_price=0.45,
            market_start=now,
            market_end=now + 900,
            elapsed_sec=300,
            volume_delta_usd=10000.0,
            orderbook_imbalance=0.6,
        )

        d = snapshot.to_dict()
        assert d["symbol"] == "BTC"
        assert d["binance_price"] == 91000.0
        assert d["volume_delta_usd"] == 10000.0


class TestMarketTrades:
    """Tests for market trade operations."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a fresh store for each test."""
        db_path = str(tmp_path / "test_market_data.db")
        return MarketDataStore(db_path=db_path)

    def test_record_and_retrieve_trade(self, store):
        """Test basic trade recording and retrieval."""
        now = int(time.time())
        market_start = (now // 900) * 900

        trade = MarketTradeRecord(
            id=f"trade_{now}",
            timestamp=now,
            symbol="BTC",
            side="UP",
            size=100.0,
            price=0.55,
            usd_value=55.0,
            market_start=market_start,
            is_buy=True,
            maker="0x1234567890abcdef",
        )

        # Record
        result = store.record_trade(trade)
        assert result is True

        # Retrieve
        trades = store.get_trades(symbol="BTC", limit=10)
        assert len(trades) == 1
        assert trades[0].symbol == "BTC"
        assert trades[0].side == "UP"
        assert trades[0].is_buy is True
        assert trades[0].maker == "0x1234567890abcdef"

    def test_trade_filters(self, store):
        """Test trade query filters."""
        now = int(time.time())
        market_start = (now // 900) * 900

        # Create trades for different sides
        for side, is_buy in [("UP", True), ("UP", False), ("DOWN", True), ("DOWN", False)]:
            trade = MarketTradeRecord(
                id=f"trade_{now}_{side}_{is_buy}",
                timestamp=now,
                symbol="BTC",
                side=side,
                size=100.0,
                price=0.50,
                usd_value=50.0,
                market_start=market_start,
                is_buy=is_buy,
            )
            store.record_trade(trade)

        # Filter by side
        up_trades = store.get_trades(side="UP")
        assert len(up_trades) == 2

        down_trades = store.get_trades(side="DOWN")
        assert len(down_trades) == 2

        # Get all
        all_trades = store.get_trades()
        assert len(all_trades) == 4

    def test_trade_min_usd_filter(self, store):
        """Test trade minimum USD filter."""
        now = int(time.time())
        market_start = (now // 900) * 900

        # Create trades with different USD values
        for i, usd in enumerate([10, 50, 100, 500]):
            trade = MarketTradeRecord(
                id=f"trade_{now}_{i}",
                timestamp=now,
                symbol="BTC",
                side="UP",
                size=usd / 0.5,
                price=0.50,
                usd_value=usd,
                market_start=market_start,
                is_buy=True,
            )
            store.record_trade(trade)

        # Filter by min USD
        large_trades = store.get_trades(min_usd=100.0)
        assert len(large_trades) == 2  # 100 and 500

        whale_trades = store.get_trades(min_usd=500.0)
        assert len(whale_trades) == 1

    def test_trade_to_dict(self, store):
        """Test trade serialization."""
        now = int(time.time())
        trade = MarketTradeRecord(
            id="test_trade",
            timestamp=now,
            symbol="BTC",
            side="UP",
            size=100.0,
            price=0.55,
            usd_value=55.0,
            market_start=now,
            is_buy=True,
            maker="0x1234",
        )

        d = trade.to_dict()
        assert d["symbol"] == "BTC"
        assert d["side"] == "UP"
        assert d["is_buy"] is True
        assert d["maker"] == "0x1234"


class TestMarketAnalysis:
    """Tests for market analysis methods."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a fresh store for each test."""
        db_path = str(tmp_path / "test_market_data.db")
        return MarketDataStore(db_path=db_path)

    def test_get_market_analysis(self, store):
        """Test market analysis for a specific window."""
        now = int(time.time())
        market_start = (now // 900) * 900

        # Create snapshots
        for i in range(5):
            snapshot = PriceSnapshot(
                id=f"snap_{now}_{i}",
                timestamp=now - (4 - i) * 60,  # Every minute
                symbol="BTC",
                binance_price=91000.0 + i * 100,  # Price going up
                pm_up_price=0.50 + i * 0.02,
                pm_down_price=0.50 - i * 0.02,
                market_start=market_start,
                market_end=market_start + 900,
                elapsed_sec=i * 60,
            )
            store.record_snapshot(snapshot)

        # Create trades
        for i, (side, is_buy, usd) in enumerate([
            ("UP", True, 100),
            ("UP", True, 200),
            ("DOWN", True, 50),
        ]):
            trade = MarketTradeRecord(
                id=f"trade_{now}_{i}",
                timestamp=now,
                symbol="BTC",
                side=side,
                size=usd / 0.5,
                price=0.50,
                usd_value=usd,
                market_start=market_start,
                is_buy=is_buy,
            )
            store.record_trade(trade)

        # Get analysis
        analysis = store.get_market_analysis(market_start=market_start, symbol="BTC")

        assert analysis["symbol"] == "BTC"
        assert analysis["snapshot_count"] == 5
        assert analysis["trade_count"] == 3
        assert analysis["first_binance_price"] == 91000.0
        assert analysis["last_binance_price"] == 91400.0
        assert analysis["price_change_pct"] > 0  # Price went up
        assert analysis["total_volume_usd"] == 350
        assert analysis["up_trades"] == 2
        assert analysis["down_trades"] == 1

    def test_get_market_analysis_no_data(self, store):
        """Test analysis returns error when no data."""
        analysis = store.get_market_analysis(market_start=1234567890, symbol="BTC")
        assert "error" in analysis


class TestStoreStats:
    """Tests for store statistics."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a fresh store for each test."""
        db_path = str(tmp_path / "test_market_data.db")
        return MarketDataStore(db_path=db_path)

    def test_get_stats_empty(self, store):
        """Test stats on empty store."""
        stats = store.get_stats()
        assert stats["snapshot_count"] == 0
        assert stats["trade_count"] == 0
        assert stats["retention_hours"] == 24

    def test_get_stats_with_data(self, store):
        """Test stats with data."""
        now = int(time.time())

        # Add some data
        snapshot = PriceSnapshot(
            id=f"snap_{now}",
            timestamp=now,
            symbol="BTC",
            binance_price=91000.0,
            pm_up_price=0.50,
            pm_down_price=0.50,
            market_start=now,
            market_end=now + 900,
            elapsed_sec=0,
        )
        store.record_snapshot(snapshot)

        trade = MarketTradeRecord(
            id=f"trade_{now}",
            timestamp=now,
            symbol="BTC",
            side="UP",
            size=100.0,
            price=0.50,
            usd_value=50.0,
            market_start=now,
            is_buy=True,
        )
        store.record_trade(trade)

        stats = store.get_stats()
        assert stats["snapshot_count"] == 1
        assert stats["trade_count"] == 1
        assert stats["oldest_timestamp"] is not None
        assert stats["newest_timestamp"] is not None


class TestDataCleanup:
    """Tests for automatic data cleanup."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a fresh store for each test."""
        db_path = str(tmp_path / "test_market_data.db")
        return MarketDataStore(db_path=db_path)

    def test_cleanup_removes_old_data(self, store):
        """Test that cleanup removes data older than retention period."""
        now = int(time.time())

        # Create old snapshot (outside retention)
        old_snapshot = PriceSnapshot(
            id="old_snap",
            timestamp=now - RETENTION_PERIOD_SEC - 100,  # Older than 24h
            symbol="BTC",
            binance_price=90000.0,
            pm_up_price=0.50,
            pm_down_price=0.50,
            market_start=now - RETENTION_PERIOD_SEC - 100,
            market_end=now - RETENTION_PERIOD_SEC,
            elapsed_sec=0,
        )
        store.record_snapshot(old_snapshot)

        # Create new snapshot (within retention)
        new_snapshot = PriceSnapshot(
            id="new_snap",
            timestamp=now,
            symbol="BTC",
            binance_price=91000.0,
            pm_up_price=0.50,
            pm_down_price=0.50,
            market_start=now,
            market_end=now + 900,
            elapsed_sec=0,
        )
        store.record_snapshot(new_snapshot)

        # Before cleanup
        snapshots = store.get_snapshots()
        assert len(snapshots) == 2

        # Run cleanup
        store.cleanup_old_data()

        # After cleanup - only new snapshot should remain
        snapshots = store.get_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].id == "new_snap"

    def test_cleanup_removes_old_trades(self, store):
        """Test that cleanup removes trades older than retention period."""
        now = int(time.time())

        # Create old trade
        old_trade = MarketTradeRecord(
            id="old_trade",
            timestamp=now - RETENTION_PERIOD_SEC - 100,
            symbol="BTC",
            side="UP",
            size=100.0,
            price=0.50,
            usd_value=50.0,
            market_start=now - RETENTION_PERIOD_SEC - 100,
            is_buy=True,
        )
        store.record_trade(old_trade)

        # Create new trade
        new_trade = MarketTradeRecord(
            id="new_trade",
            timestamp=now,
            symbol="BTC",
            side="UP",
            size=100.0,
            price=0.50,
            usd_value=50.0,
            market_start=now,
            is_buy=True,
        )
        store.record_trade(new_trade)

        # Run cleanup
        store.cleanup_old_data()

        # Only new trade should remain
        trades = store.get_trades()
        assert len(trades) == 1
        assert trades[0].id == "new_trade"
