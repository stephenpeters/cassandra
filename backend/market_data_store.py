"""
Market Data Store - Rolling 24-hour SQLite storage for price snapshots and trades.

V2 Phase 3 implementation for historical data collection.
- Price snapshots every 30 seconds
- Market trades (all significant trades)
- Automatic 24-hour rolling retention
"""

import sqlite3
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

logger = logging.getLogger("market_data_store")

# Rolling retention period (24 hours in seconds)
RETENTION_PERIOD_SEC = 24 * 60 * 60

# Cleanup runs every hour
CLEANUP_INTERVAL_SEC = 60 * 60


@dataclass
class PriceSnapshot:
    """
    A point-in-time price snapshot for analysis.

    Captured every 30 seconds during market hours.
    """
    id: str                      # Unique ID
    timestamp: int               # Unix timestamp
    symbol: str                  # BTC, ETH, SOL
    binance_price: float         # Binance spot price
    pm_up_price: float           # Polymarket UP token price
    pm_down_price: float         # Polymarket DOWN token price
    market_start: int            # Current 15-min window start
    market_end: int              # Current 15-min window end
    elapsed_sec: int             # Seconds into current window
    volume_delta_usd: Optional[float] = None  # Volume delta since window start
    orderbook_imbalance: Optional[float] = None  # Order book bid/ask imbalance
    target_price: Optional[float] = None  # Chainlink reference price (price to beat)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketTradeRecord:
    """
    A trade on a Polymarket 15-min market.

    Captured from WebSocket feed.
    """
    id: str                      # Unique trade ID
    timestamp: int               # Unix timestamp
    symbol: str                  # BTC, ETH, SOL
    side: str                    # "UP" or "DOWN"
    size: float                  # Number of contracts
    price: float                 # Trade price (0-1)
    usd_value: float             # USD value
    market_start: int            # 15-min window start
    is_buy: bool                 # True if buy, False if sell
    maker: Optional[str] = None  # Maker address (if known)

    def to_dict(self) -> dict:
        return asdict(self)


class MarketDataStore:
    """
    SQLite storage for rolling 24-hour market data.

    Features:
    - Price snapshots (every 30s)
    - Market trades
    - Automatic retention cleanup
    - Query methods for analysis
    """

    def __init__(self, db_path: str = "market_data.db"):
        """
        Initialize the market data store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._last_cleanup = 0
        self._init_database()
        logger.info(f"MarketDataStore initialized: {db_path}")

    def _init_database(self):
        """Create tables if they don't exist"""
        with self._get_connection() as conn:
            # Price snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    binance_price REAL NOT NULL,
                    pm_up_price REAL NOT NULL,
                    pm_down_price REAL NOT NULL,
                    market_start INTEGER NOT NULL,
                    market_end INTEGER NOT NULL,
                    elapsed_sec INTEGER NOT NULL,
                    volume_delta_usd REAL,
                    orderbook_imbalance REAL,
                    target_price REAL
                )
            """)

            # Migration: Add target_price column if it doesn't exist (for existing DBs)
            try:
                conn.execute("ALTER TABLE price_snapshots ADD COLUMN target_price REAL")
            except sqlite3.OperationalError:
                pass  # Column already exists

            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON price_snapshots(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_symbol ON price_snapshots(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_market_start ON price_snapshots(market_start)")

            # Market trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_trades (
                    id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    price REAL NOT NULL,
                    usd_value REAL NOT NULL,
                    market_start INTEGER NOT NULL,
                    is_buy INTEGER NOT NULL,
                    maker TEXT
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON market_trades(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON market_trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_market_start ON market_trades(market_start)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _maybe_cleanup(self):
        """Run cleanup if enough time has passed"""
        now = int(time.time())
        if now - self._last_cleanup > CLEANUP_INTERVAL_SEC:
            self.cleanup_old_data()
            self._last_cleanup = now

    # =========================================================================
    # WRITE OPERATIONS
    # =========================================================================

    def record_snapshot(self, snapshot: PriceSnapshot) -> bool:
        """
        Record a price snapshot.

        Args:
            snapshot: The snapshot to store

        Returns:
            True if successful
        """
        self._maybe_cleanup()

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO price_snapshots (
                        id, timestamp, symbol, binance_price,
                        pm_up_price, pm_down_price, market_start, market_end,
                        elapsed_sec, volume_delta_usd, orderbook_imbalance, target_price
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.id, snapshot.timestamp, snapshot.symbol, snapshot.binance_price,
                    snapshot.pm_up_price, snapshot.pm_down_price, snapshot.market_start, snapshot.market_end,
                    snapshot.elapsed_sec, snapshot.volume_delta_usd, snapshot.orderbook_imbalance,
                    snapshot.target_price
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to record snapshot: {e}")
            return False

    def record_trade(self, trade: MarketTradeRecord) -> bool:
        """
        Record a market trade.

        Args:
            trade: The trade to store

        Returns:
            True if successful
        """
        self._maybe_cleanup()

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO market_trades (
                        id, timestamp, symbol, side, size, price,
                        usd_value, market_start, is_buy, maker
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.id, trade.timestamp, trade.symbol, trade.side, trade.size, trade.price,
                    trade.usd_value, trade.market_start, 1 if trade.is_buy else 0, trade.maker
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            return False

    def cleanup_old_data(self):
        """Remove data older than retention period"""
        cutoff = int(time.time()) - RETENTION_PERIOD_SEC

        try:
            with self._get_connection() as conn:
                # Delete old snapshots
                cursor = conn.execute(
                    "DELETE FROM price_snapshots WHERE timestamp < ?", (cutoff,)
                )
                snapshots_deleted = cursor.rowcount

                # Delete old trades
                cursor = conn.execute(
                    "DELETE FROM market_trades WHERE timestamp < ?", (cutoff,)
                )
                trades_deleted = cursor.rowcount

                conn.commit()

            if snapshots_deleted > 0 or trades_deleted > 0:
                logger.info(f"Cleanup: deleted {snapshots_deleted} snapshots, {trades_deleted} trades older than 24h")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    def get_snapshots(
        self,
        symbol: Optional[str] = None,
        market_start: Optional[int] = None,
        since: Optional[int] = None,
        until: Optional[int] = None,
        limit: int = 1000,
    ) -> List[PriceSnapshot]:
        """
        Query price snapshots.

        Args:
            symbol: Filter by symbol
            market_start: Filter by specific market window
            since: Filter after this timestamp
            until: Filter before this timestamp
            limit: Max results

        Returns:
            List of snapshots, newest first
        """
        query = "SELECT * FROM price_snapshots WHERE 1=1"
        params: list = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if market_start:
            query += " AND market_start = ?"
            params.append(market_start)

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        if until:
            query += " AND timestamp <= ?"
            params.append(until)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_snapshot(row) for row in cursor.fetchall()]

    def get_trades(
        self,
        symbol: Optional[str] = None,
        market_start: Optional[int] = None,
        side: Optional[str] = None,
        min_usd: Optional[float] = None,
        since: Optional[int] = None,
        until: Optional[int] = None,
        limit: int = 1000,
    ) -> List[MarketTradeRecord]:
        """
        Query market trades.

        Args:
            symbol: Filter by symbol
            market_start: Filter by specific market window
            side: Filter by "UP" or "DOWN"
            min_usd: Minimum USD value
            since: Filter after this timestamp
            until: Filter before this timestamp
            limit: Max results

        Returns:
            List of trades, newest first
        """
        query = "SELECT * FROM market_trades WHERE 1=1"
        params: list = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if market_start:
            query += " AND market_start = ?"
            params.append(market_start)

        if side:
            query += " AND side = ?"
            params.append(side)

        if min_usd:
            query += " AND usd_value >= ?"
            params.append(min_usd)

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        if until:
            query += " AND timestamp <= ?"
            params.append(until)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_trade(row) for row in cursor.fetchall()]

    def _row_to_snapshot(self, row: sqlite3.Row) -> PriceSnapshot:
        """Convert database row to PriceSnapshot"""
        # Handle target_price which may be missing in old rows
        target_price = None
        try:
            target_price = row["target_price"]
        except (KeyError, IndexError):
            pass
        return PriceSnapshot(
            id=row["id"],
            timestamp=row["timestamp"],
            symbol=row["symbol"],
            binance_price=row["binance_price"],
            pm_up_price=row["pm_up_price"],
            pm_down_price=row["pm_down_price"],
            market_start=row["market_start"],
            market_end=row["market_end"],
            elapsed_sec=row["elapsed_sec"],
            volume_delta_usd=row["volume_delta_usd"],
            orderbook_imbalance=row["orderbook_imbalance"],
            target_price=target_price,
        )

    def _row_to_trade(self, row: sqlite3.Row) -> MarketTradeRecord:
        """Convert database row to MarketTradeRecord"""
        return MarketTradeRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            symbol=row["symbol"],
            side=row["side"],
            size=row["size"],
            price=row["price"],
            usd_value=row["usd_value"],
            market_start=row["market_start"],
            is_buy=bool(row["is_buy"]),
            maker=row["maker"],
        )

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_market_analysis(self, market_start: int, symbol: str) -> Dict[str, Any]:
        """
        Get analysis for a specific market window.

        Returns:
            Dict with price evolution, trade volume, etc.
        """
        snapshots = self.get_snapshots(symbol=symbol, market_start=market_start, limit=100)
        trades = self.get_trades(symbol=symbol, market_start=market_start, limit=500)

        if not snapshots:
            return {"error": "No data for this market"}

        # Calculate metrics
        first_snapshot = snapshots[-1] if snapshots else None
        last_snapshot = snapshots[0] if snapshots else None

        total_volume = sum(t.usd_value for t in trades)
        buy_volume = sum(t.usd_value for t in trades if t.is_buy)
        sell_volume = total_volume - buy_volume

        up_trades = [t for t in trades if t.side == "UP"]
        down_trades = [t for t in trades if t.side == "DOWN"]

        return {
            "symbol": symbol,
            "market_start": market_start,
            "snapshot_count": len(snapshots),
            "trade_count": len(trades),
            "first_binance_price": first_snapshot.binance_price if first_snapshot else None,
            "last_binance_price": last_snapshot.binance_price if last_snapshot else None,
            "price_change_pct": (
                (last_snapshot.binance_price - first_snapshot.binance_price) / first_snapshot.binance_price * 100
                if first_snapshot and last_snapshot and first_snapshot.binance_price > 0
                else 0
            ),
            "first_pm_up_price": first_snapshot.pm_up_price if first_snapshot else None,
            "last_pm_up_price": last_snapshot.pm_up_price if last_snapshot else None,
            "total_volume_usd": total_volume,
            "buy_volume_usd": buy_volume,
            "sell_volume_usd": sell_volume,
            "up_trades": len(up_trades),
            "down_trades": len(down_trades),
            "up_volume_usd": sum(t.usd_value for t in up_trades),
            "down_volume_usd": sum(t.usd_value for t in down_trades),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall store statistics"""
        with self._get_connection() as conn:
            # Count snapshots
            cursor = conn.execute("SELECT COUNT(*) as count FROM price_snapshots")
            snapshot_count = cursor.fetchone()["count"]

            # Count trades
            cursor = conn.execute("SELECT COUNT(*) as count FROM market_trades")
            trade_count = cursor.fetchone()["count"]

            # Get time range
            cursor = conn.execute("""
                SELECT
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM price_snapshots
            """)
            row = cursor.fetchone()
            oldest = row["oldest"] if row else None
            newest = row["newest"] if row else None

        return {
            "snapshot_count": snapshot_count,
            "trade_count": trade_count,
            "oldest_timestamp": oldest,
            "newest_timestamp": newest,
            "retention_hours": RETENTION_PERIOD_SEC / 3600,
            "db_path": self.db_path,
        }
