"""
Trade Ledger - SQLite-based persistent storage for all trades.

Records both paper and live trades with full audit trail.
Provides query methods for analysis and reporting.
"""

import sqlite3
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


logger = logging.getLogger("trade_ledger")


# ============================================================================
# TRADE RECORD
# ============================================================================

# ============================================================================
# TRANSACTION TYPES
# ============================================================================

class TransactionType:
    """Transaction types for the ledger"""
    TRADE = "trade"           # A completed trade (win or loss)
    CREDIT = "credit"         # Money deposited into account
    DEBIT = "debit"           # Money withdrawn from account
    ADJUSTMENT = "adjustment" # Manual P&L adjustment
    FEE = "fee"               # Trading fees


@dataclass
class Transaction:
    """
    A ledger transaction (credit, debit, fee, or adjustment).

    Separate from trades - this tracks account balance changes.
    """
    id: str                    # Unique transaction ID
    account: str               # Account name (e.g., "paper", "live", "main")
    type: str                  # credit, debit, adjustment, fee
    amount: float              # Positive = credit, negative = debit
    balance_after: float       # Running balance after this transaction
    description: str
    created_at: str            # ISO timestamp
    reference_id: Optional[str] = None  # Links to trade_id if applicable
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeRecord:
    """
    Unified trade record for both paper and live trades.

    This captures everything needed for analysis and auditing.
    """
    # Identity
    id: str                         # Unique trade ID
    mode: str                       # "paper" or "live"

    # Trade details
    symbol: str                     # BTC, ETH, etc.
    side: str                       # "UP" or "DOWN"
    direction: str                  # "BUY" or "SELL"

    # Pricing
    entry_price: float
    exit_price: float               # 1.0 if won, 0.0 if lost
    size: float                     # Number of contracts
    cost_basis: float               # Total USD cost
    settlement_value: float         # USD value at settlement

    # P&L
    pnl: float                      # Profit/loss in USD
    pnl_pct: float                  # % return

    # Timing
    entry_time: int                 # Unix timestamp
    exit_time: int                  # Unix timestamp
    market_start: int               # 15-min window start
    market_end: int                 # 15-min window end

    # Resolution
    resolution: str                 # "UP" or "DOWN" - actual outcome
    is_winner: bool

    # Binance reference prices
    binance_open: float
    binance_close: float

    # Signal info
    checkpoint: str                 # "3m", "7m", "10m", "12.5m", "latency"
    signal_confidence: float
    signal_edge: float

    # Live-specific fields (nullable for paper)
    polymarket_order_id: Optional[str] = None
    tx_hash: Optional[str] = None

    # Metadata
    created_at: Optional[str] = None  # ISO timestamp when record created
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# TRADE LEDGER
# ============================================================================

class TradeLedger:
    """
    SQLite-based trade ledger for persistent storage.

    Features:
    - ACID-compliant storage
    - Full audit trail
    - Query methods for analysis
    - Easy migration path to PostgreSQL
    """

    def __init__(self, db_path: str = "trades.db"):
        """
        Initialize the trade ledger.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"TradeLedger initialized: {db_path}")

    def _init_database(self):
        """Create tables if they don't exist"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    size REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    settlement_value REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    entry_time INTEGER NOT NULL,
                    exit_time INTEGER NOT NULL,
                    market_start INTEGER NOT NULL,
                    market_end INTEGER NOT NULL,
                    resolution TEXT NOT NULL,
                    is_winner INTEGER NOT NULL,
                    binance_open REAL NOT NULL,
                    binance_close REAL NOT NULL,
                    checkpoint TEXT NOT NULL,
                    signal_confidence REAL NOT NULL,
                    signal_edge REAL NOT NULL,
                    polymarket_order_id TEXT,
                    tx_hash TEXT,
                    created_at TEXT NOT NULL,
                    notes TEXT
                )
            """)

            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades(mode)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_is_winner ON trades(is_winner)")

            # Transactions table for credits, debits, fees, adjustments
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    account TEXT NOT NULL,
                    type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    balance_after REAL NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reference_id TEXT,
                    notes TEXT
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions(account)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at)")

            # Accounts table for tracking balances
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    name TEXT PRIMARY KEY,
                    balance REAL NOT NULL DEFAULT 0,
                    initial_balance REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # WRITE OPERATIONS
    # -------------------------------------------------------------------------

    def record_trade(self, trade: TradeRecord) -> bool:
        """
        Record a completed trade.

        Args:
            trade: The trade record to store

        Returns:
            True if successful, False otherwise
        """
        # Set created_at if not provided
        if not trade.created_at:
            trade.created_at = datetime.now(timezone.utc).isoformat()

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO trades (
                        id, mode, symbol, side, direction,
                        entry_price, exit_price, size, cost_basis, settlement_value,
                        pnl, pnl_pct, entry_time, exit_time, market_start, market_end,
                        resolution, is_winner, binance_open, binance_close,
                        checkpoint, signal_confidence, signal_edge,
                        polymarket_order_id, tx_hash, created_at, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.id, trade.mode, trade.symbol, trade.side, trade.direction,
                    trade.entry_price, trade.exit_price, trade.size, trade.cost_basis, trade.settlement_value,
                    trade.pnl, trade.pnl_pct, trade.entry_time, trade.exit_time, trade.market_start, trade.market_end,
                    trade.resolution, 1 if trade.is_winner else 0, trade.binance_open, trade.binance_close,
                    trade.checkpoint, trade.signal_confidence, trade.signal_edge,
                    trade.polymarket_order_id, trade.tx_hash, trade.created_at, trade.notes
                ))
                conn.commit()

            logger.info(f"Recorded trade: {trade.id} | {trade.mode} | {trade.symbol} {trade.side} | P&L: ${trade.pnl:+.2f}")
            return True

        except sqlite3.IntegrityError:
            logger.warning(f"Trade already exists: {trade.id}")
            return False
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            return False

    def add_note(self, trade_id: str, note: str) -> bool:
        """Add or update a note on a trade"""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE trades SET notes = ? WHERE id = ?",
                    (note, trade_id)
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add note: {e}")
            return False

    # -------------------------------------------------------------------------
    # ACCOUNT OPERATIONS
    # -------------------------------------------------------------------------

    def create_account(
        self,
        name: str,
        initial_balance: float = 0.0,
    ) -> bool:
        """
        Create a new account.

        Args:
            name: Unique account name (e.g., "paper", "live", "main")
            initial_balance: Starting balance

        Returns:
            True if created, False if already exists
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO accounts (name, balance, initial_balance, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, initial_balance, initial_balance, now, now))
                conn.commit()

            # Record initial credit if > 0
            if initial_balance > 0:
                self.record_credit(name, initial_balance, "Initial deposit")

            logger.info(f"Created account: {name} with balance ${initial_balance:,.2f}")
            return True

        except sqlite3.IntegrityError:
            logger.debug(f"Account already exists: {name}")
            return False
        except Exception as e:
            logger.error(f"Failed to create account: {e}")
            return False

    def get_account_balance(self, name: str) -> float:
        """Get current balance for an account"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT balance FROM accounts WHERE name = ?", (name,)
            )
            row = cursor.fetchone()
            return row["balance"] if row else 0.0

    def get_account_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get full account information"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM accounts WHERE name = ?", (name,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "name": row["name"],
                "balance": row["balance"],
                "initial_balance": row["initial_balance"],
                "pnl": row["balance"] - row["initial_balance"],
                "pnl_pct": ((row["balance"] / row["initial_balance"]) - 1) * 100 if row["initial_balance"] > 0 else 0,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def list_accounts(self) -> List[Dict[str, Any]]:
        """List all accounts"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM accounts ORDER BY name")
            results = []
            for row in cursor.fetchall():
                results.append({
                    "name": row["name"],
                    "balance": row["balance"],
                    "initial_balance": row["initial_balance"],
                    "pnl": row["balance"] - row["initial_balance"],
                    "created_at": row["created_at"],
                })
            return results

    def _update_account_balance(self, name: str, delta: float) -> float:
        """
        Update account balance by delta.

        Returns new balance.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            # Get current balance
            cursor = conn.execute(
                "SELECT balance FROM accounts WHERE name = ?", (name,)
            )
            row = cursor.fetchone()
            if not row:
                # Create account if doesn't exist
                conn.execute("""
                    INSERT INTO accounts (name, balance, initial_balance, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, delta, 0, now, now))
                conn.commit()
                return delta

            new_balance = row["balance"] + delta
            conn.execute(
                "UPDATE accounts SET balance = ?, updated_at = ? WHERE name = ?",
                (new_balance, now, name)
            )
            conn.commit()
            return new_balance

    # -------------------------------------------------------------------------
    # TRANSACTION OPERATIONS
    # -------------------------------------------------------------------------

    def record_credit(
        self,
        account: str,
        amount: float,
        description: str,
        notes: Optional[str] = None,
    ) -> Optional[Transaction]:
        """
        Record a credit (deposit) to an account.

        Args:
            account: Account name
            amount: Amount to credit (positive)
            description: Description of the credit
            notes: Optional notes

        Returns:
            Transaction record or None if failed
        """
        return self._record_transaction(
            account=account,
            type=TransactionType.CREDIT,
            amount=abs(amount),
            description=description,
            notes=notes,
        )

    def record_debit(
        self,
        account: str,
        amount: float,
        description: str,
        notes: Optional[str] = None,
    ) -> Optional[Transaction]:
        """
        Record a debit (withdrawal) from an account.

        Args:
            account: Account name
            amount: Amount to debit (positive value, will be recorded as negative)
            description: Description of the debit
            notes: Optional notes

        Returns:
            Transaction record or None if failed
        """
        return self._record_transaction(
            account=account,
            type=TransactionType.DEBIT,
            amount=-abs(amount),
            description=description,
            notes=notes,
        )

    def record_adjustment(
        self,
        account: str,
        amount: float,
        description: str,
        notes: Optional[str] = None,
    ) -> Optional[Transaction]:
        """
        Record a manual P&L adjustment.

        Args:
            account: Account name
            amount: Adjustment amount (positive or negative)
            description: Description of the adjustment
            notes: Optional notes

        Returns:
            Transaction record or None if failed
        """
        return self._record_transaction(
            account=account,
            type=TransactionType.ADJUSTMENT,
            amount=amount,
            description=description,
            notes=notes,
        )

    def record_fee(
        self,
        account: str,
        amount: float,
        description: str,
        reference_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[Transaction]:
        """
        Record a fee.

        Args:
            account: Account name
            amount: Fee amount (positive value, will be recorded as negative)
            description: Description of the fee
            reference_id: Optional trade or order ID this fee relates to
            notes: Optional notes

        Returns:
            Transaction record or None if failed
        """
        return self._record_transaction(
            account=account,
            type=TransactionType.FEE,
            amount=-abs(amount),
            description=description,
            reference_id=reference_id,
            notes=notes,
        )

    def record_trade_pnl(
        self,
        account: str,
        trade_id: str,
        pnl: float,
        description: str,
    ) -> Optional[Transaction]:
        """
        Record P&L from a completed trade.

        Args:
            account: Account name
            trade_id: The trade ID this P&L relates to
            pnl: Profit or loss amount
            description: Description

        Returns:
            Transaction record or None if failed
        """
        return self._record_transaction(
            account=account,
            type=TransactionType.TRADE,
            amount=pnl,
            description=description,
            reference_id=trade_id,
        )

    def _record_transaction(
        self,
        account: str,
        type: str,
        amount: float,
        description: str,
        reference_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[Transaction]:
        """
        Internal method to record a transaction.
        """
        now = datetime.now(timezone.utc).isoformat()
        tx_id = f"{type}_{int(datetime.now().timestamp() * 1000)}"

        try:
            # Update account balance and get new balance
            new_balance = self._update_account_balance(account, amount)

            # Create transaction record
            tx = Transaction(
                id=tx_id,
                account=account,
                type=type,
                amount=amount,
                balance_after=new_balance,
                description=description,
                created_at=now,
                reference_id=reference_id,
                notes=notes,
            )

            # Store in database
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO transactions (
                        id, account, type, amount, balance_after,
                        description, created_at, reference_id, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tx.id, tx.account, tx.type, tx.amount, tx.balance_after,
                    tx.description, tx.created_at, tx.reference_id, tx.notes
                ))
                conn.commit()

            logger.info(f"Transaction: {type} ${amount:+,.2f} -> {account} (balance: ${new_balance:,.2f})")
            return tx

        except Exception as e:
            logger.error(f"Failed to record transaction: {e}")
            return None

    def get_transactions(
        self,
        account: Optional[str] = None,
        type: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Transaction]:
        """
        Get transactions with optional filters.

        Args:
            account: Filter by account
            type: Filter by transaction type
            since: Filter transactions after this ISO timestamp or Unix timestamp
            until: Filter transactions before this ISO timestamp or Unix timestamp
            limit: Max results
            offset: Skip first N results

        Returns:
            List of transactions, newest first
        """
        query = "SELECT * FROM transactions WHERE 1=1"
        params: list = []

        if account:
            query += " AND account = ?"
            params.append(account)

        if type:
            query += " AND type = ?"
            params.append(type)

        if since:
            # Convert to ISO if it's a Unix timestamp
            since_iso = self._normalize_timestamp(since)
            if since_iso:
                query += " AND created_at >= ?"
                params.append(since_iso)

        if until:
            # Convert to ISO if it's a Unix timestamp
            until_iso = self._normalize_timestamp(until)
            if until_iso:
                query += " AND created_at <= ?"
                params.append(until_iso)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append(Transaction(
                    id=row["id"],
                    account=row["account"],
                    type=row["type"],
                    amount=row["amount"],
                    balance_after=row["balance_after"],
                    description=row["description"],
                    created_at=row["created_at"],
                    reference_id=row["reference_id"],
                    notes=row["notes"],
                ))
            return results

    def _normalize_timestamp(self, ts: str) -> Optional[str]:
        """
        Normalize a timestamp to ISO format.

        Accepts:
        - ISO format strings (2024-01-15T10:30:00Z)
        - Unix timestamps as strings ("1705312200")
        - Date strings (2024-01-15)

        Returns:
            ISO format string or None if invalid
        """
        if not ts:
            return None

        # Try to parse as Unix timestamp (seconds)
        try:
            unix_ts = float(ts)
            # Reasonable range check (after year 2000, before year 2100)
            if 946684800 < unix_ts < 4102444800:
                return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()
        except (ValueError, TypeError):
            pass

        # Try to parse as ISO format or date string
        try:
            # Already ISO format with timezone
            if "T" in ts and ("Z" in ts or "+" in ts):
                return ts
            # ISO format without timezone - assume UTC
            if "T" in ts:
                return ts + "Z"
            # Date only - assume start of day UTC
            if "-" in ts and len(ts) == 10:
                return ts + "T00:00:00Z"
        except Exception:
            pass

        return None

    def get_pnl_summary(self, account: str) -> Dict[str, Any]:
        """
        Get P&L summary for an account.

        Returns:
            Dict with realized_pnl, unrealized_pnl, total_pnl, etc.
        """
        with self._get_connection() as conn:
            # Get account info
            cursor = conn.execute(
                "SELECT * FROM accounts WHERE name = ?", (account,)
            )
            acct = cursor.fetchone()
            if not acct:
                return {
                    "account": account,
                    "balance": 0,
                    "initial_balance": 0,
                    "total_pnl": 0,
                    "realized_pnl": 0,
                    "credits": 0,
                    "debits": 0,
                    "fees": 0,
                    "trade_count": 0,
                }

            # Sum by transaction type
            cursor = conn.execute("""
                SELECT
                    type,
                    SUM(amount) as total,
                    COUNT(*) as count
                FROM transactions
                WHERE account = ?
                GROUP BY type
            """, (account,))

            totals = {row["type"]: {"total": row["total"], "count": row["count"]}
                      for row in cursor.fetchall()}

            return {
                "account": account,
                "balance": acct["balance"],
                "initial_balance": acct["initial_balance"],
                "total_pnl": acct["balance"] - acct["initial_balance"],
                "realized_pnl": totals.get(TransactionType.TRADE, {}).get("total", 0),
                "credits": totals.get(TransactionType.CREDIT, {}).get("total", 0),
                "debits": abs(totals.get(TransactionType.DEBIT, {}).get("total", 0)),
                "fees": abs(totals.get(TransactionType.FEE, {}).get("total", 0)),
                "adjustments": totals.get(TransactionType.ADJUSTMENT, {}).get("total", 0),
                "trade_count": totals.get(TransactionType.TRADE, {}).get("count", 0),
            }

    # -------------------------------------------------------------------------
    # READ OPERATIONS
    # -------------------------------------------------------------------------

    def _row_to_trade(self, row: sqlite3.Row) -> TradeRecord:
        """Convert a database row to TradeRecord"""
        return TradeRecord(
            id=row["id"],
            mode=row["mode"],
            symbol=row["symbol"],
            side=row["side"],
            direction=row["direction"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            size=row["size"],
            cost_basis=row["cost_basis"],
            settlement_value=row["settlement_value"],
            pnl=row["pnl"],
            pnl_pct=row["pnl_pct"],
            entry_time=row["entry_time"],
            exit_time=row["exit_time"],
            market_start=row["market_start"],
            market_end=row["market_end"],
            resolution=row["resolution"],
            is_winner=bool(row["is_winner"]),
            binance_open=row["binance_open"],
            binance_close=row["binance_close"],
            checkpoint=row["checkpoint"],
            signal_confidence=row["signal_confidence"],
            signal_edge=row["signal_edge"],
            polymarket_order_id=row["polymarket_order_id"],
            tx_hash=row["tx_hash"],
            created_at=row["created_at"],
            notes=row["notes"],
        )

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Get a single trade by ID"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            return self._row_to_trade(row) if row else None

    def get_trades(
        self,
        mode: Optional[str] = None,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        until: Optional[int] = None,
        is_winner: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TradeRecord]:
        """
        Query trades with filters.

        Args:
            mode: Filter by "paper" or "live"
            symbol: Filter by symbol (BTC, ETH, etc.)
            since: Filter trades after this Unix timestamp
            until: Filter trades before this Unix timestamp
            is_winner: Filter by win/loss
            limit: Max results to return
            offset: Skip first N results

        Returns:
            List of matching trades, newest first
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if since:
            query += " AND exit_time >= ?"
            params.append(since)

        if until:
            query += " AND exit_time <= ?"
            params.append(until)

        if is_winner is not None:
            query += " AND is_winner = ?"
            params.append(1 if is_winner else 0)

        query += " ORDER BY exit_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_trade(row) for row in cursor.fetchall()]

    def get_recent_trades(self, limit: int = 20) -> List[TradeRecord]:
        """Get most recent trades across all modes"""
        return self.get_trades(limit=limit)

    # -------------------------------------------------------------------------
    # ANALYTICS
    # -------------------------------------------------------------------------

    def get_stats(
        self,
        mode: Optional[str] = None,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics.

        Returns:
            Dict with total_trades, wins, losses, win_rate, total_pnl, etc.
        """
        query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade,
                SUM(cost_basis) as total_volume
            FROM trades WHERE 1=1
        """
        params: list = []

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if since:
            query += " AND exit_time >= ?"
            params.append(since)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()

            total = row["total_trades"] or 0
            wins = row["wins"] or 0

            return {
                "total_trades": total,
                "wins": wins,
                "losses": row["losses"] or 0,
                "win_rate": wins / total if total > 0 else 0,
                "total_pnl": row["total_pnl"] or 0,
                "avg_pnl": row["avg_pnl"] or 0,
                "best_trade": row["best_trade"] or 0,
                "worst_trade": row["worst_trade"] or 0,
                "total_volume": row["total_volume"] or 0,
            }

    def get_daily_stats(self, days: int = 30, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get daily P&L breakdown.

        Args:
            days: Number of days to look back
            mode: Filter by mode

        Returns:
            List of daily stats, each with date, trades, pnl, win_rate
        """
        query = """
            SELECT
                date(exit_time, 'unixepoch') as trade_date,
                COUNT(*) as trades,
                SUM(pnl) as pnl,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins
            FROM trades
            WHERE exit_time >= strftime('%s', 'now', ?)
        """
        params: list = [f"-{days} days"]

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        query += " GROUP BY trade_date ORDER BY trade_date DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                trades = row["trades"]
                wins = row["wins"]
                results.append({
                    "date": row["trade_date"],
                    "trades": trades,
                    "pnl": row["pnl"],
                    "wins": wins,
                    "win_rate": wins / trades if trades > 0 else 0,
                })
            return results

    def get_symbol_stats(self, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get per-symbol breakdown.

        Returns:
            List of symbol stats with trades, pnl, win_rate per symbol
        """
        query = """
            SELECT
                symbol,
                COUNT(*) as trades,
                SUM(pnl) as pnl,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                AVG(signal_confidence) as avg_confidence,
                AVG(signal_edge) as avg_edge
            FROM trades WHERE 1=1
        """
        params: list = []

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        query += " GROUP BY symbol ORDER BY pnl DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                trades = row["trades"]
                wins = row["wins"]
                results.append({
                    "symbol": row["symbol"],
                    "trades": trades,
                    "pnl": row["pnl"],
                    "wins": wins,
                    "win_rate": wins / trades if trades > 0 else 0,
                    "avg_confidence": row["avg_confidence"],
                    "avg_edge": row["avg_edge"],
                })
            return results

    def get_checkpoint_stats(self, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get per-checkpoint breakdown (to see which checkpoints are most profitable).

        Returns:
            List of checkpoint stats with trades, pnl, win_rate per checkpoint
        """
        query = """
            SELECT
                checkpoint,
                COUNT(*) as trades,
                SUM(pnl) as pnl,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                AVG(signal_edge) as avg_edge
            FROM trades WHERE 1=1
        """
        params: list = []

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        query += " GROUP BY checkpoint ORDER BY trades DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                trades = row["trades"]
                wins = row["wins"]
                results.append({
                    "checkpoint": row["checkpoint"],
                    "trades": trades,
                    "pnl": row["pnl"],
                    "wins": wins,
                    "win_rate": wins / trades if trades > 0 else 0,
                    "avg_edge": row["avg_edge"],
                })
            return results

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------

    def export_csv(self, filepath: str, mode: Optional[str] = None) -> int:
        """
        Export trades to CSV.

        Args:
            filepath: Output file path
            mode: Optional mode filter

        Returns:
            Number of trades exported
        """
        import csv

        trades = self.get_trades(mode=mode, limit=100000)  # High limit for export

        if not trades:
            return 0

        fieldnames = list(trades[0].to_dict().keys())

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in trades:
                writer.writerow(trade.to_dict())

        logger.info(f"Exported {len(trades)} trades to {filepath}")
        return len(trades)

    def export_json(self, filepath: str, mode: Optional[str] = None) -> int:
        """
        Export trades to JSON.

        Args:
            filepath: Output file path
            mode: Optional mode filter

        Returns:
            Number of trades exported
        """
        trades = self.get_trades(mode=mode, limit=100000)

        with open(filepath, "w") as f:
            json.dump([t.to_dict() for t in trades], f, indent=2)

        logger.info(f"Exported {len(trades)} trades to {filepath}")
        return len(trades)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_trade_record_from_paper(
    paper_trade: Any,
    signal_edge: float = 0.0,
) -> TradeRecord:
    """
    Create a TradeRecord from a PaperTrade.

    Args:
        paper_trade: A PaperTrade instance
        signal_edge: The edge that was calculated for the signal

    Returns:
        TradeRecord ready for ledger
    """
    return TradeRecord(
        id=paper_trade.id,
        mode="paper",
        symbol=paper_trade.symbol,
        side=paper_trade.side,
        direction="BUY",  # Paper trades are always buys
        entry_price=paper_trade.entry_price,
        exit_price=paper_trade.exit_price,
        size=paper_trade.size,
        cost_basis=paper_trade.cost_basis,
        settlement_value=paper_trade.settlement_value,
        pnl=paper_trade.pnl,
        pnl_pct=paper_trade.pnl_pct,
        entry_time=paper_trade.entry_time,
        exit_time=paper_trade.exit_time,
        market_start=paper_trade.market_start,
        market_end=paper_trade.market_end,
        resolution=paper_trade.resolution,
        is_winner=paper_trade.side == paper_trade.resolution,
        binance_open=paper_trade.binance_open,
        binance_close=paper_trade.binance_close,
        checkpoint=paper_trade.checkpoint,
        signal_confidence=paper_trade.signal_confidence,
        signal_edge=signal_edge,
    )


def create_trade_record_from_live(
    position: Any,
    resolution: str,
    binance_open: float,
    binance_close: float,
    pnl: float,
    signal_confidence: float = 0.0,
    signal_edge: float = 0.0,
    polymarket_order_id: Optional[str] = None,
    tx_hash: Optional[str] = None,
) -> TradeRecord:
    """
    Create a TradeRecord from a live position resolution.

    Args:
        position: A LivePosition instance
        resolution: "UP" or "DOWN"
        binance_open: Binance price at market start
        binance_close: Binance price at market end
        pnl: Realized P&L
        signal_confidence: Confidence at entry
        signal_edge: Edge at entry
        polymarket_order_id: Order ID from Polymarket
        tx_hash: Transaction hash

    Returns:
        TradeRecord ready for ledger
    """
    is_winner = position.side == resolution
    exit_price = 1.0 if is_winner else 0.0
    settlement_value = position.size if is_winner else 0.0
    pnl_pct = pnl / position.cost_basis_usd * 100 if position.cost_basis_usd > 0 else 0

    return TradeRecord(
        id=f"{position.symbol}_{position.market_start}_{int(datetime.now().timestamp())}",
        mode="live",
        symbol=position.symbol,
        side=position.side,
        direction="BUY",
        entry_price=position.avg_entry_price,
        exit_price=exit_price,
        size=position.size,
        cost_basis=position.cost_basis_usd,
        settlement_value=settlement_value,
        pnl=pnl,
        pnl_pct=pnl_pct,
        entry_time=int(position.entry_orders[0].split("_")[1]) if position.entry_orders else position.market_start,
        exit_time=int(datetime.now().timestamp()),
        market_start=position.market_start,
        market_end=position.market_end,
        resolution=resolution,
        is_winner=is_winner,
        binance_open=binance_open,
        binance_close=binance_close,
        checkpoint="live",
        signal_confidence=signal_confidence,
        signal_edge=signal_edge,
        polymarket_order_id=polymarket_order_id,
        tx_hash=tx_hash,
    )
