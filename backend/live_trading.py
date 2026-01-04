"""
Live Trading Engine for Polymarket 15-Minute Crypto Markets.

This module handles real order execution on Polymarket CLOB.
It wraps the paper trading signals with actual order placement.

CRITICAL: This trades real money. All safety features must be enabled.

Requirements:
- pip install py-clob-client
- Polygon wallet with USDC
- Token allowances set for Polymarket contracts
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from trade_ledger import TradeLedger

# Polymarket SDK
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType, OpenOrderParams
    from py_clob_client.order_builder.constants import BUY, SELL
    from py_clob_client.constants import POLYGON
    CLOB_AVAILABLE = True
except ImportError:
    CLOB_AVAILABLE = False
    BUY = "BUY"
    SELL = "SELL"
    print("[LiveTrading] WARNING: py-clob-client not installed. Run: pip install py-clob-client")

# Local imports
from paper_trading import (
    PaperTradingConfig,
    PaperTradingEngine,
    CheckpointSignal,
    SignalType,
)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup comprehensive logging for audit trail"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("live_trading")
    logger.setLevel(logging.DEBUG)

    # File handler - all logs
    file_handler = logging.FileHandler(
        f"{log_dir}/live_trading_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    ))

    # Trade-specific log
    trade_handler = logging.FileHandler(
        f"{log_dir}/trades_{datetime.now().strftime('%Y%m%d')}.log"
    )
    trade_handler.setLevel(logging.INFO)
    trade_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(message)s'
    ))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(trade_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# ============================================================================
# CONFIGURATION
# ============================================================================

class TradingMode(Enum):
    """Trading mode selection"""
    PAPER = "paper"  # Real signals, simulated execution (no real money)
    LIVE = "live"    # Real money trades via Polymarket CLOB


@dataclass
class LiveTradingConfig:
    """Live trading configuration with safety limits"""

    # Mode
    mode: TradingMode = TradingMode.PAPER

    # API Configuration
    clob_host: str = "https://clob.polymarket.com"
    chain_id: int = POLYGON if CLOB_AVAILABLE else 137

    # Position limits (HARD CAPS - never exceeded)
    max_position_usd: float = 5000.0  # Max per position
    max_daily_volume_usd: float = 50000.0  # Max daily trading volume
    max_open_positions: int = 3  # Max concurrent positions

    # Risk limits
    max_consecutive_losses: int = 5  # Halt after N consecutive losses
    daily_loss_limit_usd: float = 1000.0  # Halt if daily loss exceeds
    max_drawdown_pct: float = 20.0  # Halt if drawdown from peak exceeds

    # Order settings
    order_type: str = "GTC"  # GTC, FOK, or GTD
    order_expiry_sec: int = 300  # GTD expiry (5 minutes)
    max_slippage_pct: float = 2.0  # Cancel if price moves more than this

    # Confirmation requirements
    require_manual_confirm: bool = False  # Require manual approval for each trade
    min_signal_confidence: float = 0.7  # Minimum confidence to execute

    # Alerts
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook_url: Optional[str] = None

    # Assets
    enabled_assets: list = field(default_factory=lambda: ["BTC"])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "LiveTradingConfig":
        if "mode" in data:
            data["mode"] = TradingMode(data["mode"])
        return cls(**data)

    @classmethod
    def from_env(cls) -> "LiveTradingConfig":
        """Load config from environment variables"""
        return cls(
            mode=TradingMode(os.getenv("TRADING_MODE", "paper")),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
        )


# ============================================================================
# LIVE ORDER
# ============================================================================

@dataclass
class LiveOrder:
    """A live order on Polymarket"""
    id: str
    symbol: str
    side: str  # "UP" or "DOWN"
    direction: str  # "BUY" or "SELL"
    token_id: str
    size_usd: float
    price: float
    order_type: str
    status: str  # "pending", "filled", "partial", "cancelled", "failed"
    created_at: int
    filled_at: Optional[int] = None
    filled_size: float = 0.0
    filled_price: float = 0.0
    error: Optional[str] = None
    polymarket_order_id: Optional[str] = None
    tx_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LivePosition:
    """A live position on Polymarket"""
    symbol: str
    side: str  # "UP" or "DOWN"
    token_id: str
    size: float  # Number of contracts
    avg_entry_price: float
    cost_basis_usd: float
    market_start: int
    market_end: int
    entry_orders: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# CIRCUIT BREAKERS
# ============================================================================

@dataclass
class CircuitBreaker:
    """Circuit breaker state"""
    triggered: bool = False
    reason: str = ""
    triggered_at: Optional[int] = None
    consecutive_losses: int = 0
    daily_loss_usd: float = 0.0
    daily_volume_usd: float = 0.0
    peak_balance_usd: float = 0.0
    current_balance_usd: float = 0.0
    last_reset_date: str = ""

    def check_and_update(self, config: LiveTradingConfig, trade_pnl: float, trade_volume: float) -> bool:
        """
        Check circuit breakers after a trade.
        Returns True if trading should be halted.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Daily reset
        if self.last_reset_date != today:
            self.consecutive_losses = 0
            self.daily_loss_usd = 0.0
            self.daily_volume_usd = 0.0
            self.last_reset_date = today
            self.triggered = False
            self.reason = ""

        # Update metrics
        self.daily_volume_usd += trade_volume

        if trade_pnl < 0:
            self.consecutive_losses += 1
            self.daily_loss_usd += abs(trade_pnl)
        else:
            self.consecutive_losses = 0

        # Update balance tracking
        self.current_balance_usd += trade_pnl
        if self.current_balance_usd > self.peak_balance_usd:
            self.peak_balance_usd = self.current_balance_usd

        # Check limits
        if self.consecutive_losses >= config.max_consecutive_losses:
            self.triggered = True
            self.reason = f"Consecutive losses: {self.consecutive_losses}"
            self.triggered_at = int(time.time())
            return True

        if self.daily_loss_usd >= config.daily_loss_limit_usd:
            self.triggered = True
            self.reason = f"Daily loss limit: ${self.daily_loss_usd:.2f}"
            self.triggered_at = int(time.time())
            return True

        if self.daily_volume_usd >= config.max_daily_volume_usd:
            self.triggered = True
            self.reason = f"Daily volume limit: ${self.daily_volume_usd:.2f}"
            self.triggered_at = int(time.time())
            return True

        # Drawdown check
        if self.peak_balance_usd > 0:
            drawdown = (self.peak_balance_usd - self.current_balance_usd) / self.peak_balance_usd * 100
            if drawdown >= config.max_drawdown_pct:
                self.triggered = True
                self.reason = f"Max drawdown: {drawdown:.1f}%"
                self.triggered_at = int(time.time())
                return True

        return False

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# LIVE TRADING ENGINE
# ============================================================================

class LiveTradingEngine:
    """
    Live trading engine for Polymarket.

    Wraps paper trading signals with real order execution.
    Includes comprehensive safety features:
    - Circuit breakers
    - Kill switch
    - Position limits
    - Audit logging
    - Alerts
    """

    def __init__(
        self,
        private_key: Optional[str] = None,
        config: Optional[LiveTradingConfig] = None,
        data_dir: str = ".",
        ledger: Optional["TradeLedger"] = None,
    ):
        self.config = config or LiveTradingConfig()
        self.data_dir = data_dir
        self.private_key = private_key

        # Trade ledger for persistent storage
        self.ledger = ledger

        # Initialize paper trading engine for signals (share ledger)
        self.paper_engine = PaperTradingEngine(data_dir=data_dir, ledger=ledger)

        # Polymarket client (only if live mode and key provided)
        self.clob_client: Optional[ClobClient] = None
        if private_key and CLOB_AVAILABLE and self.config.mode == TradingMode.LIVE:
            self._init_clob_client()

        # State
        self.circuit_breaker = CircuitBreaker()
        self.open_positions: list[LivePosition] = []
        self.order_history: list[LiveOrder] = []
        self.kill_switch_active = False

        # Order idempotency - track processed signals to prevent duplicates
        self._processed_signals: set[str] = set()
        self._signal_cache_max_size = 1000  # Prevent unbounded growth

        # Callbacks
        self.on_order: Optional[Callable[[LiveOrder], None]] = None
        self.on_fill: Optional[Callable[[LiveOrder], None]] = None
        self.on_alert: Optional[Callable[[str, str], None]] = None

        # Load state
        self._load_state()

        logger.info(f"LiveTradingEngine initialized in {self.config.mode.value} mode")

    def _init_clob_client(self):
        """Initialize Polymarket CLOB client"""
        if not CLOB_AVAILABLE:
            logger.error("py-clob-client not installed")
            return

        try:
            # Initialize client with private key
            # signature_type: 0=EOA (MetaMask), 1=Email/Magic, 2=Browser proxy
            self.clob_client = ClobClient(
                host=self.config.clob_host,
                key=self.private_key,
                chain_id=self.config.chain_id,
                signature_type=0,  # EOA wallet
            )

            # Derive or create API credentials
            self.clob_client.set_api_creds(self.clob_client.create_or_derive_api_creds())

            logger.info("CLOB client initialized successfully")

            # Verify connection by getting balance
            try:
                balance = self.get_wallet_balance()
                logger.info(f"Wallet balance: ${balance.get('usdc_balance', 0):.2f} USDC")
            except Exception as e:
                logger.warning(f"Could not verify wallet balance: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize CLOB client: {e}")
            self.clob_client = None

    # -------------------------------------------------------------------------
    # KILL SWITCH
    # -------------------------------------------------------------------------

    async def activate_kill_switch(self, reason: str = "Manual activation"):
        """
        IMMEDIATELY halt all trading.
        This is the emergency stop button.
        """
        self.kill_switch_active = True
        self.circuit_breaker.triggered = True
        self.circuit_breaker.reason = f"KILL SWITCH: {reason}"
        self.circuit_breaker.triggered_at = int(time.time())

        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        await self._send_alert("KILL SWITCH", f"Trading halted: {reason}")

        # Cancel all open orders
        self._cancel_all_orders()

        self._save_state()

    async def deactivate_kill_switch(self):
        """Deactivate kill switch (requires manual intervention)"""
        self.kill_switch_active = False
        self.circuit_breaker.triggered = False
        self.circuit_breaker.reason = ""

        logger.warning("Kill switch deactivated - trading resumed")
        await self._send_alert("RESUME", "Trading resumed after kill switch")

        self._save_state()

    # -------------------------------------------------------------------------
    # SIGNAL PROCESSING
    # -------------------------------------------------------------------------

    async def process_signal(
        self,
        signal: CheckpointSignal,
        token_id: str,
        current_price: float,
    ) -> Optional[LiveOrder]:
        """
        Process a trading signal and potentially execute an order.

        Args:
            signal: The trading signal from paper engine
            token_id: Polymarket token ID for the UP outcome
            current_price: Current market price for UP

        Returns:
            LiveOrder if order was placed, None otherwise
        """
        # Safety checks
        if self.kill_switch_active:
            logger.debug("Kill switch active - ignoring signal")
            return None

        if self.circuit_breaker.triggered:
            logger.debug(f"Circuit breaker triggered: {self.circuit_breaker.reason}")
            return None

        if signal.signal == SignalType.HOLD:
            return None

        if signal.confidence < self.config.min_signal_confidence:
            logger.debug(f"Signal confidence {signal.confidence:.2f} below threshold")
            return None

        # Idempotency check - prevent duplicate orders for same signal
        signal_key = f"{signal.symbol}_{signal.checkpoint}_{signal.timestamp}"
        if signal_key in self._processed_signals:
            logger.debug(f"Signal already processed: {signal_key}")
            return None

        # Add to processed set (with size limit to prevent memory leak)
        self._processed_signals.add(signal_key)
        if len(self._processed_signals) > self._signal_cache_max_size:
            # Remove oldest entries (convert to list, remove first 100)
            to_remove = list(self._processed_signals)[:100]
            for key in to_remove:
                self._processed_signals.discard(key)

        # Check position limits
        if len(self.open_positions) >= self.config.max_open_positions:
            logger.warning("Max open positions reached")
            return None

        # Check if we already have a position for this symbol/window
        for pos in self.open_positions:
            if pos.symbol == signal.symbol and pos.market_start == signal.momentum.get("market_start"):
                logger.debug("Already have position for this market")
                return None

        # Determine order parameters
        if signal.signal in [SignalType.BUY_UP, SignalType.BUY_MORE_UP]:
            side = "UP"
            price = current_price
        else:
            side = "DOWN"
            price = 1 - current_price  # DOWN token price

        # Calculate position size
        size_usd = min(
            self.paper_engine._calculate_position_size(),
            self.config.max_position_usd,
        )

        # Create order
        order = LiveOrder(
            id=f"{signal.symbol}_{int(time.time())}_{signal.checkpoint}",
            symbol=signal.symbol,
            side=side,
            direction="BUY",
            token_id=token_id if side == "UP" else self._get_down_token_id(token_id),
            size_usd=size_usd,
            price=price,
            order_type=self.config.order_type,
            status="pending",
            created_at=int(time.time()),
        )

        # Log intent
        logger.info(
            f"SIGNAL: {signal.symbol} {signal.signal.value} | "
            f"Edge: {signal.edge:.1%} | Conf: {signal.confidence:.2f} | "
            f"Size: ${size_usd:.2f} @ {price:.3f}"
        )

        # Execute based on mode
        if self.config.mode == TradingMode.PAPER:
            # Paper mode: real signals, simulated execution
            # Exercises full code path except post_order()
            order.status = "paper"
            order.filled_at = int(time.time())
            order.filled_size = size_usd / price
            order.filled_price = price
            logger.info(f"PAPER ORDER: {order.id} (would execute in live mode)")

        elif self.config.mode == TradingMode.LIVE:
            # Check balance before executing
            has_balance, balance_msg = self.check_sufficient_balance(size_usd)
            if not has_balance:
                order.status = "failed"
                order.error = balance_msg
                logger.warning(f"Order rejected: {balance_msg}")
                await self._send_alert("INSUFFICIENT BALANCE", f"{order.symbol}: {balance_msg}")
            elif self.config.require_manual_confirm:
                order.status = "pending_confirmation"
                logger.info(f"ORDER PENDING CONFIRMATION: {order.id}")
                await self._send_alert("CONFIRM REQUIRED", f"{order.symbol} {order.side} ${order.size_usd:.2f}")
            else:
                await self._execute_order(order)

        # Track order
        self.order_history.append(order)

        # Create position if filled (or paper-filled)
        if order.status in ("filled", "paper"):
            self._create_position(order, signal)

        # Fire callback
        if self.on_order:
            self.on_order(order)

        self._save_state()
        return order

    async def _execute_order(self, order: LiveOrder, max_retries: int = 3):
        """
        Execute a live order on Polymarket with retry logic.

        Args:
            order: The order to execute
            max_retries: Maximum number of retry attempts
        """
        if not self.clob_client:
            order.status = "failed"
            order.error = "CLOB client not initialized"
            logger.error(f"Order failed: {order.error}")
            return

        last_error = None
        for attempt in range(max_retries):
            try:
                # Check if price has moved too much (slippage protection)
                if attempt > 0:
                    current_price = self._get_current_price(order.token_id)
                    if current_price:
                        slippage = abs(current_price - order.price) / order.price * 100
                        if slippage > self.config.max_slippage_pct:
                            order.status = "cancelled"
                            order.error = f"Slippage too high: {slippage:.2f}%"
                            logger.warning(f"Order cancelled due to slippage: {order.id}")
                            await self._send_alert("ORDER CANCELLED", f"{order.symbol}: slippage {slippage:.1f}%")
                            return
                        # Update order price for retry
                        order.price = current_price

                # Determine order type
                order_type = OrderType.GTC
                if self.config.order_type == "FOK":
                    order_type = OrderType.FOK
                elif self.config.order_type == "GTD":
                    order_type = OrderType.GTD

                # Use MarketOrderArgs for dollar-based orders (FOK for immediate fill)
                # Use OrderArgs for limit orders (GTC for persistence)
                if order_type == OrderType.FOK:
                    # Market order - specify dollar amount, fill immediately or cancel
                    market_order = MarketOrderArgs(
                        token_id=order.token_id,
                        amount=order.size_usd,
                        side=BUY,
                    )
                    signed_order = self.clob_client.create_market_order(market_order)
                else:
                    # Limit order - specify shares at price
                    limit_order = OrderArgs(
                        token_id=order.token_id,
                        price=order.price,
                        size=order.size_usd / order.price,  # Convert USD to shares
                        side=BUY,
                    )
                    signed_order = self.clob_client.create_order(limit_order)

                # Submit the order
                response = self.clob_client.post_order(signed_order, order_type)

                # Update order with response
                order.polymarket_order_id = response.get("orderID")
                order.status = "submitted"

                logger.info(f"ORDER SUBMITTED: {order.id} -> {order.polymarket_order_id} (attempt {attempt + 1})")
                await self._send_alert("ORDER", f"{order.symbol} {order.side} ${order.size_usd:.2f} submitted")
                return  # Success - exit retry loop

            except Exception as e:
                last_error = e
                logger.warning(f"Order attempt {attempt + 1}/{max_retries} failed: {e}")

                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s (non-blocking)
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

        # All retries failed
        order.status = "failed"
        order.error = str(last_error)
        logger.error(f"Order execution failed after {max_retries} attempts: {last_error}")
        await self._send_alert("ORDER FAILED", f"{order.symbol}: {last_error}")

    def _get_current_price(self, token_id: str) -> Optional[float]:
        """Get current market price for a token (for slippage check)"""
        if not self.clob_client:
            return None

        try:
            # Get order book to find current best price
            book = self.clob_client.get_order_book(token_id)
            if book and book.get("bids"):
                return float(book["bids"][0]["price"])
        except Exception as e:
            logger.warning(f"Failed to get current price: {e}")

        return None

    def _create_position(self, order: LiveOrder, signal: CheckpointSignal):
        """Create a position from a filled order"""
        position = LivePosition(
            symbol=order.symbol,
            side=order.side,
            token_id=order.token_id,
            size=order.filled_size,
            avg_entry_price=order.filled_price,
            cost_basis_usd=order.size_usd,
            market_start=signal.momentum.get("market_start", 0),
            market_end=signal.momentum.get("market_end", 0),
            entry_orders=[order.id],
        )
        self.open_positions.append(position)

        logger.info(f"POSITION OPENED: {position.symbol} {position.side} | Size: {position.size:.2f} @ {position.avg_entry_price:.3f}")

    def _get_down_token_id(self, up_token_id: str) -> str:
        """Get the DOWN token ID from UP token ID"""
        # In Polymarket, UP and DOWN tokens are paired
        # This would need to be fetched from the API in practice
        # For now, return placeholder
        return f"{up_token_id}_DOWN"

    def _cancel_all_orders(self):
        """Cancel all open orders"""
        if not self.clob_client:
            return

        try:
            self.clob_client.cancel_all()
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

    # -------------------------------------------------------------------------
    # POSITION RESOLUTION
    # -------------------------------------------------------------------------

    async def resolve_position(
        self,
        symbol: str,
        market_start: int,
        resolution: str,  # "UP" or "DOWN"
        binance_open: float = 0.0,
        binance_close: float = 0.0,
    ) -> Optional[float]:
        """
        Resolve a position when market closes.

        Returns P&L in USD.
        """
        # Find matching position
        position = None
        for pos in self.open_positions:
            if pos.symbol == symbol and pos.market_start == market_start:
                position = pos
                break

        if not position:
            return None

        # Calculate P&L
        is_winner = position.side == resolution
        exit_value = position.size if is_winner else 0
        pnl = exit_value - position.cost_basis_usd

        # Update circuit breaker
        self.circuit_breaker.check_and_update(
            self.config,
            pnl,
            position.cost_basis_usd,
        )

        # Log result
        result = "WIN" if is_winner else "LOSS"
        logger.info(
            f"POSITION CLOSED: {position.symbol} {position.side} | "
            f"Result: {result} | P&L: ${pnl:+.2f}"
        )

        await self._send_alert(
            f"TRADE {result}",
            f"{position.symbol} {position.side}: ${pnl:+.2f}"
        )

        # Record to ledger
        if self.ledger:
            try:
                from trade_ledger import create_trade_record_from_live
                trade_record = create_trade_record_from_live(
                    position=position,
                    resolution=resolution,
                    binance_open=binance_open,
                    binance_close=binance_close,
                    pnl=pnl,
                )
                self.ledger.record_trade(trade_record)
            except Exception as e:
                logger.error(f"Failed to record trade to ledger: {e}")

        # Remove position
        self.open_positions.remove(position)

        # Check if circuit breaker triggered
        if self.circuit_breaker.triggered:
            logger.warning(f"Circuit breaker triggered: {self.circuit_breaker.reason}")
            await self._send_alert("CIRCUIT BREAKER", self.circuit_breaker.reason)

        self._save_state()
        return pnl

    # -------------------------------------------------------------------------
    # ALERTS
    # -------------------------------------------------------------------------

    async def _send_alert(self, title: str, message: str):
        """Send alert via configured channels (non-blocking)"""
        full_message = f"[{title}] {message}"
        logger.info(f"ALERT: {full_message}")

        # Fire callback
        if self.on_alert:
            self.on_alert(title, message)

        # Send alerts concurrently to avoid blocking
        tasks = []
        if self.config.telegram_bot_token and self.config.telegram_chat_id:
            tasks.append(self._send_telegram(full_message))
        if self.config.discord_webhook_url:
            tasks.append(self._send_discord(full_message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_telegram(self, message: str):
        """Send Telegram message (non-blocking)"""
        try:
            import httpx
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json={
                    "chat_id": self.config.telegram_chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }, timeout=10.0)
                if response.status_code != 200:
                    logger.warning(f"Telegram API returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")

    async def _send_discord(self, message: str):
        """Send Discord webhook message (non-blocking)"""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(self.config.discord_webhook_url, json={
                    "content": message,
                }, timeout=10.0)
        except Exception as e:
            logger.error(f"Discord alert failed: {e}")

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    def _get_state_path(self) -> str:
        return os.path.join(self.data_dir, "live_trading_state.json")

    def _save_state(self):
        """Save state to JSON"""
        state = {
            "config": self.config.to_dict(),
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "kill_switch_active": self.kill_switch_active,
            "open_positions": [p.to_dict() for p in self.open_positions],
            "order_history": [o.to_dict() for o in self.order_history[-100:]],
        }

        try:
            with open(self._get_state_path(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load state from JSON"""
        try:
            path = self._get_state_path()
            if not os.path.exists(path):
                return

            with open(path, "r") as f:
                state = json.load(f)

            # Load circuit breaker
            if "circuit_breaker" in state:
                cb = state["circuit_breaker"]
                self.circuit_breaker = CircuitBreaker(**cb)

            self.kill_switch_active = state.get("kill_switch_active", False)

            # Load positions
            for p_data in state.get("open_positions", []):
                self.open_positions.append(LivePosition(**p_data))

            # Load order history
            for o_data in state.get("order_history", []):
                self.order_history.append(LiveOrder(**o_data))

            logger.info(f"Loaded state: {len(self.open_positions)} positions, {len(self.order_history)} orders")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    # -------------------------------------------------------------------------
    # WALLET / BALANCE MANAGEMENT
    # -------------------------------------------------------------------------

    def get_wallet_balance(self) -> dict:
        """
        Get wallet USDC balance from Polymarket.

        Returns dict with:
        - usdc_balance: Available USDC
        - collateral_locked: USDC locked in open positions
        - total_value: Total account value
        """
        if not self.clob_client:
            return {
                "usdc_balance": 0.0,
                "collateral_locked": 0.0,
                "total_value": 0.0,
                "error": "CLOB client not initialized",
            }

        try:
            # Get balance from Polymarket API
            balance_info = self.clob_client.get_balance_allowance()

            usdc_balance = float(balance_info.get("balance", 0)) / 1e6  # USDC has 6 decimals
            allowance = float(balance_info.get("allowance", 0)) / 1e6

            # Calculate locked collateral from open positions
            collateral_locked = sum(p.cost_basis_usd for p in self.open_positions)

            return {
                "usdc_balance": usdc_balance,
                "allowance": allowance,
                "collateral_locked": collateral_locked,
                "total_value": usdc_balance + collateral_locked,
                "available_for_trading": min(usdc_balance, allowance) - collateral_locked,
            }

        except Exception as e:
            logger.error(f"Failed to get wallet balance: {e}")
            return {
                "usdc_balance": 0.0,
                "collateral_locked": 0.0,
                "total_value": 0.0,
                "error": str(e),
            }

    def check_sufficient_balance(self, amount_usd: float) -> tuple[bool, str]:
        """
        Check if there's sufficient balance for a trade.

        Returns (is_sufficient, message)
        """
        balance = self.get_wallet_balance()

        if "error" in balance:
            return False, f"Balance check failed: {balance['error']}"

        available = balance.get("available_for_trading", 0)

        if amount_usd > available:
            return False, f"Insufficient balance: need ${amount_usd:.2f}, have ${available:.2f}"

        return True, "OK"

    def check_token_allowances(self) -> tuple[bool, str, dict]:
        """
        Check if token allowances are set for Polymarket contracts.

        Per Polymarket docs: "You only need to set these once per wallet"
        but they MUST be set before trading.

        Returns (is_approved, message, details)
        """
        if not self.clob_client:
            return False, "CLOB client not initialized", {}

        try:
            balance_info = self.clob_client.get_balance_allowance()
            allowance = float(balance_info.get("allowance", 0)) / 1e6
            balance = float(balance_info.get("balance", 0)) / 1e6

            details = {
                "usdc_balance": balance,
                "usdc_allowance": allowance,
                "has_allowance": allowance > 0,
            }

            if allowance <= 0:
                return False, "Token allowance not set. Run set_allowances() first.", details

            if allowance < balance:
                return True, f"Warning: Allowance (${allowance:.2f}) less than balance (${balance:.2f})", details

            return True, "OK", details

        except Exception as e:
            logger.error(f"Failed to check allowances: {e}")
            return False, f"Allowance check failed: {e}", {}

    def set_allowances(self) -> tuple[bool, str]:
        """
        Set token allowances for Polymarket exchange contracts.

        Per Polymarket docs: Approves USDC and conditional tokens for
        the exchange contracts. Only needs to be done once per wallet.

        Returns (success, message)
        """
        if not self.clob_client:
            return False, "CLOB client not initialized"

        try:
            # This sets unlimited allowance for the exchange contracts
            result = self.clob_client.set_allowances()
            logger.info(f"Set allowances result: {result}")
            return True, "Allowances set successfully"
        except Exception as e:
            logger.error(f"Failed to set allowances: {e}")
            return False, f"Failed to set allowances: {e}"

    # -------------------------------------------------------------------------
    # API METHODS
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get current trading status"""
        status = {
            "mode": self.config.mode.value,
            "kill_switch_active": self.kill_switch_active,
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "open_positions": len(self.open_positions),
            "enabled_assets": self.config.enabled_assets,
            "clob_connected": self.clob_client is not None,
        }

        # Add balance info if in live mode
        if self.config.mode == TradingMode.LIVE and self.clob_client:
            status["wallet"] = self.get_wallet_balance()

        return status

    def get_positions(self) -> list[dict]:
        """Get open positions"""
        return [p.to_dict() for p in self.open_positions]

    def get_order_history(self, limit: int = 50) -> list[dict]:
        """Get recent orders"""
        return [o.to_dict() for o in self.order_history[-limit:]]

    def set_mode(self, mode: str):
        """Change trading mode"""
        try:
            self.config.mode = TradingMode(mode)
            logger.info(f"Trading mode changed to: {mode}")
            self._save_state()
        except ValueError:
            logger.error(f"Invalid mode: {mode}")

    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Config updated: {key} = {value}")
        self._save_state()
