"""
Paper Trading System for Polymarket 15-Minute Crypto Markets.

Exploits the 30-60 second latency gap between Binance price movements
and Polymarket price adjustments. When Binance moves significantly,
Polymarket lags behind, creating an edge window.

Strategy:
1. Track Binance price at market window open
2. Monitor real-time Binance price vs open price
3. Calculate implied probability from current price move
4. Compare to Polymarket UP price - if gap > threshold, execute
5. Gap typically closes within 30-60 seconds
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Callable, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from trade_ledger import TradeLedger


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TradingConfig:
    """Paper trading configuration"""
    enabled: bool = True
    starting_balance: float = 1000.0
    slippage_pct: float = 0.5  # 0.5% slippage
    commission_pct: float = 0.1  # 0.1% commission
    max_position_pct: float = 2.0  # Max 2% of account per position
    max_position_usd: float = 5000.0  # Hard cap at $5K per position (liquidity constraint)
    daily_loss_limit_pct: float = 10.0  # Stop trading if 10% daily loss

    # Asset selection (hot-reloadable, no restart needed)
    enabled_assets: list = field(default_factory=lambda: ["BTC"])  # Default to BTC only

    # Latency arbitrage settings
    min_edge_pct: float = 5.0  # Minimum edge to trigger (5%)
    min_time_remaining_sec: int = 120  # Don't trade in last 2 minutes
    cooldown_sec: int = 30  # Seconds between trades per symbol

    # Checkpoint configuration (seconds into 15-min window)
    # Two-checkpoint strategy: 7:30 (450s) primary, 9:00 (540s) fallback
    # 9:00 only executes if no position was taken at 7:30
    signal_checkpoints: list = field(default_factory=lambda: [450, 540])  # 7:30 and 9:00 only
    active_checkpoint: int = 450  # Primary checkpoint (7m30s)

    # Legacy entry timing (kept for backwards compatibility)
    entry_time_up_sec: int = 450  # When to consider UP entries
    entry_time_down_sec: int = 450  # When to consider DOWN entries

    # Confirmation requirements (momentum must align with price move)
    require_volume_confirmation: bool = True  # Volume delta must support direction
    require_orderbook_confirmation: bool = True  # Order book imbalance must support
    min_volume_delta_usd: float = 10000.0  # Minimum $10K volume delta
    min_orderbook_imbalance: float = 0.1  # 10% imbalance threshold

    # ==========================================================================
    # TIERED CONFIRMATION SYSTEM (New)
    # ==========================================================================
    # Minimum confirmations: 2 = more trades with tiered sizing, 3 = more conservative
    min_confirmations: int = 2  # Minimum confirmations to trade
    partial_size_pct: float = 50.0  # Position size % when only 2 confirmations (50%)
    edge_mandatory: bool = False  # If True, edge must pass or no trade

    # Indicator toggles (enable/disable each confirmation signal)
    use_edge: bool = True
    use_volume_delta: bool = True
    use_orderbook: bool = True
    use_vwap: bool = True
    use_rsi: bool = True
    use_adx: bool = True
    use_supertrend: bool = True

    # Indicator thresholds
    rsi_oversold: float = 30.0  # RSI below this = oversold (buy UP signal)
    rsi_overbought: float = 70.0  # RSI above this = overbought (buy DOWN signal)
    adx_trend_threshold: float = 25.0  # ADX above this = strong trend
    supertrend_multiplier: float = 3.0  # ATR multiplier for Supertrend

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure enabled_assets is always a list
        if not isinstance(d.get("enabled_assets"), list):
            d["enabled_assets"] = ["BTC"]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "TradingConfig":
        # Handle enabled_assets which might be missing in old state files
        if "enabled_assets" not in data:
            data["enabled_assets"] = ["BTC"]
        return cls(**data)


# ============================================================================
# TRADING SIGNALS
# ============================================================================

class SignalType(Enum):
    """Trading signal types"""
    HOLD = "HOLD"
    BUY_UP = "BUY_UP"
    BUY_MORE_UP = "BUY_MORE_UP"
    BUY_DOWN = "BUY_DOWN"
    BUY_MORE_DOWN = "BUY_MORE_DOWN"


@dataclass
class CheckpointSignal:
    """Signal generated at a checkpoint"""
    symbol: str
    checkpoint: str  # "3m", "7m", "10m", "12.5m" or "latency"
    timestamp: int
    signal: SignalType
    fair_value: float  # Calculated fair value for UP outcome
    market_price: float  # Current market price
    edge: float  # fair_value - market_price
    confidence: float  # 0-1
    momentum: dict  # Raw momentum data
    market_start: int = 0  # Market window start timestamp

    def to_dict(self) -> dict:
        # Generate slug from symbol and market_start
        slug = f"{self.symbol.lower()}-updown-15m-{self.market_start}" if self.market_start else ""
        return {
            "symbol": self.symbol,
            "slug": slug,
            "checkpoint": self.checkpoint,
            "timestamp": self.timestamp,
            "signal": self.signal.value,
            "fair_value": round(self.fair_value, 3),
            "market_price": round(self.market_price, 3),
            "edge": round(self.edge, 3),
            "confidence": round(self.confidence, 3),
            "momentum": self.momentum,
        }


@dataclass
class LatencyGap:
    """Tracks real-time gap between Binance and Polymarket"""
    symbol: str
    timestamp: int
    binance_open: float  # Binance price at market window open
    binance_current: float  # Current Binance price
    binance_change_pct: float  # % change since open
    implied_up_prob: float  # Probability UP wins based on current Binance move
    polymarket_up_price: float  # Current Polymarket UP price
    edge: float  # implied_up_prob - polymarket_up_price (positive = buy UP)
    time_remaining_sec: int  # Seconds until market closes

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "binance_open": round(self.binance_open, 2),
            "binance_current": round(self.binance_current, 2),
            "binance_change_pct": round(self.binance_change_pct, 4),
            "implied_up_prob": round(self.implied_up_prob, 3),
            "polymarket_up_price": round(self.polymarket_up_price, 3),
            "edge": round(self.edge, 3),
            "time_remaining_sec": self.time_remaining_sec,
        }


# ============================================================================
# POSITIONS AND TRADES
# ============================================================================

@dataclass
class Position:
    """An open paper trading position"""
    id: str
    symbol: str
    side: str  # "UP" or "DOWN"
    entry_price: float
    size: float  # Number of contracts
    cost_basis: float  # Total cost including slippage/commission
    entry_time: int
    market_start: int  # When the 15-min window started
    market_end: int  # When the 15-min window ends
    checkpoint: str  # Which checkpoint triggered entry

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L at current fair value (placeholder)"""
        return 0.0

    def to_dict(self) -> dict:
        slug = f"{self.symbol.lower()}-updown-15m-{self.market_start}"
        return {
            "id": self.id,
            "symbol": self.symbol,
            "slug": slug,
            "side": self.side,
            "entry_price": round(self.entry_price, 4),
            "size": round(self.size, 2),
            "cost_basis": round(self.cost_basis, 2),
            "entry_time": self.entry_time,
            "market_start": self.market_start,
            "market_end": self.market_end,
            "checkpoint": self.checkpoint,
        }


@dataclass
class Trade:
    """A completed paper trade with resolution"""
    id: str
    symbol: str
    side: str  # "UP" or "DOWN"
    entry_price: float
    exit_price: float  # 1.0 if correct, 0.0 if wrong
    size: float
    cost_basis: float
    settlement_value: float
    pnl: float  # Profit or loss in USD
    pnl_pct: float  # % return
    entry_time: int
    exit_time: int
    market_start: int
    market_end: int
    resolution: str  # "UP" or "DOWN" - what actually happened
    binance_open: float  # Binance price at market start
    binance_close: float  # Binance price at market end
    checkpoint: str
    signal_confidence: float

    def to_dict(self) -> dict:
        slug = f"{self.symbol.lower()}-updown-15m-{self.market_start}"
        return {
            "id": self.id,
            "symbol": self.symbol,
            "slug": slug,
            "side": self.side,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "size": round(self.size, 2),
            "cost_basis": round(self.cost_basis, 2),
            "settlement_value": round(self.settlement_value, 2),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "market_start": self.market_start,
            "market_end": self.market_end,
            "resolution": self.resolution,
            "binance_open": round(self.binance_open, 2),
            "binance_close": round(self.binance_close, 2),
            "checkpoint": self.checkpoint,
            "signal_confidence": round(self.signal_confidence, 3),
        }


# ============================================================================
# ACCOUNT STATE
# ============================================================================

@dataclass
class TradingAccount:
    """Paper trading account state"""
    balance: float
    starting_balance: float
    total_pnl: float = 0.0
    today_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    positions: list[Position] = field(default_factory=list)
    trade_history: list[Trade] = field(default_factory=list)
    last_reset_date: str = ""
    trading_halted: bool = False
    halt_reason: str = ""

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_return_pct(self) -> float:
        return (self.total_pnl / self.starting_balance) * 100

    def to_dict(self) -> dict:
        return {
            "balance": round(self.balance, 2),
            "starting_balance": round(self.starting_balance, 2),
            "total_pnl": round(self.total_pnl, 2),
            "today_pnl": round(self.today_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate * 100, 1),
            "positions": [p.to_dict() for p in self.positions],
            "recent_trades": [t.to_dict() for t in self.trade_history[:10]],
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
        }


# ============================================================================
# MARKET WINDOW TRACKING
# ============================================================================

@dataclass
class MarketWindow:
    """Tracks Binance prices for market resolution"""
    symbol: str
    start_time: int
    end_time: int
    open_price: Optional[float] = None
    close_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    resolved: bool = False
    resolution: Optional[str] = None  # "UP" or "DOWN"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "open_price": self.open_price,
            "close_price": self.close_price,
            "resolution": self.resolution,
            "resolved": self.resolved,
        }


# ============================================================================
# PAPER TRADING ENGINE
# ============================================================================

class TradingEngine:
    """
    Core paper trading engine using latency arbitrage.

    The strategy exploits the 30-60 second gap between Binance price movements
    and Polymarket price adjustments. When Binance moves significantly during
    a 15-minute window, we can predict the outcome before Polymarket adjusts.

    Handles:
    - Real-time latency gap detection
    - Position management
    - Risk enforcement
    - Market resolution tracking
    - Persistence
    """

    # Checkpoint times in seconds from market start
    # User-specified: 3m, 6m, 7:30m, 9m, 12m
    CHECKPOINTS = {
        "3m": 180,
        "6m": 360,
        "7:30m": 450,
        "9m": 540,
        "12m": 720,
    }

    # Confidence thresholds for each checkpoint (later = higher)
    CONFIDENCE_THRESHOLDS = {
        "3m": 0.55,
        "6m": 0.58,
        "7:30m": 0.62,
        "9m": 0.68,
        "12m": 0.75,
    }

    def __init__(self, data_dir: str = ".", ledger: Optional["TradeLedger"] = None):
        self.config = TradingConfig()
        self.account = TradingAccount(
            balance=self.config.starting_balance,
            starting_balance=self.config.starting_balance,
        )
        self.market_windows: dict[str, MarketWindow] = {}  # key: "SYMBOL_start_time"
        self.recent_signals: list[CheckpointSignal] = []
        self.latency_gaps: list[LatencyGap] = []  # Track recent gaps for UI
        self.data_dir = data_dir

        # Trade ledger for persistent storage
        self.ledger = ledger

        # Track last trade time per symbol for cooldown
        self._last_trade_time: dict[str, int] = {}

        # Track Binance open prices for each active window
        self._binance_opens: dict[str, float] = {}  # "SYMBOL_market_start" -> open price

        # Track which checkpoints have been triggered for each market window
        # Key: "SYMBOL_market_start", Value: set of checkpoint seconds that have fired
        self._triggered_checkpoints: dict[str, set[int]] = {}

        # Callbacks
        self.on_signal: Optional[Callable[[CheckpointSignal], None]] = None
        self.on_trade: Optional[Callable[[Trade], None]] = None
        self.on_position_open: Optional[Callable[[Position], None]] = None
        self.on_latency_gap: Optional[Callable[[LatencyGap], None]] = None
        self.on_alert: Optional[Callable[[str, str], None]] = None  # (title, message)

        # Trading mode (set by server.py from live_trading)
        self.trading_mode: str = "paper"  # "paper" or "live"

        # Load persisted state
        self._load_state()

        # Check for daily reset
        self._check_daily_reset()

    # -------------------------------------------------------------------------
    # SIGNAL CALCULATION
    # -------------------------------------------------------------------------

    def calculate_fair_value(self, momentum: dict) -> float:
        """
        Calculate fair value for UP outcome based on momentum.

        Uses:
        - Volume delta (buy vs sell volume)
        - Price change
        - Order book imbalance

        Returns probability 0.05 to 0.95
        """
        base = 0.5

        # Volume delta contribution (+/- 10% max)
        volume_delta = momentum.get("volume_delta", 0)
        # Normalize: $100K delta = 10% contribution
        vol_contribution = min(0.10, max(-0.10, volume_delta / 1_000_000))

        # Price change contribution (+/- 15% max)
        price_change = momentum.get("price_change_pct", 0) / 100  # Convert from %
        # 0.1% move = 15% contribution
        price_contribution = min(0.15, max(-0.15, price_change * 150))

        # Order book imbalance contribution (+/- 10% max)
        imbalance = momentum.get("orderbook_imbalance", 0)
        imbalance_contribution = imbalance * 0.10

        fair_value = base + vol_contribution + price_contribution + imbalance_contribution

        # Clamp to valid range
        return max(0.05, min(0.95, fair_value))

    def generate_signal(
        self,
        symbol: str,
        checkpoint: str,
        momentum: dict,
        market_price: float,
        has_existing_position: bool,
        market_start: int = 0,
    ) -> CheckpointSignal:
        """
        Generate trading signal for a checkpoint.

        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            checkpoint: "3m", "7m", "10m", "12.5m"
            momentum: Momentum data from MomentumCalculator
            market_price: Current Polymarket price for UP outcome
            has_existing_position: Whether we already have a position
            market_start: Unix timestamp when the 15-min window started
        """
        fair_value = self.calculate_fair_value(momentum)
        edge = fair_value - market_price

        # Calculate confidence from momentum strength
        direction = momentum.get("direction", "NEUTRAL")
        momentum_confidence = momentum.get("confidence", 0.5)

        # Edge-based confidence
        edge_confidence = min(1.0, abs(edge) * 5)  # 20% edge = 100% confidence

        # Combined confidence
        confidence = (momentum_confidence + edge_confidence) / 2

        # Determine signal
        threshold = self.CONFIDENCE_THRESHOLDS.get(checkpoint, 0.60)
        signal = SignalType.HOLD

        if confidence >= threshold:
            if edge > 0.03:  # 3% edge for UP
                if direction == "UP":
                    if has_existing_position:
                        signal = SignalType.BUY_MORE_UP
                    else:
                        signal = SignalType.BUY_UP
            elif edge < -0.03:  # 3% edge for DOWN
                if direction == "DOWN":
                    if has_existing_position:
                        signal = SignalType.BUY_MORE_DOWN
                    else:
                        signal = SignalType.BUY_DOWN

        checkpoint_signal = CheckpointSignal(
            symbol=symbol,
            checkpoint=checkpoint,
            timestamp=int(time.time()),
            signal=signal,
            fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            confidence=confidence,
            momentum=momentum,
            market_start=market_start,
        )

        # Track recent signals
        self.recent_signals.append(checkpoint_signal)
        if len(self.recent_signals) > 100:
            self.recent_signals = self.recent_signals[-100:]

        # Fire callback
        if self.on_signal:
            self.on_signal(checkpoint_signal)

        return checkpoint_signal

    def process_checkpoint(
        self,
        symbol: str,
        checkpoint: str,
        momentum: dict,
        market_price: float,
        market_start: int,
        market_end: int,
    ) -> Optional[CheckpointSignal]:
        """
        Process a checkpoint and potentially open/add to position.

        Called by the server when a checkpoint time is reached.
        """
        if not self.config.enabled:
            return None

        if self.account.trading_halted:
            return None

        # Check if this asset is enabled
        if not self.is_asset_enabled(symbol):
            return None

        # Check if we have an existing position for this market window
        has_position = any(
            p.symbol == symbol and p.market_start == market_start
            for p in self.account.positions
        )

        # Generate signal
        signal = self.generate_signal(
            symbol=symbol,
            checkpoint=checkpoint,
            momentum=momentum,
            market_price=market_price,
            has_existing_position=has_position,
            market_start=market_start,
        )

        # Execute based on signal
        if signal.signal == SignalType.BUY_UP:
            self._open_position(symbol, "UP", market_price, market_start, market_end, checkpoint, signal.confidence)
        elif signal.signal == SignalType.BUY_DOWN:
            self._open_position(symbol, "DOWN", 1 - market_price, market_start, market_end, checkpoint, signal.confidence)
        elif signal.signal == SignalType.BUY_MORE_UP:
            self._add_to_position(symbol, "UP", market_price, market_start, checkpoint, signal.confidence)
        elif signal.signal == SignalType.BUY_MORE_DOWN:
            self._add_to_position(symbol, "DOWN", 1 - market_price, market_start, checkpoint, signal.confidence)

        return signal

    # -------------------------------------------------------------------------
    # LATENCY ARBITRAGE
    # -------------------------------------------------------------------------

    def record_window_open(self, symbol: str, market_start: int, binance_price: float):
        """
        Record the Binance price at the start of a market window.
        This is the reference price for calculating the move.
        """
        key = f"{symbol}_{market_start}"
        self._binance_opens[key] = binance_price

        # Clean up old checkpoint tracking data (windows older than 1 hour)
        cutoff = int(time.time()) - 3600
        old_keys = [k for k in self._triggered_checkpoints.keys() if int(k.split("_")[1]) < cutoff]
        for k in old_keys:
            del self._triggered_checkpoints[k]

        old_opens = [k for k in self._binance_opens.keys() if int(k.split("_")[1]) < cutoff]
        for k in old_opens:
            del self._binance_opens[k]

    def calculate_implied_probability(self, price_change_pct: float, time_remaining_sec: int) -> float:
        """
        Calculate implied probability that UP will win based on:
        1. Current price change from window open
        2. Time remaining in the window

        A positive price_change_pct means price is higher than open,
        so UP is more likely to win.

        With more time remaining, there's more chance for reversal,
        so we dampen the probability slightly.
        """
        # Base probability from current price change
        # A 0.1% move has ~65% chance of staying up/down
        # A 0.5% move has ~85% chance
        base_prob = 0.5

        # Sigmoid-like scaling for price impact
        # Small moves have less certainty, large moves approach certainty
        abs_change = abs(price_change_pct)

        if abs_change < 0.01:
            price_impact = 0
        elif abs_change < 0.05:
            price_impact = abs_change * 2  # 0.05% move = 10% edge
        elif abs_change < 0.2:
            price_impact = 0.1 + (abs_change - 0.05) * 1.5  # 0.2% move = 32% edge
        else:
            price_impact = min(0.45, 0.32 + (abs_change - 0.2) * 0.5)  # Max out at 95%

        # Time dampening: more time = less certainty
        # Last 2 min: no dampening. First 5 min: 30% dampening
        time_factor = 1.0
        if time_remaining_sec > 600:  # > 10 min left
            time_factor = 0.7
        elif time_remaining_sec > 300:  # 5-10 min left
            time_factor = 0.85
        elif time_remaining_sec > 120:  # 2-5 min left
            time_factor = 0.95

        adjusted_impact = price_impact * time_factor

        # Direction: positive change = UP more likely
        if price_change_pct >= 0:
            return min(0.95, base_prob + adjusted_impact)
        else:
            return max(0.05, base_prob - adjusted_impact)

    def check_latency_opportunity(
        self,
        symbol: str,
        binance_current: float,
        polymarket_up_price: float,
        market_start: int,
        market_end: int,
    ) -> Optional[LatencyGap]:
        """
        Check for a latency arbitrage opportunity.

        Called frequently (every 1-2 seconds) with current prices.
        Returns a LatencyGap if there's a significant edge.
        """
        now = int(time.time())
        time_remaining = market_end - now

        # Get the Binance open price for this window
        key = f"{symbol}_{market_start}"
        binance_open = self._binance_opens.get(key)

        if binance_open is None:
            return None

        # Calculate price change
        price_change_pct = ((binance_current - binance_open) / binance_open) * 100

        # Calculate implied probability
        implied_prob = self.calculate_implied_probability(price_change_pct, time_remaining)

        # Calculate edge
        edge = implied_prob - polymarket_up_price

        gap = LatencyGap(
            symbol=symbol,
            timestamp=now,
            binance_open=binance_open,
            binance_current=binance_current,
            binance_change_pct=price_change_pct,
            implied_up_prob=implied_prob,
            polymarket_up_price=polymarket_up_price,
            edge=edge,
            time_remaining_sec=time_remaining,
        )

        # Track for UI
        self.latency_gaps.append(gap)
        if len(self.latency_gaps) > 100:
            self.latency_gaps = self.latency_gaps[-100:]

        # Fire callback for UI updates
        if self.on_latency_gap:
            self.on_latency_gap(gap)

        return gap

    def process_latency_opportunity(
        self,
        symbol: str,
        binance_current: float,
        polymarket_up_price: float,
        market_start: int,
        market_end: int,
        momentum: Optional[dict] = None,
    ) -> Optional[CheckpointSignal]:
        """
        Process a potential latency arbitrage opportunity.

        This is the main entry point called every 1-2 seconds.
        It checks for edge and CONFIRMS with momentum indicators before executing.

        Confirmation requires:
        1. Price edge above threshold (Binance moved, Polymarket lagging)
        2. Volume delta supports direction (aggressive buyers/sellers)
        3. Order book imbalance supports direction
        """
        if not self.config.enabled:
            return None

        if self.account.trading_halted:
            return None

        # Check if this asset is enabled
        if not self.is_asset_enabled(symbol):
            return None

        now = int(time.time())
        elapsed_sec = now - market_start
        time_remaining = market_end - now

        # Don't trade in the last min_time_remaining seconds
        if time_remaining < self.config.min_time_remaining_sec:
            return None

        # =====================================================================
        # CHECKPOINT-BASED SIGNAL GENERATION
        # Only generate signals at configured checkpoint times (±3 second tolerance)
        # =====================================================================
        window_key = f"{symbol}_{market_start}"
        if window_key not in self._triggered_checkpoints:
            self._triggered_checkpoints[window_key] = set()

        # Find if we're at a checkpoint (within 3 second tolerance)
        current_checkpoint = None
        checkpoint_tolerance = 3  # seconds
        for checkpoint_sec in self.config.signal_checkpoints:
            if abs(elapsed_sec - checkpoint_sec) <= checkpoint_tolerance:
                # Check if this checkpoint hasn't been triggered yet
                if checkpoint_sec not in self._triggered_checkpoints[window_key]:
                    current_checkpoint = checkpoint_sec
                    break

        # If not at a checkpoint, don't generate signal
        if current_checkpoint is None:
            # Debug: Log every 30s what elapsed time is
            if elapsed_sec % 30 < 2:
                print(f"[PT] {symbol} elapsed={elapsed_sec}s, checkpoints={self.config.signal_checkpoints}", flush=True)
            return None

        print(f"[PT] {symbol} HIT CHECKPOINT {current_checkpoint}s at elapsed={elapsed_sec}s", flush=True)

        # Mark this checkpoint as triggered
        self._triggered_checkpoints[window_key].add(current_checkpoint)

        # Format checkpoint string for display
        cp_min = current_checkpoint // 60
        cp_sec = current_checkpoint % 60
        if cp_sec == 30:
            checkpoint_str = f"{cp_min}m30s"
        elif cp_sec > 0:
            checkpoint_str = f"{cp_min}m{cp_sec}s"
        else:
            checkpoint_str = f"{cp_min}m"

        # Check cooldown
        last_trade = self._last_trade_time.get(symbol, 0)
        if now - last_trade < self.config.cooldown_sec:
            print(f"[PT] {symbol} checkpoint {checkpoint_str}: COOLDOWN (last trade {now - last_trade}s ago)", flush=True)
            return None

        # Check if we already have a position for this window
        has_position = any(
            p.symbol == symbol and p.market_start == market_start
            for p in self.account.positions
        )
        if has_position:
            print(f"[PT] {symbol} checkpoint {checkpoint_str}: ALREADY HAS POSITION", flush=True)
            return None

        # Check for latency opportunity
        gap = self.check_latency_opportunity(
            symbol=symbol,
            binance_current=binance_current,
            polymarket_up_price=polymarket_up_price,
            market_start=market_start,
            market_end=market_end,
        )

        if gap is None:
            print(f"[PT] {symbol} checkpoint {checkpoint_str}: NO LATENCY GAP", flush=True)
            return None

        # Determine intended direction from price gap
        price_direction = "UP" if gap.binance_change_pct > 0 else "DOWN"

        # =====================================================================
        # TIERED CONFIRMATION SYSTEM
        # Count how many indicators confirm the trade, allow partial sizing
        # =====================================================================

        min_edge = self.config.min_edge_pct / 100
        abs_edge = abs(gap.edge)

        volume_delta = momentum.get("volume_delta", 0) if momentum else 0
        orderbook_imbalance = momentum.get("orderbook_imbalance", 0) if momentum else 0

        # Get new indicators from momentum dict
        vwap_signal = momentum.get("vwap_signal", "NEUTRAL") if momentum else "NEUTRAL"
        rsi = momentum.get("rsi", 50.0) if momentum else 50.0
        adx = momentum.get("adx", 0.0) if momentum else 0.0
        supertrend_dir = momentum.get("supertrend_direction", "NEUTRAL") if momentum else "NEUTRAL"

        # Count confirmations
        confirmations = []

        # 1. Edge confirmation
        if self.config.use_edge and abs_edge >= min_edge:
            confirmations.append("edge")
        elif self.config.edge_mandatory:
            # Edge is mandatory but failed
            print(f"[PT] {symbol} checkpoint {checkpoint_str}: EDGE MANDATORY FAIL ({abs_edge:.1%} < {min_edge:.1%})", flush=True)
            return None

        # 2. Volume delta confirmation
        if self.config.use_volume_delta:
            vol_ok = False
            if price_direction == "UP" and volume_delta >= self.config.min_volume_delta_usd:
                vol_ok = True
            elif price_direction == "DOWN" and volume_delta <= -self.config.min_volume_delta_usd:
                vol_ok = True
            if vol_ok:
                confirmations.append("volume")

        # 3. Orderbook imbalance confirmation
        if self.config.use_orderbook:
            book_ok = False
            if price_direction == "UP" and orderbook_imbalance >= self.config.min_orderbook_imbalance:
                book_ok = True
            elif price_direction == "DOWN" and orderbook_imbalance <= -self.config.min_orderbook_imbalance:
                book_ok = True
            if book_ok:
                confirmations.append("orderbook")

        # 4. VWAP confirmation
        if self.config.use_vwap and vwap_signal == price_direction:
            confirmations.append("vwap")

        # 5. RSI confirmation (oversold for UP, overbought for DOWN)
        if self.config.use_rsi:
            rsi_ok = False
            if price_direction == "UP" and rsi < self.config.rsi_oversold:
                rsi_ok = True
            elif price_direction == "DOWN" and rsi > self.config.rsi_overbought:
                rsi_ok = True
            if rsi_ok:
                confirmations.append("rsi")

        # 6. ADX confirmation (strong trend)
        if self.config.use_adx and adx >= self.config.adx_trend_threshold:
            confirmations.append("adx")

        # 7. Supertrend confirmation
        if self.config.use_supertrend and supertrend_dir == price_direction:
            confirmations.append("supertrend")

        conf_count = len(confirmations)

        # Check if we have enough confirmations
        if conf_count < self.config.min_confirmations:
            print(f"[PT] {symbol} checkpoint {checkpoint_str}: INSUFFICIENT CONFIRMATIONS ({conf_count}/{self.config.min_confirmations}) - Got: {confirmations}", flush=True)
            return None

        # Determine position multiplier based on confirmation count
        # 3+ confirmations = full size, 2 confirmations = partial size
        if conf_count >= 3:
            position_multiplier = 1.0
        else:
            position_multiplier = self.config.partial_size_pct / 100

        print(f"[PT] {symbol} checkpoint {checkpoint_str}: {conf_count} CONFIRMATIONS ({confirmations}) -> {position_multiplier*100:.0f}% size", flush=True)

        # =====================================================================
        # Confirmations passed - execute trade
        # =====================================================================

        # Determine signal type
        if gap.edge > 0:
            signal_type = SignalType.BUY_UP
            entry_price = polymarket_up_price
            side = "UP"
        else:
            signal_type = SignalType.BUY_DOWN
            entry_price = 1 - polymarket_up_price
            side = "DOWN"

        # Calculate confidence from confirmation strength
        edge_confidence = min(1.0, abs_edge * 5) if "edge" in confirmations else 0
        volume_confidence = min(1.0, abs(volume_delta) / 50000) if "volume" in confirmations else 0
        book_confidence = min(1.0, abs(orderbook_imbalance) * 3) if "orderbook" in confirmations else 0
        vwap_confidence = 0.8 if "vwap" in confirmations else 0
        rsi_confidence = 0.7 if "rsi" in confirmations else 0
        adx_confidence = min(1.0, adx / 50) if "adx" in confirmations else 0
        supertrend_confidence = 0.8 if "supertrend" in confirmations else 0

        # Average confidence across active confirmations
        conf_values = [c for c in [edge_confidence, volume_confidence, book_confidence,
                                    vwap_confidence, rsi_confidence, adx_confidence,
                                    supertrend_confidence] if c > 0]
        confidence = sum(conf_values) / len(conf_values) if conf_values else 0.5

        # Build momentum dict from gap data + confirmations
        gap_momentum = {
            "direction": price_direction,
            "confidence": round(confidence, 3),
            "volume_delta": volume_delta,
            "price_change_pct": gap.binance_change_pct,
            "orderbook_imbalance": orderbook_imbalance,
            "binance_open": gap.binance_open,
            "binance_current": gap.binance_current,
            "time_remaining": gap.time_remaining_sec,
            "confirmed_by": confirmations,
            "confirmation_count": conf_count,
            "position_multiplier": position_multiplier,
            "checkpoint_sec": current_checkpoint,
            # New indicators
            "vwap_signal": vwap_signal,
            "rsi": rsi,
            "adx": adx,
            "supertrend_direction": supertrend_dir,
        }

        signal = CheckpointSignal(
            symbol=symbol,
            checkpoint=checkpoint_str,
            timestamp=now,
            signal=signal_type,
            fair_value=gap.implied_up_prob,
            market_price=polymarket_up_price,
            edge=gap.edge,
            confidence=confidence,
            momentum=gap_momentum,
            market_start=market_start,
        )

        # Track signal
        self.recent_signals.append(signal)
        if len(self.recent_signals) > 100:
            self.recent_signals = self.recent_signals[-100:]

        # Fire callback (always, for UI display)
        if self.on_signal:
            self.on_signal(signal)

        # Only execute trades at the active checkpoint
        is_active_checkpoint = current_checkpoint == self.config.active_checkpoint
        if is_active_checkpoint:
            self._open_position(symbol, side, entry_price, market_start, market_end, checkpoint_str, confidence, position_multiplier)
            self._last_trade_time[symbol] = now
            print(f"[Signal] {symbol} @ {checkpoint_str}: EXECUTED {side} trade | Edge: {gap.edge:.1%} | Size: {position_multiplier*100:.0f}%", flush=True)
        else:
            print(f"[Signal] {symbol} @ {checkpoint_str}: {signal_type.value} (signal only) | Edge: {gap.edge:.1%}", flush=True)

        return signal

    def get_latency_gaps(self, limit: int = 20) -> list[dict]:
        """Get recent latency gaps for UI"""
        return [g.to_dict() for g in self.latency_gaps[-limit:]]

    # -------------------------------------------------------------------------
    # POSITION MANAGEMENT
    # -------------------------------------------------------------------------

    def _calculate_position_size(self, is_adding: bool = False, position_multiplier: float = 1.0) -> float:
        """
        Calculate position size based on risk management.

        Uses the LESSER of:
        1. 2% of account balance
        2. $5K hard cap (liquidity constraint)

        Then applies position_multiplier for tiered confirmation sizing.
        """
        pct_based = self.account.balance * (self.config.max_position_pct / 100)
        hard_cap = self.config.max_position_usd

        max_position = min(pct_based, hard_cap)

        if is_adding:
            # When adding to position, use 50% of max
            max_position = max_position * 0.5

        # Apply position multiplier (for tiered confirmation: 2 conf = 50%, 3+ conf = 100%)
        return max_position * position_multiplier

    def is_asset_enabled(self, symbol: str) -> bool:
        """Check if an asset is enabled for trading"""
        return symbol.upper() in [a.upper() for a in self.config.enabled_assets]

    def set_enabled_assets(self, assets: list[str]):
        """Hot-reload enabled assets without restart"""
        self.config.enabled_assets = [a.upper() for a in assets]
        self._save_state()

    def _apply_slippage_and_commission(self, price: float, size: float, is_buy: bool) -> tuple[float, float, float]:
        """
        Apply slippage and commission to trade.

        Returns:
            Tuple of (adjusted_price, total_cost, commission)
        """
        slippage_mult = 1 + (self.config.slippage_pct / 100) if is_buy else 1 - (self.config.slippage_pct / 100)
        adjusted_price = price * slippage_mult

        # Commission is on dollar value
        commission = size * adjusted_price * (self.config.commission_pct / 100)
        total_cost = size * adjusted_price + commission

        return adjusted_price, total_cost, commission

    def _open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        market_start: int,
        market_end: int,
        checkpoint: str,
        confidence: float,
        position_multiplier: float = 1.0,
    ):
        """Open a new position"""
        position_value = self._calculate_position_size(is_adding=False, position_multiplier=position_multiplier)
        size = position_value / price  # Number of contracts

        adjusted_price, cost_basis, commission = self._apply_slippage_and_commission(price, size, is_buy=True)

        # Check if we have enough balance
        if cost_basis > self.account.balance:
            return

        position_id = f"{symbol}_{market_start}_{int(time.time())}"

        position = Position(
            id=position_id,
            symbol=symbol,
            side=side,
            entry_price=adjusted_price,
            size=size,
            cost_basis=cost_basis,
            entry_time=int(time.time()),
            market_start=market_start,
            market_end=market_end,
            checkpoint=checkpoint,
        )

        # Deduct from balance
        self.account.balance -= cost_basis
        self.account.positions.append(position)

        # Record commission fee to ledger
        if self.ledger and commission > 0:
            try:
                self.ledger.record_fee(
                    account="paper",
                    amount=commission,
                    description=f"Entry commission: {symbol} {side} @ {adjusted_price:.4f}",
                    reference_id=position_id,
                    notes=f"Checkpoint: {checkpoint}, Commission rate: {self.config.commission_pct}%",
                )
            except Exception as e:
                print(f"[Trading] Failed to record fee to ledger: {e}")

        # Ensure market window is tracked
        window_key = f"{symbol}_{market_start}"
        if window_key not in self.market_windows:
            self.market_windows[window_key] = MarketWindow(
                symbol=symbol,
                start_time=market_start,
                end_time=market_end,
            )

        # Fire callback
        if self.on_position_open:
            self.on_position_open(position)

        # Save state
        self._save_state()

    def _add_to_position(
        self,
        symbol: str,
        side: str,
        price: float,
        market_start: int,
        checkpoint: str,
        confidence: float,
    ):
        """Add to an existing position"""
        # Find existing position
        existing = None
        for p in self.account.positions:
            if p.symbol == symbol and p.market_start == market_start and p.side == side:
                existing = p
                break

        if not existing:
            return

        position_value = self._calculate_position_size(is_adding=True)
        size = position_value / price

        adjusted_price, additional_cost, commission = self._apply_slippage_and_commission(price, size, is_buy=True)

        # Check balance
        if additional_cost > self.account.balance:
            return

        # Update position with weighted average price
        total_size = existing.size + size
        existing.entry_price = (existing.entry_price * existing.size + adjusted_price * size) / total_size
        existing.size = total_size
        existing.cost_basis += additional_cost

        # Deduct from balance
        self.account.balance -= additional_cost

        # Record commission fee to ledger
        if self.ledger and commission > 0:
            try:
                self.ledger.record_fee(
                    account="paper",
                    amount=commission,
                    description=f"Add-on commission: {symbol} {side} @ {adjusted_price:.4f}",
                    reference_id=existing.id,
                    notes=f"Checkpoint: {checkpoint}, Commission rate: {self.config.commission_pct}%",
                )
            except Exception as e:
                print(f"[Trading] Failed to record fee to ledger: {e}")

        # Save state
        self._save_state()

    # -------------------------------------------------------------------------
    # MARKET RESOLUTION
    # -------------------------------------------------------------------------

    def record_binance_price(self, symbol: str, price: float, market_start: int, is_open: bool):
        """Record Binance price at market open/close for resolution"""
        window_key = f"{symbol}_{market_start}"
        window = self.market_windows.get(window_key)

        if not window:
            return

        if is_open:
            window.open_price = price
        else:
            window.close_price = price

    def resolve_market(self, symbol: str, market_start: int, market_end: int, binance_open: float, binance_close: float):
        """
        Resolve a market window and settle positions.

        Called when a 15-minute window ends.
        """
        window_key = f"{symbol}_{market_start}"
        window = self.market_windows.get(window_key)

        if window:
            window.open_price = binance_open
            window.close_price = binance_close
            window.resolved = True
            window.resolution = "UP" if binance_close > binance_open else "DOWN"

        resolution = "UP" if binance_close > binance_open else "DOWN"

        # Settle all positions for this market window
        positions_to_remove = []

        for position in self.account.positions:
            if position.symbol == symbol and position.market_start == market_start:
                # Position settles at $1 if correct, $0 if wrong
                is_correct = position.side == resolution
                exit_price = 1.0 if is_correct else 0.0
                settlement_value = position.size * exit_price
                pnl = settlement_value - position.cost_basis
                pnl_pct = (pnl / position.cost_basis) * 100 if position.cost_basis > 0 else 0

                trade = Trade(
                    id=position.id,
                    symbol=symbol,
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    size=position.size,
                    cost_basis=position.cost_basis,
                    settlement_value=settlement_value,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    entry_time=position.entry_time,
                    exit_time=market_end,
                    market_start=market_start,
                    market_end=market_end,
                    resolution=resolution,
                    binance_open=binance_open,
                    binance_close=binance_close,
                    checkpoint=position.checkpoint,
                    signal_confidence=0.0,  # Would need to store this
                )

                # Update account
                self.account.balance += settlement_value
                self.account.total_pnl += pnl
                self.account.today_pnl += pnl
                self.account.total_trades += 1

                if is_correct:
                    self.account.winning_trades += 1
                else:
                    self.account.losing_trades += 1

                # Add to history
                self.account.trade_history.insert(0, trade)
                if len(self.account.trade_history) > 100:
                    self.account.trade_history = self.account.trade_history[:100]

                positions_to_remove.append(position)

                # Record to ledger for persistent storage
                if self.ledger:
                    try:
                        from trade_ledger import create_trade_record_from_paper
                        trade_record = create_trade_record_from_paper(trade)
                        self.ledger.record_trade(trade_record)

                        # Also record P&L as a transaction for account tracking
                        self.ledger.record_trade_pnl(
                            account="paper",
                            trade_id=trade.id,
                            pnl=pnl,
                            description=f"Trade P&L: {symbol} {position.side} {'WIN' if is_correct else 'LOSS'}",
                        )
                    except Exception as e:
                        print(f"[Trading] Failed to record trade to ledger: {e}")

                # Fire trade callback
                if self.on_trade:
                    self.on_trade(trade)

                # Individual trade alerts removed - using end-of-market summary instead

        # Remove settled positions
        for p in positions_to_remove:
            self.account.positions.remove(p)

        # Check daily loss limit
        self._check_loss_limit()

        # Send end-of-market summary alert ONLY if we had a position
        if self.on_alert and positions_to_remove:
            price_move = ((binance_close - binance_open) / binance_open) * 100
            move_emoji = "📈" if resolution == "UP" else "📉"
            mode_tag = "🔴 LIVE" if self.trading_mode == "live" else "📝 PAPER"

            # Had position(s) - show result
            total_pnl = sum(t.pnl for t in self.account.trade_history[:len(positions_to_remove)])
            wins = sum(1 for t in self.account.trade_history[:len(positions_to_remove)] if t.pnl > 0)
            losses = len(positions_to_remove) - wins
            result_emoji = "✅" if total_pnl >= 0 else "❌"
            self.on_alert(
                f"[{mode_tag}] MARKET END {symbol}",
                f"{move_emoji} Resolved: {resolution} ({price_move:+.2f}%)\n"
                f"{result_emoji} P&L: ${total_pnl:+.2f} ({wins}W/{losses}L)\n"
                f"💰 Balance: ${self.account.balance:,.2f}"
            )

        # Save state
        self._save_state()

    # -------------------------------------------------------------------------
    # RISK MANAGEMENT
    # -------------------------------------------------------------------------

    def _check_loss_limit(self):
        """Check if daily loss limit has been hit"""
        loss_limit = self.config.starting_balance * (self.config.daily_loss_limit_pct / 100)

        if self.account.today_pnl < -loss_limit:
            self.account.trading_halted = True
            self.account.halt_reason = f"Daily loss limit reached: ${abs(self.account.today_pnl):.2f}"

    def _check_daily_reset(self):
        """Check and perform daily reset at UTC midnight"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self.account.last_reset_date != today:
            self.account.today_pnl = 0.0
            self.account.trading_halted = False
            self.account.halt_reason = ""
            self.account.last_reset_date = today
            self._save_state()

    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------

    def update_config(self, **kwargs):
        """Update configuration settings"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._save_state()

    def reset_account(self):
        """Reset account to starting state"""
        self.account = TradingAccount(
            balance=self.config.starting_balance,
            starting_balance=self.config.starting_balance,
            last_reset_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
        self.market_windows.clear()
        self.recent_signals.clear()
        self._save_state()

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    def _get_state_path(self) -> str:
        return os.path.join(self.data_dir, "paper_trading_state.json")

    def _save_state(self):
        """Save state to JSON file"""
        state = {
            "trading_mode": self.trading_mode,
            "config": self.config.to_dict(),
            "account": {
                "balance": self.account.balance,
                "starting_balance": self.account.starting_balance,
                "total_pnl": self.account.total_pnl,
                "today_pnl": self.account.today_pnl,
                "total_trades": self.account.total_trades,
                "winning_trades": self.account.winning_trades,
                "losing_trades": self.account.losing_trades,
                "last_reset_date": self.account.last_reset_date,
                "trading_halted": self.account.trading_halted,
                "halt_reason": self.account.halt_reason,
                "positions": [p.to_dict() for p in self.account.positions],
                "trade_history": [t.to_dict() for t in self.account.trade_history],
            },
        }

        try:
            with open(self._get_state_path(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[Trading] Error saving state: {e}")

    def _load_state(self):
        """Load state from JSON file"""
        try:
            path = self._get_state_path()
            if not os.path.exists(path):
                return

            with open(path, "r") as f:
                state = json.load(f)

            # Load trading mode
            if "trading_mode" in state:
                self.trading_mode = state["trading_mode"]
                print(f"[PT] Loaded trading_mode: {self.trading_mode}", flush=True)

            # Load config
            if "config" in state:
                print(f"[PT] Loading config from state: signal_checkpoints={state['config'].get('signal_checkpoints')}", flush=True)
                self.config = TradingConfig.from_dict(state["config"])
                print(f"[PT] Loaded config: signal_checkpoints={self.config.signal_checkpoints}", flush=True)

            # Load account
            if "account" in state:
                acc = state["account"]
                self.account = TradingAccount(
                    balance=acc.get("balance", self.config.starting_balance),
                    starting_balance=acc.get("starting_balance", self.config.starting_balance),
                    total_pnl=acc.get("total_pnl", 0.0),
                    today_pnl=acc.get("today_pnl", 0.0),
                    total_trades=acc.get("total_trades", 0),
                    winning_trades=acc.get("winning_trades", 0),
                    losing_trades=acc.get("losing_trades", 0),
                    last_reset_date=acc.get("last_reset_date", ""),
                    trading_halted=acc.get("trading_halted", False),
                    halt_reason=acc.get("halt_reason", ""),
                )

                # Load positions
                for p_data in acc.get("positions", []):
                    self.account.positions.append(Position(
                        id=p_data["id"],
                        symbol=p_data["symbol"],
                        side=p_data["side"],
                        entry_price=p_data["entry_price"],
                        size=p_data["size"],
                        cost_basis=p_data["cost_basis"],
                        entry_time=p_data["entry_time"],
                        market_start=p_data["market_start"],
                        market_end=p_data["market_end"],
                        checkpoint=p_data["checkpoint"],
                    ))

                # Load trade history
                for t_data in acc.get("trade_history", []):
                    self.account.trade_history.append(Trade(
                        id=t_data["id"],
                        symbol=t_data["symbol"],
                        side=t_data["side"],
                        entry_price=t_data["entry_price"],
                        exit_price=t_data["exit_price"],
                        size=t_data["size"],
                        cost_basis=t_data["cost_basis"],
                        settlement_value=t_data["settlement_value"],
                        pnl=t_data["pnl"],
                        pnl_pct=t_data["pnl_pct"],
                        entry_time=t_data["entry_time"],
                        exit_time=t_data["exit_time"],
                        market_start=t_data["market_start"],
                        market_end=t_data["market_end"],
                        resolution=t_data["resolution"],
                        binance_open=t_data["binance_open"],
                        binance_close=t_data["binance_close"],
                        checkpoint=t_data["checkpoint"],
                        signal_confidence=t_data.get("signal_confidence", 0.0),
                    ))

        except Exception as e:
            print(f"[Trading] Error loading state: {e}")

    # -------------------------------------------------------------------------
    # API METHODS
    # -------------------------------------------------------------------------

    def get_account_summary(self) -> dict:
        """Get account summary for API"""
        return self.account.to_dict()

    def get_config(self) -> dict:
        """Get current config"""
        return self.config.to_dict()

    def get_recent_signals(self, limit: int = 20) -> list[dict]:
        """Get recent trading signals"""
        return [s.to_dict() for s in self.recent_signals[-limit:]]

    def get_trade_history(self, limit: int = 50) -> list[dict]:
        """Get trade history"""
        return [t.to_dict() for t in self.account.trade_history[:limit]]

    def get_positions(self) -> list[dict]:
        """Get open positions"""
        return [p.to_dict() for p in self.account.positions]

    # -------------------------------------------------------------------------
    # WHALE FOLLOWING
    # -------------------------------------------------------------------------

    def process_whale_bias(
        self,
        whale_name: str,
        symbol: str,
        bias: str,  # "UP" or "DOWN"
        bias_confidence: float,
        market_start: int,
        market_end: int,
        detection_latency_sec: float,
        polymarket_price: float,  # Current UP price
    ) -> Optional[CheckpointSignal]:
        """
        Process a whale bias signal and potentially open a position.

        Called when WhaleTradeDetector detects that a whale (e.g., gabagool22)
        has taken a directional position in a market.

        Args:
            whale_name: Name of the whale (e.g., "gabagool22")
            symbol: Crypto symbol (BTC, ETH, etc.)
            bias: Detected bias - "UP" or "DOWN"
            bias_confidence: How confident we are in the bias (0-1)
            market_start: Market window start timestamp
            market_end: Market window end timestamp
            detection_latency_sec: How long after whale's first trade we detected
            polymarket_price: Current Polymarket UP price

        Returns:
            Signal if position opened, None otherwise
        """
        if not self.config.enabled:
            return None

        if self.account.trading_halted:
            return None

        # Check if this asset is enabled
        if not self.is_asset_enabled(symbol):
            return None

        now = int(time.time())
        time_remaining = market_end - now

        # Need sufficient time remaining (at least 3 minutes)
        if time_remaining < 180:
            return None

        # Don't follow if detection latency is too high (stale signal)
        max_latency = 120  # 2 minutes max
        if detection_latency_sec > max_latency:
            print(f"[WhaleFollow] Skipping {symbol} - latency too high: {detection_latency_sec:.0f}s > {max_latency}s")
            return None

        # Check if we already have a position for this window
        has_position = any(
            p.symbol == symbol and p.market_start == market_start
            for p in self.account.positions
        )
        if has_position:
            return None

        # Require minimum confidence
        min_confidence = 0.55
        if bias_confidence < min_confidence:
            print(f"[WhaleFollow] Skipping {symbol} - confidence too low: {bias_confidence:.1%} < {min_confidence:.1%}")
            return None

        # Calculate entry price
        if bias == "UP":
            entry_price = polymarket_price
            side = "UP"
            signal_type = SignalType.BUY_UP
        else:
            entry_price = 1 - polymarket_price
            side = "DOWN"
            signal_type = SignalType.BUY_DOWN

        # Create checkpoint string for tracking
        checkpoint_str = f"whale:{whale_name}"

        # Build momentum dict for signal
        momentum = {
            "direction": bias,
            "confidence": bias_confidence,
            "whale_name": whale_name,
            "detection_latency_sec": detection_latency_sec,
            "time_remaining": time_remaining,
            "strategy": "whale_following",
        }

        signal = CheckpointSignal(
            symbol=symbol,
            checkpoint=checkpoint_str,
            timestamp=now,
            signal=signal_type,
            fair_value=bias_confidence,  # Use confidence as proxy for fair value
            market_price=polymarket_price,
            edge=bias_confidence - 0.5,  # Edge relative to 50/50
            confidence=bias_confidence,
            momentum=momentum,
            market_start=market_start,
        )

        # Track signal
        self.recent_signals.append(signal)
        if len(self.recent_signals) > 100:
            self.recent_signals = self.recent_signals[-100:]

        # Fire callback
        if self.on_signal:
            self.on_signal(signal)

        # Open position
        self._open_position(
            symbol=symbol,
            side=side,
            price=entry_price,
            market_start=market_start,
            market_end=market_end,
            checkpoint=checkpoint_str,
            confidence=bias_confidence,
        )

        print(
            f"[WhaleFollow] EXECUTED: Follow {whale_name} -> {side} on {symbol} "
            f"(conf={bias_confidence:.1%}, latency={detection_latency_sec:.0f}s, remaining={time_remaining}s)"
        )

        return signal
