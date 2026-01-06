"""
Whale Following Strategy - Mirror trades from top performers like gabagool22

Strategy: When a tracked whale buys/sells on a 15-min market, mirror their trade
within a configurable delay window.

Real-Time Detection Analysis (2026-01-05):
- API polling latency: ~595ms average
- Trade visibility: ~60-90 seconds after execution
- gabagool22 pattern: Waits 2-3 min after market open, then burst trades
- Position bias: NOT purely hedged - shows directional bias each market
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable
from collections import deque, defaultdict
import logging
import time

from config import CRYPTO_WHALE_WALLETS, PolymarketAPI

logger = logging.getLogger(__name__)


# =============================================================================
# Real-Time Trade Detection
# =============================================================================


@dataclass
class WhaleMarketBias:
    """Detected position bias for a whale in a specific market"""
    whale_name: str
    symbol: str
    market_start: int
    market_end: int
    market_slug: str
    net_up: float  # Net UP position (buys - sells)
    net_down: float  # Net DOWN position (buys - sells)
    bias: str  # "UP", "DOWN", or "NEUTRAL"
    bias_confidence: float  # 0-1, how strong the bias is
    total_trades: int
    total_volume_usd: float
    first_trade_time: int
    last_trade_time: int
    detection_time: int  # When we detected this bias
    detection_latency_sec: float  # Time from whale's first trade to detection

    def to_dict(self) -> dict:
        return {
            "whale_name": self.whale_name,
            "symbol": self.symbol,
            "market_start": self.market_start,
            "market_end": self.market_end,
            "market_slug": self.market_slug,
            "net_up": self.net_up,
            "net_down": self.net_down,
            "bias": self.bias,
            "bias_confidence": self.bias_confidence,
            "total_trades": self.total_trades,
            "total_volume_usd": self.total_volume_usd,
            "first_trade_time": self.first_trade_time,
            "last_trade_time": self.last_trade_time,
            "detection_time": self.detection_time,
            "detection_latency_sec": self.detection_latency_sec,
        }


class WhaleTradeDetector:
    """
    Real-time detector for whale trades on Polymarket 15-min markets.

    Polls the Polymarket API to detect when tracked whales (like gabagool22)
    take positions, then analyzes their net bias (UP vs DOWN).

    Key findings from analysis:
    - gabagool22 waits ~2-3 minutes after market open before trading
    - They burst trade for ~15-20 seconds (600+ trades/minute)
    - They show clear directional bias, not purely hedged
    - API visibility is ~60-90 seconds after trade execution
    - With 500ms polling, we detect trades within ~1-2 minutes of execution
    """

    def __init__(
        self,
        whales: list[str] = None,
        poll_interval_sec: float = 1.0,
        on_bias_detected: Optional[Callable[[WhaleMarketBias], None]] = None,
    ):
        """
        Args:
            whales: List of whale names to track (default: ["gabagool22"])
            poll_interval_sec: How often to poll for new trades
            on_bias_detected: Callback when a whale's bias is detected
        """
        self.whales = whales or ["gabagool22"]
        self.poll_interval_sec = poll_interval_sec
        self.on_bias_detected = on_bias_detected

        # Track last seen trade per whale to detect new trades
        self._last_trade_id: dict[str, str] = {}

        # Track detected biases per market to avoid duplicate signals
        # Key: (whale_name, market_slug)
        self._detected_biases: dict[tuple, WhaleMarketBias] = {}

        # Track trades per market for bias calculation
        # Key: (whale_name, market_slug) -> list of trades
        self._market_trades: dict[tuple, list] = defaultdict(list)

        # Running state
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None

        # Stats
        self.polls_made = 0
        self.trades_detected = 0
        self.biases_generated = 0

    async def start(self):
        """Start the real-time detection loop"""
        if self._running:
            return

        self._running = True
        self._session = aiohttp.ClientSession()

        logger.info(f"[WhaleDetector] Starting real-time detection for: {self.whales}")

        while self._running:
            try:
                await self._poll_all_whales()
            except Exception as e:
                logger.error(f"[WhaleDetector] Poll error: {e}")

            await asyncio.sleep(self.poll_interval_sec)

    async def stop(self):
        """Stop the detection loop"""
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None

    async def _poll_all_whales(self):
        """Poll all tracked whales for new trades"""
        self.polls_made += 1

        for whale_name in self.whales:
            try:
                await self._poll_whale(whale_name)
            except Exception as e:
                logger.warning(f"[WhaleDetector] Error polling {whale_name}: {e}")

    async def _poll_whale(self, whale_name: str):
        """Poll a single whale for new trades"""
        whale_info = CRYPTO_WHALE_WALLETS.get(whale_name)
        if not whale_info:
            return

        wallet = whale_info["address"]

        # Fetch recent trades
        url = f"https://data-api.polymarket.com/trades"
        params = {"maker": wallet, "limit": 50}

        async with self._session.get(url, params=params, timeout=10) as resp:
            if resp.status != 200:
                return
            trades = await resp.json()

        if not trades:
            return

        # Check for new trades
        latest_trade = trades[0]
        trade_id = f"{latest_trade.get('timestamp')}_{latest_trade.get('transactionHash')}"

        if self._last_trade_id.get(whale_name) == trade_id:
            return  # No new trades

        self._last_trade_id[whale_name] = trade_id

        # Process new trades for 15-min markets
        now = int(time.time())

        for trade in trades:
            slug = trade.get("eventSlug", "") or trade.get("slug", "")

            # Filter for 15-min crypto markets
            if "updown-15m" not in slug:
                continue

            # Extract symbol
            symbol = None
            for sym in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
                if sym.lower() in slug.lower():
                    symbol = sym
                    break

            if not symbol:
                continue

            # Extract market timing
            market_start = None
            parts = slug.split("-")
            for part in parts:
                if part.isdigit() and len(part) >= 10:
                    market_start = int(part)
                    break

            if not market_start:
                continue

            market_end = market_start + 900  # 15 minutes

            # Skip past markets
            if market_end < now:
                continue

            # Add to market trades
            market_key = (whale_name, slug)
            trade_ts = trade.get("timestamp", now)

            trade_data = {
                "timestamp": trade_ts,
                "side": trade.get("side", "").upper(),
                "outcome": trade.get("outcome", ""),
                "price": float(trade.get("price", 0)),
                "size": float(trade.get("size", 0)),
            }

            # Avoid duplicates
            existing_times = [t["timestamp"] for t in self._market_trades[market_key]]
            if trade_ts not in existing_times:
                self._market_trades[market_key].append(trade_data)
                self.trades_detected += 1

            # Calculate bias after accumulating trades
            await self._calculate_bias(
                whale_name, symbol, market_start, market_end, slug
            )

    async def _calculate_bias(
        self,
        whale_name: str,
        symbol: str,
        market_start: int,
        market_end: int,
        market_slug: str,
    ):
        """Calculate the whale's position bias for a market"""
        market_key = (whale_name, market_slug)
        trades = self._market_trades.get(market_key, [])

        if not trades:
            return

        # Calculate net positions
        net_up = 0.0
        net_down = 0.0
        total_volume = 0.0

        for t in trades:
            side = t["side"]
            outcome = t["outcome"]
            size = t["size"]
            price = t["price"]
            volume = size * price

            total_volume += volume

            if outcome == "Up":
                if side == "BUY":
                    net_up += size
                else:
                    net_up -= size
            elif outcome == "Down":
                if side == "BUY":
                    net_down += size
                else:
                    net_down -= size

        # Determine bias
        # Require significant difference to declare a bias
        total_exposure = abs(net_up) + abs(net_down)
        if total_exposure < 10:  # Too small
            return

        if net_up > net_down * 1.3:  # 30% more UP exposure
            bias = "UP"
            bias_confidence = net_up / total_exposure
        elif net_down > net_up * 1.3:
            bias = "DOWN"
            bias_confidence = net_down / total_exposure
        else:
            bias = "NEUTRAL"
            bias_confidence = 0.5

        # Check if we already signaled this market
        if market_key in self._detected_biases:
            existing = self._detected_biases[market_key]
            # Only update if bias changed or confidence increased significantly
            if existing.bias == bias and existing.bias_confidence >= bias_confidence - 0.1:
                return

        # Create bias signal
        now = int(time.time())
        trade_times = [t["timestamp"] for t in trades]
        first_trade = min(trade_times)
        last_trade = max(trade_times)

        market_bias = WhaleMarketBias(
            whale_name=whale_name,
            symbol=symbol,
            market_start=market_start,
            market_end=market_end,
            market_slug=market_slug,
            net_up=net_up,
            net_down=net_down,
            bias=bias,
            bias_confidence=bias_confidence,
            total_trades=len(trades),
            total_volume_usd=total_volume,
            first_trade_time=first_trade,
            last_trade_time=last_trade,
            detection_time=now,
            detection_latency_sec=now - first_trade,
        )

        self._detected_biases[market_key] = market_bias
        self.biases_generated += 1

        time_remaining = market_end - now

        logger.info(
            f"[WhaleDetector] BIAS DETECTED: {whale_name} -> {bias} on {symbol} "
            f"(conf={bias_confidence:.1%}, trades={len(trades)}, "
            f"latency={market_bias.detection_latency_sec:.0f}s, remaining={time_remaining}s)"
        )

        if self.on_bias_detected:
            self.on_bias_detected(market_bias)

    def get_current_bias(self, whale_name: str, symbol: str) -> Optional[WhaleMarketBias]:
        """Get the current detected bias for a whale in an active market"""
        now = int(time.time())

        for (whale, slug), bias in self._detected_biases.items():
            if whale == whale_name and bias.symbol == symbol:
                if bias.market_end > now:  # Still active
                    return bias

        return None

    def get_stats(self) -> dict:
        """Get detector statistics"""
        return {
            "running": self._running,
            "whales_tracked": self.whales,
            "polls_made": self.polls_made,
            "trades_detected": self.trades_detected,
            "biases_generated": self.biases_generated,
            "active_markets": len([b for b in self._detected_biases.values()
                                   if b.market_end > int(time.time())]),
        }

    def get_active_biases(self) -> list[WhaleMarketBias]:
        """Get all active market biases"""
        now = int(time.time())
        return [b for b in self._detected_biases.values() if b.market_end > now]


@dataclass
class WhaleFollowSignal:
    """Signal generated when a whale makes a trade we want to follow"""
    whale_name: str
    symbol: str
    side: str  # "UP" or "DOWN"
    action: str  # "BUY" or "SELL"
    whale_price: float
    whale_size: float
    whale_usd: float
    timestamp: int
    market_start: int
    market_end: int
    delay_sec: float  # How long after whale's trade we generated this signal

    def to_dict(self) -> dict:
        return {
            "whale_name": self.whale_name,
            "symbol": self.symbol,
            "side": self.side,
            "action": self.action,
            "whale_price": self.whale_price,
            "whale_size": self.whale_size,
            "whale_usd": self.whale_usd,
            "timestamp": self.timestamp,
            "market_start": self.market_start,
            "market_end": self.market_end,
            "delay_sec": self.delay_sec,
        }


@dataclass
class WhaleFollowTrade:
    """A simulated trade following a whale"""
    signal: WhaleFollowSignal
    our_entry_price: float
    our_size: float
    our_cost_basis: float
    entry_time: int
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    settlement_value: Optional[float] = None
    pnl: Optional[float] = None
    resolution: Optional[str] = None
    is_winner: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "signal": self.signal.to_dict(),
            "our_entry_price": self.our_entry_price,
            "our_size": self.our_size,
            "our_cost_basis": self.our_cost_basis,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "settlement_value": self.settlement_value,
            "pnl": self.pnl,
            "resolution": self.resolution,
            "is_winner": self.is_winner,
        }


@dataclass
class WhaleFollowConfig:
    """Configuration for whale following strategy"""
    enabled: bool = False
    follow_whales: list = field(default_factory=lambda: ["gabagool22"])

    # Position sizing
    position_size_usd: float = 25.0  # Fixed position size
    max_position_per_market: float = 50.0  # Max exposure per market

    # Timing
    min_time_remaining_sec: int = 180  # Need at least 3 min left in market
    max_follow_delay_sec: float = 30.0  # Max time after whale trade to follow

    # Filters
    min_whale_usd: float = 50.0  # Minimum whale trade size to follow
    symbols: list = field(default_factory=lambda: ["BTC", "ETH", "SOL"])

    # Only follow buys (not sells) - whales often sell to take profit
    follow_buys_only: bool = True


@dataclass
class BacktestResult:
    """Results from backtesting whale following strategy"""
    whale_name: str
    period_start: int
    period_end: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_cost: float
    roi_pct: float
    win_rate: float
    avg_pnl_per_trade: float
    trades: list  # List of WhaleFollowTrade

    def to_dict(self) -> dict:
        return {
            "whale_name": self.whale_name,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "total_cost": self.total_cost,
            "roi_pct": self.roi_pct,
            "win_rate": self.win_rate,
            "avg_pnl_per_trade": self.avg_pnl_per_trade,
            "trades": [t.to_dict() for t in self.trades[-50:]],  # Last 50 trades
        }


class WhaleFollowingStrategy:
    """
    Strategy that follows whale trades on Polymarket 15-min markets.

    When gabagool22 or other tracked whales buy UP/DOWN on a crypto market,
    we mirror the trade (with configurable delay and position sizing).
    """

    def __init__(
        self,
        config: WhaleFollowConfig = None,
        on_signal: Optional[Callable[[WhaleFollowSignal], None]] = None,
    ):
        self.config = config or WhaleFollowConfig()
        self.on_signal = on_signal

        # Track recent signals to avoid duplicates
        self._recent_signals: deque[str] = deque(maxlen=100)

        # Active positions by market
        self._positions: dict[str, WhaleFollowTrade] = {}

        # Historical trades for this session
        self._trades: list[WhaleFollowTrade] = []

        # Stats
        self._total_pnl = 0.0
        self._total_trades = 0
        self._winning_trades = 0

    def process_whale_trade(
        self,
        whale_name: str,
        symbol: str,
        outcome: str,  # "Up" or "Down"
        side: str,  # "BUY" or "SELL"
        price: float,
        size: float,
        timestamp: int,
        market_start: int,
        market_end: int,
        current_market_price: float,  # Current price we can get
    ) -> Optional[WhaleFollowSignal]:
        """
        Process a whale trade and potentially generate a follow signal.

        Returns a signal if we should follow this trade, None otherwise.
        """
        if not self.config.enabled:
            return None

        # Check if we're following this whale
        if whale_name not in self.config.follow_whales:
            return None

        # Check symbol filter
        if symbol not in self.config.symbols:
            return None

        # Check if it's a buy (if configured to only follow buys)
        if self.config.follow_buys_only and side != "BUY":
            return None

        # Check minimum trade size
        usd_value = size * price
        if usd_value < self.config.min_whale_usd:
            return None

        # Check time remaining in market
        now = int(datetime.now(timezone.utc).timestamp())
        time_remaining = market_end - now
        if time_remaining < self.config.min_time_remaining_sec:
            return None

        # Check delay from whale trade
        delay = now - timestamp
        if delay > self.config.max_follow_delay_sec:
            return None

        # Create unique signal ID to avoid duplicates
        signal_id = f"{whale_name}_{symbol}_{outcome}_{timestamp}"
        if signal_id in self._recent_signals:
            return None
        self._recent_signals.append(signal_id)

        # Generate signal
        signal = WhaleFollowSignal(
            whale_name=whale_name,
            symbol=symbol,
            side=outcome.upper(),  # "UP" or "DOWN"
            action=side,  # "BUY" or "SELL"
            whale_price=price,
            whale_size=size,
            whale_usd=usd_value,
            timestamp=now,
            market_start=market_start,
            market_end=market_end,
            delay_sec=delay,
        )

        logger.info(
            f"[WhaleFollow] Signal: Follow {whale_name} {side} {outcome} on {symbol} "
            f"(${usd_value:.0f}, delay={delay:.1f}s)"
        )

        if self.on_signal:
            self.on_signal(signal)

        return signal

    def get_stats(self) -> dict:
        """Get current strategy statistics"""
        return {
            "enabled": self.config.enabled,
            "following": self.config.follow_whales,
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "total_pnl": self._total_pnl,
            "win_rate": self._winning_trades / max(1, self._total_trades),
            "active_positions": len(self._positions),
        }


async def fetch_whale_trades(
    wallet: str,
    limit: int = 500,
    market_filter: Optional[str] = None,  # e.g., "updown-15m" to filter 15-min markets
) -> list[dict]:
    """
    Fetch historical trades for a whale wallet from Polymarket API.

    Args:
        wallet: Wallet address
        limit: Max trades to fetch
        market_filter: Optional filter for market slugs

    Returns:
        List of trade dictionaries
    """
    trades = []

    async with aiohttp.ClientSession() as session:
        # Polymarket trades API with pagination
        cursor = None
        fetched = 0

        while fetched < limit:
            url = f"{PolymarketAPI.TRADES}"
            params = {
                "user": wallet.lower(),
                "limit": min(100, limit - fetched),
            }
            if cursor:
                params["cursor"] = cursor

            try:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to fetch trades: {resp.status}")
                        break

                    data = await resp.json()

                    if not data:
                        break

                    for trade in data:
                        # Apply market filter if specified
                        if market_filter:
                            market_slug = trade.get("market_slug", "") or trade.get("slug", "")
                            if market_filter not in market_slug:
                                continue

                        trades.append(trade)
                        fetched += 1

                        if fetched >= limit:
                            break

                    # Check for pagination
                    if len(data) < 100:
                        break

                    # Use last trade's ID as cursor (simplified pagination)
                    cursor = data[-1].get("id")
                    if not cursor:
                        break

            except Exception as e:
                logger.error(f"Error fetching whale trades: {e}")
                break

    return trades


async def fetch_binance_price_at(
    session: aiohttp.ClientSession,
    symbol: str,
    timestamp: int,
) -> Optional[float]:
    """
    Fetch the Binance price for a symbol at a specific timestamp.

    Uses Binance klines API to get 1-minute candle containing the timestamp.
    """
    binance_symbol = f"{symbol}USDT".upper()

    try:
        # Get 1-minute kline starting at the timestamp
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": binance_symbol,
            "interval": "1m",
            "startTime": timestamp * 1000,
            "limit": 1,
        }

        async with session.get(url, params=params, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data and len(data) > 0:
                    # Kline format: [open_time, open, high, low, close, ...]
                    return float(data[0][4])  # Close price
    except Exception as e:
        logger.debug(f"Could not fetch Binance price for {symbol} at {timestamp}: {e}")

    return None


async def fetch_market_resolution_from_binance(
    session: aiohttp.ClientSession,
    symbol: str,
    market_start: int,
    market_end: int,
) -> Optional[str]:
    """
    Determine the actual market resolution by checking Binance price movement.

    For 15-min UP/DOWN markets:
    - If end_price > start_price: UP wins
    - If end_price <= start_price: DOWN wins

    Returns "UP" or "DOWN" based on actual price movement.
    """
    start_price = await fetch_binance_price_at(session, symbol, market_start)
    end_price = await fetch_binance_price_at(session, symbol, market_end)

    if start_price is None or end_price is None:
        logger.warning(f"Could not fetch prices for {symbol} {market_start}-{market_end}")
        return None

    resolution = "UP" if end_price > start_price else "DOWN"
    logger.debug(
        f"Market {symbol} {market_start}: ${start_price:.2f} -> ${end_price:.2f} = {resolution}"
    )
    return resolution


async def backtest_whale_following(
    whale_name: str,
    days: int = 7,
    position_size: float = 25.0,
) -> BacktestResult:
    """
    Backtest the whale following strategy on historical data.

    Fetches the whale's trades on 15-min markets and determines outcomes
    using actual Binance price data to see if markets resolved UP or DOWN.

    Args:
        whale_name: Name of whale to follow (e.g., "gabagool22")
        days: Number of days to backtest
        position_size: USD position size per trade

    Returns:
        BacktestResult with performance metrics
    """
    # Get wallet address
    whale_info = CRYPTO_WHALE_WALLETS.get(whale_name)
    if not whale_info:
        raise ValueError(f"Unknown whale: {whale_name}")

    wallet = whale_info["address"]

    # Fetch historical trades (15-min markets only)
    logger.info(f"Fetching trades for {whale_name} ({wallet[:10]}...)")
    raw_trades = await fetch_whale_trades(
        wallet=wallet,
        limit=1000,
        market_filter="updown-15m",
    )

    logger.info(f"Found {len(raw_trades)} trades on 15-min markets")

    # Parse and filter trades
    now = int(datetime.now(timezone.utc).timestamp())
    cutoff = now - (days * 24 * 60 * 60)

    # First pass: group trades by market window to ensure consistent resolution
    # Key: (symbol, market_start) -> list of trades
    market_trades: dict[tuple, list] = {}

    for raw in raw_trades:
        try:
            # Parse trade data
            timestamp = raw.get("timestamp")
            if isinstance(timestamp, str):
                # Parse ISO format
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp = int(dt.timestamp())

            if timestamp < cutoff:
                continue  # Too old

            # Extract trade details
            side = raw.get("side", "").upper()
            if side not in ["BUY", "SELL"]:
                continue

            # Only follow buys
            if side != "BUY":
                continue

            outcome = raw.get("outcome", "")
            if outcome not in ["Up", "Down"]:
                continue

            price = float(raw.get("price", 0))
            size = float(raw.get("size", 0))

            if price <= 0 or size <= 0:
                continue

            # Parse market info
            market_slug = raw.get("market_slug", "") or raw.get("slug", "")

            # Extract symbol from slug (e.g., "btc-updown-15m-1234567890")
            symbol = None
            for sym in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
                if sym.lower() in market_slug.lower():
                    symbol = sym
                    break

            if not symbol:
                continue

            # Try to extract market timing from slug
            # Format: {symbol}-updown-15m-{start_timestamp}
            parts = market_slug.split("-")
            market_start = None
            for part in parts:
                if part.isdigit() and len(part) >= 10:
                    market_start = int(part)
                    break

            if not market_start:
                # Approximate from trade timestamp
                # Markets start at :00, :15, :30, :45
                trade_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                minute = trade_dt.minute
                market_minute = (minute // 15) * 15
                market_start = int(trade_dt.replace(minute=market_minute, second=0, microsecond=0).timestamp())

            # Group by market window
            market_key = (symbol, market_start, market_slug)
            if market_key not in market_trades:
                market_trades[market_key] = []

            market_trades[market_key].append({
                "timestamp": timestamp,
                "side": side,
                "outcome": outcome,
                "price": price,
                "size": size,
                "market_slug": market_slug,
                "symbol": symbol,
                "market_start": market_start,
            })

        except Exception as e:
            logger.warning(f"Error parsing trade: {e}")
            continue

    # Second pass: fetch actual resolutions from Binance and create trades
    backtest_trades = []
    resolution_cache: dict[tuple, str] = {}  # (symbol, market_start) -> resolution

    async with aiohttp.ClientSession() as session:
        for (symbol, market_start, market_slug), trades in market_trades.items():
            market_end = market_start + 900  # 15 minutes

            # Skip markets that haven't ended yet
            if market_end > now:
                continue

            # Get actual resolution from Binance price data
            cache_key = (symbol, market_start)
            if cache_key in resolution_cache:
                market_resolution = resolution_cache[cache_key]
            else:
                market_resolution = await fetch_market_resolution_from_binance(
                    session, symbol, market_start, market_end
                )
                if market_resolution is None:
                    logger.warning(f"Could not determine resolution for {symbol} {market_start}")
                    continue
                resolution_cache[cache_key] = market_resolution

            # Process each trade in this market
            for trade_data in trades:
                timestamp = trade_data["timestamp"]
                outcome = trade_data["outcome"]
                price = trade_data["price"]
                size = trade_data["size"]

                # Calculate our entry (assume 1% slippage from whale's price)
                our_price = min(0.99, price * 1.01)
                our_size = position_size / our_price
                our_cost = position_size

                # Determine if this trade wins based on actual market resolution
                # If whale bought UP and market resolved UP -> win
                # If whale bought DOWN and market resolved DOWN -> win
                whale_bet = outcome.upper()  # "UP" or "DOWN"
                is_winner = (whale_bet == market_resolution)

                if is_winner:
                    exit_price = 1.0
                    settlement = our_size * 1.0
                else:
                    exit_price = 0.0
                    settlement = 0.0

                pnl = settlement - our_cost

                signal = WhaleFollowSignal(
                    whale_name=whale_name,
                    symbol=symbol,
                    side=whale_bet,
                    action="BUY",
                    whale_price=price,
                    whale_size=size,
                    whale_usd=size * price,
                    timestamp=timestamp,
                    market_start=market_start,
                    market_end=market_end,
                    delay_sec=5.0,  # Assumed delay
                )

                trade = WhaleFollowTrade(
                    signal=signal,
                    our_entry_price=our_price,
                    our_size=our_size,
                    our_cost_basis=our_cost,
                    entry_time=timestamp + 5,  # 5 sec delay
                    exit_time=market_end,
                    exit_price=exit_price,
                    settlement_value=settlement,
                    pnl=pnl,
                    resolution=market_resolution,
                    is_winner=is_winner,
                )

                backtest_trades.append(trade)

    # Sort by timestamp (most recent first)
    backtest_trades.sort(key=lambda t: t.entry_time, reverse=True)

    # Calculate statistics
    total_trades = len(backtest_trades)
    winning_trades = sum(1 for t in backtest_trades if t.is_winner)
    losing_trades = total_trades - winning_trades
    total_pnl = sum(t.pnl or 0 for t in backtest_trades)
    total_cost = sum(t.our_cost_basis for t in backtest_trades)

    return BacktestResult(
        whale_name=whale_name,
        period_start=cutoff,
        period_end=now,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        total_pnl=total_pnl,
        total_cost=total_cost,
        roi_pct=(total_pnl / max(1, total_cost)) * 100,
        win_rate=winning_trades / max(1, total_trades),
        avg_pnl_per_trade=total_pnl / max(1, total_trades),
        trades=backtest_trades,
    )


async def fetch_whale_positions_pnl(
    whale_name: str,
) -> dict:
    """
    Fetch a whale's actual positions and P&L from Polymarket.

    IMPORTANT: The positions API only returns CURRENT/RECENT positions,
    not historical all-time P&L. For gabagool22, the activity page shows
    $516K+ all-time P&L, but this API only shows today's positions.

    The API returns:
    - cashPnl: Realized + unrealized P&L for the position
    - percentPnl: Percentage return
    - initialValue: Total cost basis
    - currentValue: Current market value

    Returns:
        dict with performance metrics based on current positions
    """
    whale_info = CRYPTO_WHALE_WALLETS.get(whale_name)
    if not whale_info:
        raise ValueError(f"Unknown whale: {whale_name}")

    wallet = whale_info["address"]

    import aiohttp
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        # Get positions
        url = f"https://data-api.polymarket.com/positions?user={wallet}"
        async with session.get(url, headers=headers, timeout=30) as resp:
            if resp.status != 200:
                raise ValueError(f"Failed to fetch positions: {resp.status}")
            positions = await resp.json()

        # Get recent trade count to show activity level
        trades_url = f"https://data-api.polymarket.com/trades?user={wallet}&limit=500"
        async with session.get(trades_url, headers=headers, timeout=30) as resp:
            recent_trades = await resp.json() if resp.status == 200 else []

    # Filter for 15-min crypto markets
    crypto_positions = [
        p for p in positions
        if '15' in p.get('title', '') or 'Up or Down' in p.get('title', '')
    ]

    # Calculate stats from real positions using correct field names
    total_cash_pnl = 0.0
    total_initial = 0.0
    total_current = 0.0
    wins = 0
    losses = 0
    position_details = []

    for p in crypto_positions:
        # Use the correct field names from the API
        cash_pnl = float(p.get('cashPnl', 0) or 0)
        pct_pnl = float(p.get('percentPnl', 0) or 0)
        initial = float(p.get('initialValue', 0) or 0)
        current = float(p.get('currentValue', 0) or 0)
        title = p.get('title', '')
        asset = p.get('asset', '')  # YES or NO (UP or DOWN)

        total_cash_pnl += cash_pnl
        total_initial += initial
        total_current += current

        # Count wins/losses based on P&L
        if cash_pnl > 5:  # Small threshold to avoid noise
            wins += 1
        elif cash_pnl < -5:
            losses += 1

        position_details.append({
            "outcome": asset,
            "cost": initial,
            "current_value": current,
            "cash_pnl": cash_pnl,
            "pct_pnl": pct_pnl,
            "title": title[:60],
            "is_winner": cash_pnl > 0 if abs(cash_pnl) > 1 else None,
        })

    # Sort by absolute P&L (largest first)
    position_details.sort(key=lambda x: abs(x.get('cash_pnl', 0)), reverse=True)

    # Calculate ROI
    roi_pct = (total_cash_pnl / max(1, total_initial)) * 100 if total_initial > 0 else 0

    # Count today's trades (all should be from today based on positions)
    today_trades = len(recent_trades)

    return {
        "whale_name": whale_name,
        "wallet": wallet,
        # Position stats
        "total_positions": len(crypto_positions),
        "settled_positions": wins + losses,
        "winning_trades": wins,
        "losing_trades": losses,
        # Financial metrics
        "total_cost": total_initial,
        "current_value": total_current,
        "total_pnl": total_cash_pnl,
        "roi_pct": roi_pct,
        "win_rate": wins / max(1, wins + losses) if (wins + losses) > 0 else 0,
        # Activity metrics
        "recent_trade_count": today_trades,
        # Disclaimer
        "data_note": "Shows current/recent positions only. Historical all-time P&L not available via API.",
        # Position breakdown
        "positions": position_details[:20],  # Top 20 by P&L
    }


async def fetch_whale_market_history(
    whale_name: str,
    limit: int = 100,
) -> list[dict]:
    """
    Fetch a whale's 15-min market trade history with enriched data.

    Returns trades grouped by market window with outcome information.
    """
    whale_info = CRYPTO_WHALE_WALLETS.get(whale_name)
    if not whale_info:
        return []

    wallet = whale_info["address"]

    raw_trades = await fetch_whale_trades(
        wallet=wallet,
        limit=limit * 2,  # Fetch more to filter
        market_filter="updown-15m",
    )

    # Group by market and enrich
    markets = {}

    for trade in raw_trades:
        market_slug = trade.get("market_slug", "") or trade.get("slug", "")
        if not market_slug:
            continue

        if market_slug not in markets:
            markets[market_slug] = {
                "slug": market_slug,
                "trades": [],
                "total_usd": 0,
                "net_position": {},  # {outcome: net_size}
            }

        markets[market_slug]["trades"].append(trade)

        size = float(trade.get("size", 0))
        price = float(trade.get("price", 0))
        side = trade.get("side", "").upper()
        outcome = trade.get("outcome", "")

        usd = size * price
        markets[market_slug]["total_usd"] += usd

        if outcome:
            if outcome not in markets[market_slug]["net_position"]:
                markets[market_slug]["net_position"][outcome] = 0

            if side == "BUY":
                markets[market_slug]["net_position"][outcome] += size
            else:
                markets[market_slug]["net_position"][outcome] -= size

    # Convert to list and sort by recency
    result = list(markets.values())[:limit]

    return result
