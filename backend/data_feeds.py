"""
Real-time data feeds for crypto prices, order books, and whale tracking.
Uses WebSockets for low-latency updates.
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from collections import deque
import aiohttp
import websockets

from retry import (
    retry_http_request,
    connection_monitor,
    HTTP_RETRY_CONFIG,
    RetryConfig,
)
from config import (
    CryptoExchangeAPI,
    PolymarketAPI,
    CRYPTO_WHALE_WALLETS,
    DEFAULT_CONFIG,
)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class OHLCV:
    """OHLCV candle data"""
    timestamp: int  # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            "time": self.timestamp // 1000,  # TradingView uses seconds
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class Trade:
    """Individual trade"""
    timestamp: int
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    exchange: str


@dataclass
class OrderBookLevel:
    """Single price level in order book"""
    price: float
    size: float


@dataclass
class OrderBook:
    """Order book snapshot"""
    symbol: str
    timestamp: int
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.asks[0].price + self.bids[0].price) / 2
        return 0.0

    @property
    def imbalance(self) -> float:
        """Bid/ask imbalance - positive = more buy pressure"""
        bid_depth = sum(l.size for l in self.bids[:5])
        ask_depth = sum(l.size for l in self.asks[:5])
        total = bid_depth + ask_depth
        if total == 0:
            return 0.0
        return (bid_depth - ask_depth) / total


@dataclass
class MomentumSignal:
    """Aggregated momentum signal for a symbol"""
    symbol: str
    timestamp: int
    volume_delta: float = 0.0  # Cumulative buy - sell volume
    price_change_pct: float = 0.0  # % change in window
    orderbook_imbalance: float = 0.0  # -1 to 1
    liquidation_pressure: float = 0.0  # Net liquidation direction
    cross_exchange_lead: Optional[str] = None  # Which exchange is leading

    # New technical indicators
    vwap: float = 0.0  # Volume Weighted Average Price
    vwap_signal: str = "NEUTRAL"  # "UP", "DOWN", "NEUTRAL" based on price vs VWAP
    rsi: float = 50.0  # Relative Strength Index (0-100)
    adx: float = 0.0  # Average Directional Index (trend strength, 0-100)
    supertrend_direction: str = "NEUTRAL"  # Supertrend signal
    supertrend_value: float = 0.0  # Supertrend line value

    @property
    def direction(self) -> str:
        """Overall direction signal"""
        score = (
            (1 if self.volume_delta > 0 else -1) * 0.35 +
            (1 if self.price_change_pct > 0 else -1) * 0.25 +
            self.orderbook_imbalance * 0.20 +
            self.liquidation_pressure * 0.20
        )
        if score > 0.15:
            return "UP"
        elif score < -0.15:
            return "DOWN"
        return "NEUTRAL"

    @property
    def confidence(self) -> float:
        """Confidence score 0-1"""
        score = abs(
            self.volume_delta / 1_000_000 * 0.35 +
            self.price_change_pct * 10 * 0.25 +
            self.orderbook_imbalance * 0.20 +
            self.liquidation_pressure * 0.20
        )
        return min(score, 1.0)


@dataclass
class WhaleTrade:
    """Trade by a tracked whale"""
    whale_name: str
    wallet: str
    market_title: str
    market_slug: str
    outcome: str
    side: str
    size: float
    price: float
    timestamp: int
    tx_hash: str
    icon: str = ""

    @property
    def usd_value(self) -> float:
        return self.size * self.price

    def to_dict(self) -> dict:
        return {
            "whale": self.whale_name,
            "wallet": self.wallet[:10] + "...",
            "market": self.market_title,
            "slug": self.market_slug,
            "outcome": self.outcome,
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "usd_value": self.usd_value,
            "timestamp": self.timestamp,
            "tx_hash": self.tx_hash[:16],
            "icon": self.icon,
        }


# ============================================================================
# BINANCE WEBSOCKET FEED
# ============================================================================

class BinanceFeed:
    """
    Real-time Binance WebSocket feed for crypto prices.
    Provides: trades, klines (OHLCV), order book updates.
    """

    def __init__(self, symbols: list[str]):
        self.symbols = [s.lower() for s in symbols]
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False

        # Data stores
        self.trades: dict[str, deque] = {s: deque(maxlen=1000) for s in self.symbols}
        self.candles: dict[str, list[OHLCV]] = {s: [] for s in self.symbols}
        self.orderbooks: dict[str, OrderBook] = {}

        # Callbacks
        self.on_trade: Optional[Callable[[str, Trade], None]] = None
        self.on_candle: Optional[Callable[[str, OHLCV], None]] = None
        self.on_orderbook: Optional[Callable[[str, OrderBook], None]] = None

    def _build_stream_url(self) -> str:
        """Build combined stream URL for all symbols"""
        streams = []
        for symbol in self.symbols:
            streams.extend([
                f"{symbol}@trade",           # Real-time trades
                f"{symbol}@kline_1m",        # 1-minute candles
                f"{symbol}@depth5@100ms",    # Top 5 order book levels
            ])
        stream_str = "/".join(streams)
        # Use /stream?streams= for combined streams
        return f"wss://stream.binance.com:9443/stream?streams={stream_str}"

    async def connect(self):
        """Connect to Binance WebSocket with exponential backoff reconnection"""
        url = self._build_stream_url()
        self.running = True
        reconnect_delay = 1  # Start with 1 second
        max_reconnect_delay = 60  # Max 60 seconds
        consecutive_failures = 0

        while self.running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                ) as ws:
                    self.ws = ws
                    reconnect_delay = 1  # Reset on successful connection
                    consecutive_failures = 0
                    connection_monitor.mark_success("binance_ws")
                    print(f"[Binance] Connected to {len(self.symbols)} streams")

                    async for message in ws:
                        if not self.running:
                            break
                        connection_monitor.mark_success("binance_ws")
                        await self._handle_message(json.loads(message))

            except websockets.ConnectionClosed as e:
                consecutive_failures += 1
                connection_monitor.mark_error("binance_ws")
                connection_monitor.mark_disconnected("binance_ws")
                print(f"[Binance] Connection closed: {e} (attempt {consecutive_failures})")
            except Exception as e:
                consecutive_failures += 1
                connection_monitor.mark_error("binance_ws")
                print(f"[Binance] Connection error: {e} (attempt {consecutive_failures})")

            if self.running:
                # Exponential backoff with jitter
                jitter = random.uniform(0, reconnect_delay * 0.1)
                wait_time = min(reconnect_delay + jitter, max_reconnect_delay)
                print(f"[Binance] Reconnecting in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def _handle_message(self, data: dict):
        """Process incoming WebSocket message"""
        stream = data.get("stream", "")
        payload = data.get("data", data)

        if "@trade" in stream:
            await self._handle_trade(payload)
        elif "@kline" in stream:
            await self._handle_kline(payload)
        elif "@depth" in stream:
            await self._handle_depth(stream, payload)

    async def _handle_trade(self, data: dict):
        """Process trade message"""
        symbol = data["s"].lower()
        trade = Trade(
            timestamp=data["T"],
            price=float(data["p"]),
            size=float(data["q"]),
            side="BUY" if data["m"] is False else "SELL",  # m=True means buyer is maker
            exchange="binance",
        )
        self.trades[symbol].append(trade)

        if self.on_trade:
            self.on_trade(symbol, trade)

    async def _handle_kline(self, data: dict):
        """Process kline/candle message"""
        k = data["k"]
        symbol = k["s"].lower()

        candle = OHLCV(
            timestamp=k["t"],
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
        )

        # Update or append candle
        if self.candles[symbol] and self.candles[symbol][-1].timestamp == candle.timestamp:
            self.candles[symbol][-1] = candle
        else:
            self.candles[symbol].append(candle)
            # Keep last 500 candles
            if len(self.candles[symbol]) > 500:
                self.candles[symbol] = self.candles[symbol][-500:]

        if self.on_candle:
            self.on_candle(symbol, candle)

    async def _handle_depth(self, stream: str, data: dict):
        """Process order book depth message"""
        # Extract symbol from stream name
        symbol = stream.split("@")[0]

        book = OrderBook(
            symbol=symbol,
            timestamp=int(time.time() * 1000),
            bids=[OrderBookLevel(float(p), float(q)) for p, q in data.get("bids", [])],
            asks=[OrderBookLevel(float(p), float(q)) for p, q in data.get("asks", [])],
        )
        self.orderbooks[symbol] = book

        if self.on_orderbook:
            self.on_orderbook(symbol, book)

    def get_volume_delta(self, symbol: str, window_sec: int = 300) -> float:
        """Calculate cumulative volume delta (CVD) over window"""
        cutoff = time.time() * 1000 - window_sec * 1000
        trades = [t for t in self.trades.get(symbol, []) if t.timestamp > cutoff]

        buys = sum(t.size * t.price for t in trades if t.side == "BUY")
        sells = sum(t.size * t.price for t in trades if t.side == "SELL")

        return buys - sells

    def get_recent_candles(self, symbol: str, count: int = 100) -> list[dict]:
        """Get recent candles for charting"""
        candles = self.candles.get(symbol, [])[-count:]
        return [c.to_dict() for c in candles]

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT" or "btcusdt")

        Returns:
            Current price or None if not available
        """
        symbol = symbol.lower()

        # Try to get from recent trades first (most up-to-date)
        trades = self.trades.get(symbol, [])
        if trades:
            return trades[-1].price

        # Fall back to last candle close price
        candles = self.candles.get(symbol, [])
        if candles:
            return candles[-1].close

        return None

    async def stop(self):
        """Stop the feed"""
        self.running = False
        if self.ws:
            await self.ws.close()


# ============================================================================
# POLYMARKET WHALE TRACKER
# ============================================================================

class WhaleTracker:
    """
    Track whale trades on Polymarket.
    Uses REST polling with efficient caching.
    """

    def __init__(self):
        self.last_seen: dict[str, str] = {}  # wallet -> last tx hash
        self.trades: deque[WhaleTrade] = deque(maxlen=500)
        self.running = False

        # Callbacks
        self.on_whale_trade: Optional[Callable[[WhaleTrade], None]] = None

        # Whale name lookup
        self.wallet_to_name = {
            v["address"].lower(): k
            for k, v in CRYPTO_WHALE_WALLETS.items()
            if v["address"].startswith("0x") and len(v["address"]) > 10
        }

    async def start(self, poll_interval: float = 3.0):
        """Start polling for whale trades"""
        self.running = True

        async with aiohttp.ClientSession() as session:
            while self.running:
                await self._poll_all_whales(session)
                await asyncio.sleep(poll_interval)

    async def _poll_all_whales(self, session: aiohttp.ClientSession):
        """Poll all whale wallets for new trades"""
        tasks = []
        for name, info in CRYPTO_WHALE_WALLETS.items():
            addr = info["address"]
            if addr.startswith("0x") and len(addr) > 10:
                tasks.append(self._poll_wallet(session, name, addr))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_wallet(self, session: aiohttp.ClientSession, name: str, wallet: str):
        """Poll a single wallet for trades with retry logic"""
        try:
            url = f"{PolymarketAPI.TRADES}"
            params = {"user": wallet.lower(), "limit": 20}

            # Use retry wrapper for resilient HTTP requests
            resp = await retry_http_request(
                session, "GET", url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
                config=HTTP_RETRY_CONFIG,
            )

            async with resp:
                if resp.status != 200:
                    return

                trades = await resp.json()
                connection_monitor.mark_success("whale_tracker")

                for trade_data in trades:
                    tx_hash = trade_data.get("transactionHash", "")

                    # Stop if we've seen this trade
                    if self.last_seen.get(wallet) == tx_hash:
                        break

                    # Parse trade
                    whale_trade = self._parse_trade(name, wallet, trade_data)
                    if whale_trade:
                        self.trades.appendleft(whale_trade)

                        if self.on_whale_trade:
                            self.on_whale_trade(whale_trade)

                # Update last seen
                if trades:
                    self.last_seen[wallet] = trades[0].get("transactionHash", "")

        except Exception as e:
            connection_monitor.mark_error("whale_tracker")
            print(f"[WhaleTracker] Error polling {name}: {e}")

    def _parse_trade(self, name: str, wallet: str, data: dict) -> Optional[WhaleTrade]:
        """Parse raw trade data into WhaleTrade"""
        try:
            return WhaleTrade(
                whale_name=name,
                wallet=wallet,
                market_title=data.get("title", "Unknown"),
                market_slug=data.get("slug", ""),
                outcome=data.get("outcome", "Unknown"),
                side=data.get("side", "BUY"),
                size=float(data.get("size", 0)),
                price=float(data.get("price", 0)),
                timestamp=int(data.get("timestamp", 0)),
                tx_hash=data.get("transactionHash", ""),
                icon=data.get("icon", ""),
            )
        except Exception:
            return None

    def get_recent_trades(self, limit: int = 50) -> list[dict]:
        """Get recent whale trades"""
        return [t.to_dict() for t in list(self.trades)[:limit]]

    def stop(self):
        """Stop the tracker"""
        self.running = False


# ============================================================================
# MOMENTUM CALCULATOR
# ============================================================================

class MomentumCalculator:
    """
    Calculates momentum signals from multiple data sources.
    Used for 15-minute crypto market predictions.
    """

    def __init__(self, binance_feed: BinanceFeed):
        self.feed = binance_feed
        self.signals: dict[str, MomentumSignal] = {}

    def calculate(self, symbol: str) -> MomentumSignal:
        """Calculate current momentum signal for symbol"""
        now = int(time.time() * 1000)

        # Volume delta (5-min window)
        volume_delta = self.feed.get_volume_delta(symbol, window_sec=300)

        # Price change
        candles = self.feed.candles.get(symbol, [])
        current_price = candles[-1].close if candles else 0.0
        if len(candles) >= 5:
            price_change = (candles[-1].close - candles[-5].close) / candles[-5].close
        else:
            price_change = 0.0

        # Order book imbalance
        book = self.feed.orderbooks.get(symbol)
        imbalance = book.imbalance if book else 0.0

        # Calculate new technical indicators
        vwap = self.calculate_vwap(symbol)
        vwap_signal = "NEUTRAL"
        if vwap > 0 and current_price > 0:
            vwap_diff_pct = (current_price - vwap) / vwap
            if vwap_diff_pct > 0.001:  # Price > VWAP by 0.1%
                vwap_signal = "UP"
            elif vwap_diff_pct < -0.001:  # Price < VWAP by 0.1%
                vwap_signal = "DOWN"

        # Use longer periods for more stable/predictive signals
        # RSI-60 = 1 hour of 1-min candles (traditional RSI-14 is for daily bars)
        # ADX-60 = 1 hour to identify meaningful trends, not noise
        # Supertrend-30 = 30 min for stable trend bands
        rsi = self.calculate_rsi(symbol, period=60)
        adx = self.calculate_adx(symbol, period=60)
        supertrend_direction, supertrend_value = self.calculate_supertrend(symbol, period=30)

        signal = MomentumSignal(
            symbol=symbol,
            timestamp=now,
            volume_delta=volume_delta,
            price_change_pct=price_change,
            orderbook_imbalance=imbalance,
            vwap=vwap,
            vwap_signal=vwap_signal,
            rsi=rsi,
            adx=adx,
            supertrend_direction=supertrend_direction,
            supertrend_value=supertrend_value,
        )

        self.signals[symbol] = signal
        return signal

    def calculate_vwap(self, symbol: str, window_candles: int = 15) -> float:
        """Calculate Volume Weighted Average Price"""
        candles = self.feed.candles.get(symbol, [])[-window_candles:]
        if not candles:
            return 0.0

        cumulative_tpv = 0.0
        cumulative_vol = 0.0

        for c in candles:
            typical_price = (c.high + c.low + c.close) / 3
            cumulative_tpv += typical_price * c.volume
            cumulative_vol += c.volume

        return cumulative_tpv / cumulative_vol if cumulative_vol > 0 else 0.0

    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """Calculate Relative Strength Index (0-100)"""
        candles = self.feed.candles.get(symbol, [])
        if len(candles) < period + 1:
            return 50.0  # Neutral

        closes = [c.close for c in candles[-(period + 1):]]
        gains = []
        losses = []

        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_adx(self, symbol: str, period: int = 14) -> float:
        """Calculate Average Directional Index (trend strength 0-100)"""
        candles = self.feed.candles.get(symbol, [])
        if len(candles) < period + 1:
            return 0.0

        tr_list = []
        plus_dm_list = []
        minus_dm_list = []

        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            close_prev = candles[i - 1].close
            high_prev = candles[i - 1].high
            low_prev = candles[i - 1].low

            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
            tr_list.append(tr)

            plus_dm = high - high_prev if high - high_prev > low_prev - low else 0
            minus_dm = low_prev - low if low_prev - low > high - high_prev else 0
            plus_dm_list.append(max(0, plus_dm))
            minus_dm_list.append(max(0, minus_dm))

        atr = sum(tr_list[-period:]) / period if tr_list else 0
        if atr == 0:
            return 0.0

        plus_di = 100 * sum(plus_dm_list[-period:]) / period / atr
        minus_di = 100 * sum(minus_dm_list[-period:]) / period / atr

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0

        dx = abs(plus_di - minus_di) / di_sum * 100
        return dx

    def calculate_supertrend(self, symbol: str, period: int = 10, multiplier: float = 3.0) -> tuple:
        """Calculate Supertrend indicator. Returns (direction, supertrend_value)"""
        candles = self.feed.candles.get(symbol, [])
        if len(candles) < period + 1:
            return ("NEUTRAL", 0.0)

        # Calculate ATR
        tr_list = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            close_prev = candles[i - 1].close
            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
            tr_list.append(tr)

        if not tr_list:
            return ("NEUTRAL", 0.0)

        atr = sum(tr_list[-period:]) / min(period, len(tr_list))

        # Current values
        c = candles[-1]
        hl2 = (c.high + c.low) / 2

        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # Determine trend based on close vs bands
        if c.close > upper_band:
            return ("UP", lower_band)
        elif c.close < lower_band:
            return ("DOWN", upper_band)
        else:
            # Use momentum direction as tiebreaker
            prev_close = candles[-2].close if len(candles) > 1 else c.close
            if c.close > prev_close:
                return ("UP", lower_band)
            elif c.close < prev_close:
                return ("DOWN", upper_band)
            return ("NEUTRAL", hl2)

    def get_all_signals(self) -> dict[str, dict]:
        """Get all current momentum signals"""
        result = {}
        for symbol in self.feed.symbols:
            sig = self.calculate(symbol)
            result[symbol.upper()] = {
                "direction": sig.direction,
                "confidence": round(sig.confidence, 2),
                "volume_delta": round(sig.volume_delta, 0),
                "price_change_pct": round(sig.price_change_pct * 100, 3),
                "orderbook_imbalance": round(sig.orderbook_imbalance, 3),
                # New indicators
                "vwap": round(sig.vwap, 2),
                "vwap_signal": sig.vwap_signal,
                "rsi": round(sig.rsi, 1),
                "adx": round(sig.adx, 1),
                "supertrend_direction": sig.supertrend_direction,
                "supertrend_value": round(sig.supertrend_value, 2),
            }
        return result


# ============================================================================
# HISTORICAL DATA FETCHER
# ============================================================================

async def fetch_historical_candles(
    symbol: str,
    interval: str = "1m",
    limit: int = 500
) -> list[dict]:
    """Fetch historical OHLCV data from Binance REST API"""
    url = f"{CryptoExchangeAPI.BINANCE_REST}/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return []

            data = await resp.json()
            return [
                {
                    "time": int(k[0]) // 1000,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
                for k in data
            ]


# ============================================================================
# POLYMARKET 15-MIN MARKET FEED
# ============================================================================

@dataclass
class Market15Min:
    """A Polymarket 15-minute crypto up/down market"""
    condition_id: str
    token_id: str  # UP token ID (legacy field)
    question: str
    symbol: str  # BTC, ETH, SOL, etc.
    outcome: str  # "Up" or "Down"
    start_time: datetime
    end_time: datetime
    price: float  # Current price (probability)
    volume: float
    is_active: bool
    up_token_id: str = ""  # Explicit UP token ID
    down_token_id: str = ""  # DOWN token ID
    target_price: float = 0.0  # Chainlink reference price (price to beat)

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "token_id": self.token_id,
            "up_token_id": self.up_token_id or self.token_id,  # Fallback to legacy
            "down_token_id": self.down_token_id,
            "question": self.question,
            "symbol": self.symbol,
            "outcome": self.outcome,
            "start_time": int(self.start_time.timestamp()),
            "end_time": int(self.end_time.timestamp()),
            "price": self.price,
            "volume": self.volume,
            "is_active": self.is_active,
            "target_price": self.target_price,
        }


@dataclass
class MarketTrade:
    """A trade on a Polymarket 15-min market"""
    condition_id: str
    symbol: str
    outcome: str  # "Up" or "Down"
    side: str  # "BUY" or "SELL"
    size: float
    price: float
    timestamp: int
    maker: str
    taker: str

    @property
    def usd_value(self) -> float:
        return self.size * self.price

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "symbol": self.symbol,
            "outcome": self.outcome,
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "usd_value": self.usd_value,
            "timestamp": self.timestamp,
            "maker": self.maker[:10] + "..." if len(self.maker) > 10 else self.maker,
            "taker": self.taker[:10] + "..." if len(self.taker) > 10 else self.taker,
        }


def scrape_polymarket_target_price(symbol: str, market_start: int) -> float | None:
    """
    Scrape the target price (price to beat) from Polymarket for a 15-min market.

    The target price is the Chainlink reference price at market start.
    It appears on the PM page as the closePrice of the previous market window.

    Args:
        symbol: Crypto symbol (BTC, ETH, etc.)
        market_start: Unix timestamp of market start

    Returns:
        Target price in USD, or None if not found
    """
    import requests
    import re

    slug = f"{symbol.lower()}-updown-15m-{market_start}"
    url = f"https://polymarket.com/event/{slug}"

    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; CassandraBot/1.0)"
        })
        if resp.status_code != 200:
            return None

        content = resp.text

        # Look for openPrice in the page data - this is the target price ("PRICE TO BEAT")
        # Multiple openPrice values exist - page lists windows from OLDEST to NEWEST
        # The LAST openPrice is the CURRENT market's starting price
        matches = re.findall(r'"openPrice":([0-9.]+)', content)
        # Filter to only valid prices (non-empty, reasonable range for crypto)
        valid_prices = [float(m) for m in matches if m and float(m) > 0]
        if valid_prices:
            return valid_prices[-1]  # LAST openPrice = current market's target

        # Fallback: look for any price around expected BTC range
        if symbol.upper() == "BTC":
            matches = re.findall(r'([89][0-9]{4}\.[0-9]+)', content)
            if matches:
                # Return the most recent/highest one
                prices = [float(m) for m in matches]
                return max(prices)

    except Exception as e:
        print(f"[PM-Scrape] Error scraping target price for {symbol}: {e}")

    return None


class Polymarket15MinFeed:
    """
    Tracks active Polymarket 15-minute crypto markets and their trades.
    All 5 crypto markets (BTC, ETH, SOL, XRP, DOGE) operate on the same
    15-minute windows starting at :00, :15, :30, :45.

    Market slugs follow the pattern: {symbol}-updown-15m-{timestamp}
    where timestamp is the Unix epoch of the window start time.
    """

    # Crypto symbols for 15-min markets
    CRYPTO_SYMBOLS = ["btc", "eth", "sol", "xrp", "doge"]

    def __init__(self):
        self.running = False
        self.active_markets: dict[str, Market15Min] = {}  # symbol -> market
        self.market_trades: deque[MarketTrade] = deque(maxlen=500)
        self.last_trade_ids: dict[str, str] = {}  # condition_id -> last trade id
        self._last_window: int = 0  # Track last fetched window to avoid duplicate fetches
        self._target_price_cache: dict[str, float] = {}  # "SYMBOL_market_start" -> target price

        # Callbacks
        self.on_market_update: Optional[Callable[[Market15Min], None]] = None
        self.on_market_trade: Optional[Callable[[MarketTrade], None]] = None

    def _get_current_window(self) -> int:
        """Get the current 15-minute window timestamp"""
        now = int(time.time())
        return (now // 900) * 900

    async def start(self, poll_interval: float = 2.0):
        """Start polling for active markets and trades"""
        self.running = True

        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    # Fetch active 15-min markets using slug-based discovery
                    await self._fetch_active_markets(session)

                    # Fetch recent trades for each active market
                    await self._fetch_market_trades(session)

                except Exception as e:
                    print(f"[Polymarket15Min] Error: {e}")

                await asyncio.sleep(poll_interval)

    async def _fetch_active_markets(self, session: aiohttp.ClientSession):
        """
        Fetch currently active 15-minute markets using slug-based discovery.

        Market slugs follow pattern: {symbol}-updown-15m-{timestamp}
        Example: btc-updown-15m-1767460500
        """
        current_window = self._get_current_window()

        # Fetch markets for current and next window
        windows_to_fetch = [current_window, current_window + 900]

        found_markets = 0
        for symbol in self.CRYPTO_SYMBOLS:
            for window in windows_to_fetch:
                slug = f"{symbol}-updown-15m-{window}"

                try:
                    url = f"{PolymarketAPI.GAMMA_API}/markets"
                    params = {"slug": slug}
                    headers = {"User-Agent": "Mozilla/5.0"}

                    # Use retry wrapper for resilient HTTP requests
                    resp = await retry_http_request(
                        session, "GET", url,
                        params=params,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                        config=HTTP_RETRY_CONFIG,
                    )

                    async with resp:
                        if resp.status != 200:
                            continue

                        markets = await resp.json()
                        connection_monitor.mark_success("polymarket_api")

                        if markets and len(markets) > 0:
                            market = markets[0]
                            await self._parse_market(symbol.upper(), market, window)
                            found_markets += 1

                except Exception as e:
                    connection_monitor.mark_error("polymarket_api")
                    print(f"[Polymarket15Min] Error fetching {slug}: {e}")

        # Log active market count periodically
        if found_markets > 0 and len(self.active_markets) > 0:
            print(f"[Polymarket15Min] Active markets: {list(self.active_markets.keys())}")

    async def _parse_market(self, symbol: str, market: dict, window_timestamp: int):
        """
        Parse market data and update active markets.

        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            market: Market data from Gamma API
            window_timestamp: The 15-minute window start timestamp
        """
        try:
            condition_id = market.get("conditionId", "")
            question = market.get("question", "")
            volume = float(market.get("volume", 0) or 0)

            # Parse eventStartTime for accurate timing
            event_start_str = market.get("eventStartTime", "")
            if event_start_str:
                start_time = datetime.fromisoformat(event_start_str.replace("Z", "+00:00"))
                # Convert to timestamp for consistent handling
                start_ts = int(start_time.timestamp())
            else:
                # Fallback to window timestamp
                start_ts = window_timestamp
                start_time = datetime.fromtimestamp(start_ts)

            # End time is 15 minutes after start
            end_ts = start_ts + 900
            end_time = datetime.fromtimestamp(end_ts)

            # Check if market is currently active (using timestamps to avoid timezone issues)
            now_ts = int(time.time())
            is_active = start_ts <= now_ts <= end_ts

            # Get price from outcomePrices (first is "Up")
            outcome_prices = market.get("outcomePrices", "[0.5, 0.5]")
            try:
                import json as json_module
                prices = json_module.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                up_price = float(prices[0]) if prices else 0.5
            except:
                up_price = 0.5

            # Get CLOB token IDs (both Up and Down)
            clob_token_ids = market.get("clobTokenIds", "[]")
            try:
                import json as json_module
                token_ids = json_module.loads(clob_token_ids) if isinstance(clob_token_ids, str) else clob_token_ids
                up_token_id = token_ids[0] if len(token_ids) > 0 else ""
                down_token_id = token_ids[1] if len(token_ids) > 1 else ""
            except:
                up_token_id = ""
                down_token_id = ""

            # Get target price from cache or scrape from Polymarket
            cache_key = f"{symbol}_{start_ts}"
            target_price = self._target_price_cache.get(cache_key, 0.0)

            # Only scrape if we don't have a cached target price and market is active
            if target_price == 0.0 and is_active:
                # Scrape target price from Polymarket (run in thread to avoid blocking)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(scrape_polymarket_target_price, symbol, start_ts)
                    try:
                        scraped_price = future.result(timeout=5)
                        if scraped_price:
                            target_price = scraped_price
                            self._target_price_cache[cache_key] = target_price
                            print(f"[PM-Scrape] {symbol} target price: ${target_price:,.2f}")
                    except Exception as e:
                        print(f"[PM-Scrape] Failed to get target price for {symbol}: {e}")

            mkt = Market15Min(
                condition_id=condition_id,
                token_id=up_token_id,  # Legacy field
                question=question,
                symbol=symbol,
                outcome="Up",
                start_time=start_time,
                end_time=end_time,
                price=up_price,
                volume=volume,
                is_active=is_active,
                up_token_id=up_token_id,
                down_token_id=down_token_id,
                target_price=target_price,
            )

            # Only update if this is for the current window
            current_window = self._get_current_window()
            if window_timestamp == current_window:
                self.active_markets[symbol] = mkt

                if self.on_market_update:
                    self.on_market_update(mkt)

        except Exception as e:
            print(f"[Polymarket15Min] Error parsing market for {symbol}: {e}")

    async def _fetch_market_trades(self, session: aiohttp.ClientSession):
        """Fetch recent trades for active markets with retry logic"""
        for symbol, market in self.active_markets.items():
            if not market.is_active:
                continue

            try:
                # Get recent trades for this market
                url = f"{PolymarketAPI.DATA_API}/trades"
                params = {
                    "market": market.condition_id,
                    "limit": 50,
                }

                # Use retry wrapper for resilient HTTP requests
                resp = await retry_http_request(
                    session, "GET", url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                    config=HTTP_RETRY_CONFIG,
                )

                async with resp:
                    if resp.status != 200:
                        continue

                    trades = await resp.json()
                    connection_monitor.mark_success("polymarket_trades")

                    for trade_data in trades:
                        trade_id = trade_data.get("id", "") or trade_data.get("transactionHash", "")

                        # Skip if we've already seen this trade
                        if self.last_trade_ids.get(market.condition_id) == trade_id:
                            break

                        # Parse trade
                        trade = self._parse_trade(market, trade_data)
                        if trade:
                            self.market_trades.appendleft(trade)

                            if self.on_market_trade:
                                self.on_market_trade(trade)

                    # Update last seen trade
                    if trades:
                        self.last_trade_ids[market.condition_id] = trades[0].get("id", "") or trades[0].get("transactionHash", "")

            except Exception as e:
                connection_monitor.mark_error("polymarket_trades")
                print(f"[Polymarket15Min] Error fetching trades for {symbol}: {e}")

    def _parse_trade(self, market: Market15Min, data: dict) -> Optional[MarketTrade]:
        """Parse raw trade data into MarketTrade"""
        try:
            outcome = data.get("outcome", "")
            # Determine if this is an Up or Down trade
            is_up = "up" in outcome.lower() if outcome else True

            return MarketTrade(
                condition_id=market.condition_id,
                symbol=market.symbol,
                outcome="Up" if is_up else "Down",
                side=data.get("side", "BUY"),
                size=float(data.get("size", 0) or 0),
                price=float(data.get("price", 0) or 0),
                timestamp=int(data.get("timestamp", 0) or time.time()),
                maker=data.get("maker", ""),
                taker=data.get("taker", ""),
            )
        except Exception:
            return None

    def get_active_markets(self) -> dict[str, dict]:
        """Get all active 15-min markets"""
        return {sym: mkt.to_dict() for sym, mkt in self.active_markets.items()}

    def get_current_markets(self) -> list[dict]:
        """
        Get current active markets as a list.

        Returns list of dicts with symbol, up_price, down_price, etc.
        Used by whale detector and other components.
        """
        result = []
        for symbol, market in self.active_markets.items():
            result.append({
                "symbol": symbol,
                "up_price": market.price,
                "down_price": 1.0 - market.price,
                "condition_id": market.condition_id,
                "start_time": market.start_time,
                "end_time": market.end_time,
                "is_active": market.is_active,
            })
        return result

    def get_recent_trades(self, symbol: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Get recent market trades, optionally filtered by symbol"""
        trades = list(self.market_trades)
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        return [t.to_dict() for t in trades[:limit]]

    def get_next_market_time(self) -> dict:
        """Get timing info for the current and next 15-minute market window.

        Uses Eastern Time (America/New_York) to match Polymarket's market schedule.
        """
        # Use Eastern Time to match Polymarket
        ET = ZoneInfo("America/New_York")
        now = datetime.now(ET)
        minutes = now.minute

        # Current window starts at most recent :00, :15, :30, or :45
        current_slot = (minutes // 15) * 15
        current_start = now.replace(minute=current_slot, second=0, microsecond=0)
        current_end = current_start + timedelta(seconds=900)

        # Next window
        next_slot = ((minutes // 15) + 1) * 15
        next_start = now.replace(second=0, microsecond=0)

        if next_slot >= 60:
            # Use timedelta to handle hour rollover (23:xx -> 00:xx next day)
            next_start = next_start.replace(minute=0) + timedelta(hours=1)
        else:
            next_start = next_start.replace(minute=next_slot)

        time_until_next = (next_start - now).total_seconds()

        # We're inside a window if we're between current_start and current_end
        is_open = current_start <= now < current_end

        return {
            "start": int(current_start.timestamp()),
            "end": int(current_end.timestamp()),
            "time_until_start": max(0, int(time_until_next)),
            "is_open": is_open,
        }

    def stop(self):
        """Stop the feed"""
        self.running = False
