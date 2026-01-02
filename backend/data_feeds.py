"""
Real-time data feeds for crypto prices, order books, and whale tracking.
Uses WebSockets for low-latency updates.
"""

import asyncio
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from collections import deque
import aiohttp
import websockets

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
        return f"{CryptoExchangeAPI.BINANCE_WS}/{stream_str}"

    async def connect(self):
        """Connect to Binance WebSocket"""
        url = self._build_stream_url()
        self.running = True

        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    self.ws = ws
                    print(f"[Binance] Connected to {len(self.symbols)} streams")

                    async for message in ws:
                        if not self.running:
                            break
                        await self._handle_message(json.loads(message))

            except Exception as e:
                print(f"[Binance] Connection error: {e}")
                if self.running:
                    await asyncio.sleep(5)  # Reconnect delay

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
        """Poll a single wallet for trades"""
        try:
            url = f"{PolymarketAPI.TRADES}"
            params = {"user": wallet.lower(), "limit": 20}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return

                trades = await resp.json()

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
        if len(candles) >= 5:
            price_change = (candles[-1].close - candles[-5].close) / candles[-5].close
        else:
            price_change = 0.0

        # Order book imbalance
        book = self.feed.orderbooks.get(symbol)
        imbalance = book.imbalance if book else 0.0

        signal = MomentumSignal(
            symbol=symbol,
            timestamp=now,
            volume_delta=volume_delta,
            price_change_pct=price_change,
            orderbook_imbalance=imbalance,
        )

        self.signals[symbol] = signal
        return signal

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
