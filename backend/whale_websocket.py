"""
WebSocket-based Whale Trade Detection

Connects to Polymarket CLOB WebSocket for real-time trade detection.
Reduces latency from ~39s (polling) to ~5-15s (WebSocket).

Target whale: Account88888 (69% win rate on BTC 15-min markets)
"""

import asyncio
import aiohttp
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable
from collections import defaultdict

from retry import connection_monitor

logger = logging.getLogger(__name__)

# Profitable whale wallets (from Polymarket profiles)
# Account88888: +$446K lifetime, 60% win rate on BTC 15-min
# gabagool22: +$529K lifetime, 62% win rate on BTC 15-min
# Note: updateupdate excluded - only 18% accuracy on 15-min markets
ACCOUNT88888_WALLET = "0x7f69983eb28245bba0d5083502a78744a8f66162"
GABAGOOL22_WALLET = "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d"

# Polymarket WebSocket endpoint
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class WhaleTradeEvent:
    """Real-time whale trade detected via WebSocket"""
    wallet: str
    whale_name: str
    symbol: str
    side: str  # BUY or SELL
    outcome: str  # Up or Down
    size: float
    price: float
    usd_value: float
    timestamp: int
    market_slug: str
    detection_latency_ms: int  # How fast we detected it

    def to_dict(self) -> dict:
        return {
            "wallet": self.wallet,
            "whale_name": self.whale_name,
            "symbol": self.symbol,
            "side": self.side,
            "outcome": self.outcome,
            "size": self.size,
            "price": self.price,
            "usd_value": self.usd_value,
            "timestamp": self.timestamp,
            "market_slug": self.market_slug,
            "detection_latency_ms": self.detection_latency_ms,
        }


@dataclass
class WhaleBiasUpdate:
    """Aggregated bias from whale's trades in current market"""
    whale_name: str
    symbol: str
    bias: str  # UP, DOWN, or NEUTRAL
    net_up: float
    net_down: float
    confidence: float  # 0-1 based on trade volume
    num_trades: int
    total_volume_usd: float
    first_trade_elapsed: int  # Seconds after market open
    last_trade_elapsed: int

    def to_dict(self) -> dict:
        return {
            "whale_name": self.whale_name,
            "symbol": self.symbol,
            "bias": self.bias,
            "net_up": self.net_up,
            "net_down": self.net_down,
            "confidence": self.confidence,
            "num_trades": self.num_trades,
            "total_volume_usd": self.total_volume_usd,
            "first_trade_elapsed": self.first_trade_elapsed,
            "last_trade_elapsed": self.last_trade_elapsed,
        }


class WhaleWebSocketDetector:
    """
    Real-time whale detection using Polymarket WebSocket.

    Subscribes to trade events for crypto 15-min markets and
    filters for tracked whale wallets (Account88888).
    """

    def __init__(
        self,
        on_whale_trade: Optional[Callable[[WhaleTradeEvent], None]] = None,
        on_bias_update: Optional[Callable[[WhaleBiasUpdate], None]] = None,
    ):
        self.on_whale_trade = on_whale_trade
        self.on_bias_update = on_bias_update

        # Track whale positions per market window
        # Key: "SYMBOL_market_start"
        self._positions: dict[str, dict] = defaultdict(lambda: {
            "net_up": 0.0,
            "net_down": 0.0,
            "trades": [],
            "first_trade_time": None,
            "last_trade_time": None,
        })

        # Tracked wallets (only profitable whales)
        self._tracked_wallets = {
            ACCOUNT88888_WALLET.lower(): "Account88888",
            GABAGOOL22_WALLET.lower(): "gabagool22",
        }

        # WebSocket connection
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._running = False
        self._reconnect_delay = 1.0

        # Current market windows (set externally)
        self._active_markets: dict[str, dict] = {}  # symbol -> {start_time, end_time, up_token, down_token}

    def set_active_markets(self, markets: dict[str, dict]):
        """Update the active market windows to track"""
        self._active_markets = markets
        logger.info(f"[WhaleWS] Tracking markets: {list(markets.keys())}")

    def get_token_to_symbol(self) -> dict[str, tuple[str, str]]:
        """Map token IDs to (symbol, outcome)"""
        mapping = {}
        for symbol, market in self._active_markets.items():
            if "up_token_id" in market:
                mapping[market["up_token_id"]] = (symbol, "Up")
            if "down_token_id" in market:
                mapping[market["down_token_id"]] = (symbol, "Down")
        return mapping

    async def start(self):
        """Start the WebSocket connection"""
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                connection_monitor.mark_error("whale_ws")
                connection_monitor.mark_disconnected("whale_ws")
                logger.error(f"[WhaleWS] Connection error: {e}")
                if self._running:
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def stop(self):
        """Stop the WebSocket connection"""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _connect_and_listen(self):
        """Connect to WebSocket and process messages"""
        async with aiohttp.ClientSession() as session:
            logger.info(f"[WhaleWS] Connecting to {WS_URL}")

            async with session.ws_connect(WS_URL, heartbeat=30) as ws:
                self._ws = ws
                self._reconnect_delay = 1.0  # Reset on successful connect
                connection_monitor.mark_success("whale_ws")
                logger.info("[WhaleWS] Connected successfully")

                # Subscribe to markets
                await self._subscribe_to_markets(ws)

                # Process messages
                async for msg in ws:
                    connection_monitor.mark_success("whale_ws")
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._process_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        connection_monitor.mark_error("whale_ws")
                        logger.error(f"[WhaleWS] Error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        connection_monitor.mark_disconnected("whale_ws")
                        logger.info("[WhaleWS] Connection closed")
                        break

    async def _subscribe_to_markets(self, ws: aiohttp.ClientWebSocketResponse):
        """Subscribe to trade events for active markets"""
        token_mapping = self.get_token_to_symbol()

        if not token_mapping:
            logger.warning("[WhaleWS] No active markets to subscribe to")
            return

        # Subscribe to each token
        for token_id in token_mapping.keys():
            subscribe_msg = {
                "type": "subscribe",
                "channel": "market",
                "market": token_id,
            }
            await ws.send_str(json.dumps(subscribe_msg))
            logger.debug(f"[WhaleWS] Subscribed to {token_id}")

        logger.info(f"[WhaleWS] Subscribed to {len(token_mapping)} tokens")

    async def _process_message(self, data: str):
        """Process incoming WebSocket message"""
        try:
            msg = json.loads(data)

            # Handle trade events
            if msg.get("event_type") == "trade" or msg.get("type") == "trade":
                await self._handle_trade(msg)

        except json.JSONDecodeError:
            logger.warning(f"[WhaleWS] Invalid JSON: {data[:100]}")
        except Exception as e:
            logger.error(f"[WhaleWS] Error processing message: {e}")

    async def _handle_trade(self, msg: dict):
        """Handle a trade event"""
        trade_data = msg.get("data", msg)

        # Get maker/taker wallet
        maker = trade_data.get("maker", "").lower()
        taker = trade_data.get("taker", "").lower()

        # Check if this is a tracked whale
        whale_name = None
        whale_wallet = None

        if maker in self._tracked_wallets:
            whale_name = self._tracked_wallets[maker]
            whale_wallet = maker
        elif taker in self._tracked_wallets:
            whale_name = self._tracked_wallets[taker]
            whale_wallet = taker

        if not whale_name:
            return  # Not a whale trade

        # Map token to symbol/outcome
        token_id = trade_data.get("asset_id", trade_data.get("token_id", ""))
        token_mapping = self.get_token_to_symbol()

        if token_id not in token_mapping:
            return  # Not a tracked market

        symbol, outcome = token_mapping[token_id]
        market = self._active_markets.get(symbol, {})
        market_start = market.get("start_time", 0)

        # Parse trade details
        side = trade_data.get("side", "BUY").upper()
        size = float(trade_data.get("size", 0))
        price = float(trade_data.get("price", 0))
        usd_value = size * price
        timestamp = int(trade_data.get("timestamp", datetime.now(timezone.utc).timestamp()))

        # Calculate detection latency
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        trade_ts_ms = timestamp * 1000 if timestamp < 1e12 else timestamp
        latency_ms = now - trade_ts_ms

        # Create trade event
        event = WhaleTradeEvent(
            wallet=whale_wallet,
            whale_name=whale_name,
            symbol=symbol,
            side=side,
            outcome=outcome,
            size=size,
            price=price,
            usd_value=usd_value,
            timestamp=timestamp,
            market_slug=f"{symbol.lower()}-updown-15m-{market_start}",
            detection_latency_ms=latency_ms,
        )

        logger.info(
            f"[WhaleWS] {whale_name} {side} {outcome} on {symbol}: "
            f"${usd_value:.2f} (latency: {latency_ms}ms)"
        )

        # Emit trade event
        if self.on_whale_trade:
            self.on_whale_trade(event)

        # Update position tracking
        await self._update_position(event, market_start)

    async def _update_position(self, event: WhaleTradeEvent, market_start: int):
        """Update whale's position and emit bias update"""
        key = f"{event.symbol}_{market_start}"
        pos = self._positions[key]

        # Update net position
        delta = event.size if event.side == "BUY" else -event.size
        if event.outcome == "Up":
            pos["net_up"] += delta
        else:
            pos["net_down"] += delta

        # Track trades
        pos["trades"].append(event)
        if pos["first_trade_time"] is None:
            pos["first_trade_time"] = event.timestamp
        pos["last_trade_time"] = event.timestamp

        # Calculate bias
        net_up = pos["net_up"]
        net_down = pos["net_down"]

        if abs(net_up - net_down) < 10:  # Minimum threshold
            bias = "NEUTRAL"
        elif net_up > net_down:
            bias = "UP"
        else:
            bias = "DOWN"

        total_volume = sum(t.usd_value for t in pos["trades"])
        confidence = min(1.0, total_volume / 1000)  # Max confidence at $1000

        # Calculate elapsed times
        first_elapsed = pos["first_trade_time"] - market_start if pos["first_trade_time"] else 0
        last_elapsed = pos["last_trade_time"] - market_start if pos["last_trade_time"] else 0

        # Emit bias update
        bias_update = WhaleBiasUpdate(
            whale_name=event.whale_name,
            symbol=event.symbol,
            bias=bias,
            net_up=net_up,
            net_down=net_down,
            confidence=confidence,
            num_trades=len(pos["trades"]),
            total_volume_usd=total_volume,
            first_trade_elapsed=first_elapsed,
            last_trade_elapsed=last_elapsed,
        )

        if self.on_bias_update:
            self.on_bias_update(bias_update)

    def get_current_bias(self, symbol: str, market_start: int) -> Optional[WhaleBiasUpdate]:
        """Get current whale bias for a market"""
        key = f"{symbol}_{market_start}"
        if key not in self._positions:
            return None

        pos = self._positions[key]
        if not pos["trades"]:
            return None

        net_up = pos["net_up"]
        net_down = pos["net_down"]

        if abs(net_up - net_down) < 10:
            bias = "NEUTRAL"
        elif net_up > net_down:
            bias = "UP"
        else:
            bias = "DOWN"

        total_volume = sum(t.usd_value for t in pos["trades"])

        return WhaleBiasUpdate(
            whale_name="Account88888",
            symbol=symbol,
            bias=bias,
            net_up=net_up,
            net_down=net_down,
            confidence=min(1.0, total_volume / 1000),
            num_trades=len(pos["trades"]),
            total_volume_usd=total_volume,
            first_trade_elapsed=pos["first_trade_time"] - market_start if pos["first_trade_time"] else 0,
            last_trade_elapsed=pos["last_trade_time"] - market_start if pos["last_trade_time"] else 0,
        )

    def clear_market(self, symbol: str, market_start: int):
        """Clear position data for a completed market"""
        key = f"{symbol}_{market_start}"
        if key in self._positions:
            del self._positions[key]


# Singleton instance
_whale_ws_detector: Optional[WhaleWebSocketDetector] = None


def get_whale_ws_detector() -> WhaleWebSocketDetector:
    """Get or create the whale WebSocket detector singleton"""
    global _whale_ws_detector
    if _whale_ws_detector is None:
        _whale_ws_detector = WhaleWebSocketDetector()
    return _whale_ws_detector


async def start_whale_websocket(
    on_whale_trade: Optional[Callable[[WhaleTradeEvent], None]] = None,
    on_bias_update: Optional[Callable[[WhaleBiasUpdate], None]] = None,
):
    """Start the whale WebSocket detector"""
    detector = get_whale_ws_detector()
    detector.on_whale_trade = on_whale_trade
    detector.on_bias_update = on_bias_update
    await detector.start()
