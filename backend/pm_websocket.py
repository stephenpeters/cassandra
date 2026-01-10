"""
Polymarket WebSocket Client for Real-Time Price Updates

Connects to wss://ws-subscriptions-clob.polymarket.com for instant price updates
instead of polling the Gamma API every 2 seconds.

Message types:
- price_change: Emitted when orders placed/cancelled
- last_trade_price: Emitted when trades execute
- book: Full orderbook snapshot
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Callable, Optional
import websockets

from retry import connection_monitor

PM_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class PMPriceUpdate:
    """Real-time price update from Polymarket WebSocket"""
    asset_id: str
    symbol: str  # BTC, ETH, etc
    outcome: str  # UP or DOWN
    price: float
    best_bid: float
    best_ask: float
    timestamp: int
    update_type: str  # "price_change", "last_trade_price", "book"

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "symbol": self.symbol,
            "outcome": self.outcome,
            "price": round(self.price, 4),
            "best_bid": round(self.best_bid, 4),
            "best_ask": round(self.best_ask, 4),
            "timestamp": self.timestamp,
            "update_type": self.update_type,
        }


class PMWebSocketClient:
    """
    Real-time Polymarket price WebSocket client.

    Subscribes to market channels for specific asset IDs (token IDs)
    and receives instant price updates.
    """

    def __init__(
        self,
        on_price_update: Optional[Callable[[PMPriceUpdate], None]] = None,
    ):
        self.on_price_update = on_price_update
        self.ws = None
        self.running = False
        self.subscribed_assets: dict[str, dict] = {}  # asset_id -> {symbol, outcome}
        self._reconnect_delay = 1
        self._last_message_time = 0
        self._message_count = 0  # For debugging first few messages
        # Price cache: "SYMBOL_OUTCOME" -> {"price": float, "timestamp": int}
        self._price_cache: dict[str, dict] = {}

    def get_price(self, symbol: str, outcome: str) -> Optional[float]:
        """Get cached real-time price for symbol/outcome. Returns None if not available."""
        key = f"{symbol.upper()}_{outcome.upper()}"
        cached = self._price_cache.get(key)
        if cached:
            # Only return if price is less than 10 seconds old
            age = time.time() - cached.get("timestamp", 0)
            if age < 10:
                return cached.get("price")
        return None

    def get_prices(self, symbol: str) -> tuple[Optional[float], Optional[float]]:
        """Get cached UP and DOWN prices for a symbol. Returns (up_price, down_price)."""
        up = self.get_price(symbol, "UP")
        down = self.get_price(symbol, "DOWN")
        return (up, down)

    async def start(self):
        """Start the WebSocket connection with auto-reconnect"""
        self.running = True

        while self.running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                connection_monitor.mark_error("pm_ws")
                connection_monitor.mark_disconnected("pm_ws")
                print(f"[PM-WS] Connection error: {e}", flush=True)

            if self.running:
                print(f"[PM-WS] Reconnecting in {self._reconnect_delay}s...", flush=True)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)

    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages"""
        print(f"[PM-WS] Connecting to {PM_WS_URL}...", flush=True)

        async with websockets.connect(
            PM_WS_URL,
            ping_interval=None,  # Disable automatic pings, we'll send PING manually
            ping_timeout=None,
        ) as ws:
            self.ws = ws
            self._reconnect_delay = 1
            connection_monitor.mark_success("pm_ws")
            print("[PM-WS] Connected!", flush=True)

            # Re-subscribe to any assets we were tracking
            if self.subscribed_assets:
                await self._send_subscribe(list(self.subscribed_assets.keys()))

            # Start keepalive task
            keepalive_task = asyncio.create_task(self._keepalive_loop())

            try:
                # Listen for messages
                async for message in ws:
                    self._last_message_time = time.time()
                    connection_monitor.mark_success("pm_ws")
                    await self._handle_message(message)
            finally:
                keepalive_task.cancel()

    async def _keepalive_loop(self):
        """Send PING keepalive every 10 seconds"""
        while True:
            try:
                await asyncio.sleep(10)
                if self.ws:
                    await self.ws.send("PING")
            except Exception:
                break

    async def _handle_message(self, message: str):
        """Parse and handle incoming WebSocket message"""
        # Handle PONG response
        if message == "PONG":
            return

        # Handle raw string numbers (heartbeats)
        if message.isdigit() or (message.startswith('-') and message[1:].isdigit()):
            return

        try:
            data = json.loads(message)

            # Ignore integer/float messages (heartbeats/sequence numbers)
            if isinstance(data, (int, float)):
                return

            # Debug first 5 useful messages to diagnose issues
            if self._message_count < 5 and isinstance(data, dict) and data.get("event_type"):
                self._message_count += 1
                print(f"[PM-WS] Sample #{self._message_count}: {str(data)[:200]}", flush=True)

            # Messages might be arrays (batch updates)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        await self._process_single_message(item)
                    # Skip non-dict items (integers, etc.)
                return

            # Single message
            if isinstance(data, dict):
                await self._process_single_message(data)

        except json.JSONDecodeError:
            # Non-JSON messages like "INVALID OPERATION"
            if message not in ("PING", "PONG") and len(message) > 2:
                print(f"[PM-WS] Non-JSON message: {message[:50]}", flush=True)
        except Exception as e:
            import traceback
            print(f"[PM-WS] Error handling message: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    async def _process_single_message(self, data: dict):
        """Process a single message object"""
        # PM sends price_changes array in each message
        if "price_changes" in data:
            for pc in data.get("price_changes", []):
                await self._handle_price_change(pc)
            return

        # Handle different message types
        msg_type = data.get("event_type") or data.get("type")

        if msg_type == "price_change":
            await self._handle_price_change(data)
        elif msg_type == "last_trade_price":
            await self._handle_last_trade(data)
        elif msg_type == "book":
            await self._handle_book(data)
        elif msg_type in ("subscribed", "unsubscribed"):
            print(f"[PM-WS] {msg_type}: {data.get('assets_ids', [])}", flush=True)
        # Don't log unknown types to reduce noise

    async def _handle_price_change(self, data: dict):
        """Handle price_change message"""
        asset_id = data.get("asset_id", "")

        if asset_id not in self.subscribed_assets:
            # Debug: log first few misses
            if self._message_count < 10:
                print(f"[PM-WS] Asset not subscribed: {asset_id[:20]}... (have {len(self.subscribed_assets)} assets)", flush=True)
            return

        info = self.subscribed_assets[asset_id]

        # Extract best bid/ask - these are the actual market prices
        best_bid = float(data.get("best_bid", 0))
        best_ask = float(data.get("best_ask", 0))

        # Calculate midpoint as the fair price (matches Polymarket UI)
        # Note: data.get("price") is the order price level, NOT the market price
        if best_bid > 0 and best_ask > 0:
            midpoint = (best_bid + best_ask) / 2
        elif best_bid > 0:
            midpoint = best_bid
        elif best_ask > 0:
            midpoint = best_ask
        else:
            # Fallback to order price if no bid/ask
            midpoint = float(data.get("price", 0))

        # Debug: log first matched price update
        if self._message_count < 8:
            print(f"[PM-WS] Price update: {info.get('symbol')} {info.get('outcome')} bid={best_bid:.4f} ask={best_ask:.4f} mid={midpoint:.4f}", flush=True)

        update = PMPriceUpdate(
            asset_id=asset_id,
            symbol=info.get("symbol", ""),
            outcome=info.get("outcome", ""),
            price=midpoint,  # Use midpoint as the displayed price
            best_bid=best_bid,
            best_ask=best_ask,
            timestamp=data.get("timestamp", int(time.time() * 1000)),
            update_type="price_change",
        )

        # Cache the midpoint price for synchronous access
        cache_key = f"{update.symbol}_{update.outcome}"
        self._price_cache[cache_key] = {
            "price": midpoint,
            "timestamp": time.time(),
        }

        if self.on_price_update:
            self.on_price_update(update)

    async def _handle_last_trade(self, data: dict):
        """Handle last_trade_price message"""
        asset_id = data.get("asset_id", "")

        if asset_id not in self.subscribed_assets:
            return

        info = self.subscribed_assets[asset_id]

        update = PMPriceUpdate(
            asset_id=asset_id,
            symbol=info.get("symbol", ""),
            outcome=info.get("outcome", ""),
            price=float(data.get("price", 0)),
            best_bid=0,
            best_ask=0,
            timestamp=data.get("timestamp", int(time.time() * 1000)),
            update_type="last_trade_price",
        )

        # Cache the price for synchronous access
        cache_key = f"{update.symbol}_{update.outcome}"
        self._price_cache[cache_key] = {
            "price": update.price,
            "timestamp": time.time(),
        }

        if self.on_price_update:
            self.on_price_update(update)

    async def _handle_book(self, data: dict):
        """Handle book (orderbook snapshot) message"""
        asset_id = data.get("asset_id", "")

        if asset_id not in self.subscribed_assets:
            return

        info = self.subscribed_assets[asset_id]

        # Extract best bid/ask from orderbook
        # Handle both formats: [{"price": x, "size": y}] and [[price, size]]
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        def get_price(orders: list) -> float:
            if not orders:
                return 0
            first = orders[0]
            if isinstance(first, dict):
                return float(first.get("price", 0))
            elif isinstance(first, (list, tuple)) and len(first) > 0:
                return float(first[0])
            return 0

        best_bid = get_price(bids)
        best_ask = get_price(asks)
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

        # Calculate spread - wide spreads indicate thin/unreliable orderbooks
        spread = best_ask - best_bid if best_bid and best_ask else 1.0

        # Debug: log first book snapshots
        if self._message_count < 8:
            print(f"[PM-WS] Book snapshot: {info.get('symbol')} {info.get('outcome')} bid={best_bid:.4f} ask={best_ask:.4f} mid={mid_price:.4f} spread={spread:.4f}", flush=True)

        update = PMPriceUpdate(
            asset_id=asset_id,
            symbol=info.get("symbol", ""),
            outcome=info.get("outcome", ""),
            price=mid_price,
            best_bid=best_bid,
            best_ask=best_ask,
            timestamp=data.get("timestamp", int(time.time() * 1000)),
            update_type="book",
        )

        # Only cache book snapshot prices if the spread is reasonable (<50 cents)
        # Wide spreads (like 0.01/0.99) indicate thin orderbooks and shouldn't
        # overwrite more accurate price_change updates
        cache_key = f"{update.symbol}_{update.outcome}"
        if spread < 0.50:
            self._price_cache[cache_key] = {
                "price": mid_price,
                "timestamp": time.time(),
            }

        if self.on_price_update:
            self.on_price_update(update)

    async def _send_subscribe(self, asset_ids: list[str]):
        """Send subscription message for asset IDs"""
        if not self.ws:
            return

        # Polymarket WS subscription format
        msg = {
            "assets_ids": asset_ids,
            "type": "market",
        }

        await self.ws.send(json.dumps(msg))
        print(f"[PM-WS] Subscribed to {len(asset_ids)} assets: {[aid[:20]+'...' for aid in asset_ids]}", flush=True)

    async def subscribe(self, asset_id: str, symbol: str, outcome: str):
        """Subscribe to price updates for an asset"""
        self.subscribed_assets[asset_id] = {
            "symbol": symbol,
            "outcome": outcome,
        }

        if self.ws:
            await self._send_subscribe([asset_id])

    async def subscribe_market(self, symbol: str, up_token_id: str, down_token_id: str):
        """Subscribe to both UP and DOWN tokens for a market"""
        if up_token_id:
            self.subscribed_assets[up_token_id] = {"symbol": symbol, "outcome": "UP"}
        if down_token_id:
            self.subscribed_assets[down_token_id] = {"symbol": symbol, "outcome": "DOWN"}

        asset_ids = [aid for aid in [up_token_id, down_token_id] if aid]
        if self.ws and asset_ids:
            await self._send_subscribe(asset_ids)

    async def subscribe_markets_batch(self, markets: dict[str, tuple[str, str]]):
        """
        Subscribe to multiple markets in a single batch message.
        markets: dict of symbol -> (up_token_id, down_token_id)

        This is needed because Polymarket WS may replace previous subscriptions
        when a new subscription message is sent.
        """
        # Clear old subscriptions first
        self.subscribed_assets.clear()

        all_asset_ids = []
        for symbol, (up_token_id, down_token_id) in markets.items():
            if up_token_id:
                self.subscribed_assets[up_token_id] = {"symbol": symbol, "outcome": "UP"}
                all_asset_ids.append(up_token_id)
            if down_token_id:
                self.subscribed_assets[down_token_id] = {"symbol": symbol, "outcome": "DOWN"}
                all_asset_ids.append(down_token_id)

        if self.ws and all_asset_ids:
            print(f"[PM-WS] Batch subscribing to {len(all_asset_ids)} assets for {len(markets)} markets", flush=True)
            await self._send_subscribe(all_asset_ids)

    async def unsubscribe(self, asset_id: str):
        """Unsubscribe from an asset"""
        if asset_id in self.subscribed_assets:
            del self.subscribed_assets[asset_id]

        if self.ws:
            msg = {
                "assets_ids": [asset_id],
                "type": "market",
                "operation": "unsubscribe",
            }
            await self.ws.send(json.dumps(msg))

    def stop(self):
        """Stop the WebSocket client"""
        self.running = False
        if self.ws:
            asyncio.create_task(self.ws.close())

    def get_status(self) -> dict:
        """Get current connection status"""
        return {
            "connected": self.ws is not None and self.running,
            "subscribed_assets": len(self.subscribed_assets),
            "last_message_age_sec": int(time.time() - self._last_message_time) if self._last_message_time else None,
        }


# Singleton instance
_pm_ws_client: Optional[PMWebSocketClient] = None


def get_pm_ws_client() -> PMWebSocketClient:
    """Get or create the PM WebSocket client singleton"""
    global _pm_ws_client
    if _pm_ws_client is None:
        _pm_ws_client = PMWebSocketClient()
    return _pm_ws_client
