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

        try:
            data = json.loads(message)

            # Ignore integer messages (heartbeats/sequence numbers)
            if isinstance(data, int):
                return

            # Debug: Disabled to reduce memory allocation from string formatting
            # if self._message_count < 5:
            #     self._message_count += 1
            #     print(f"[PM-WS] Sample message #{self._message_count}: {str(data)[:200]}", flush=True)

            # Messages might be arrays (batch updates)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        await self._process_single_message(item)
                return

            # Single message
            if isinstance(data, dict):
                await self._process_single_message(data)

        except json.JSONDecodeError:
            # Non-JSON messages like "INVALID OPERATION"
            if message not in ("PING", "PONG") and len(message) > 2:
                print(f"[PM-WS] Non-JSON message: {message[:50]}", flush=True)
        except Exception as e:
            print(f"[PM-WS] Error handling message: {e}", flush=True)

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
            return

        info = self.subscribed_assets[asset_id]

        update = PMPriceUpdate(
            asset_id=asset_id,
            symbol=info.get("symbol", ""),
            outcome=info.get("outcome", ""),
            price=float(data.get("price", 0)),
            best_bid=float(data.get("best_bid", 0)),
            best_ask=float(data.get("best_ask", 0)),
            timestamp=data.get("timestamp", int(time.time() * 1000)),
            update_type="price_change",
        )

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

        if self.on_price_update:
            self.on_price_update(update)

    async def _handle_book(self, data: dict):
        """Handle book (orderbook snapshot) message"""
        asset_id = data.get("asset_id", "")

        if asset_id not in self.subscribed_assets:
            return

        info = self.subscribed_assets[asset_id]

        # Extract best bid/ask from orderbook
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

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

        if self.on_price_update:
            self.on_price_update(update)

    async def _send_subscribe(self, asset_ids: list[str]):
        """Send subscription message for asset IDs"""
        if not self.ws:
            return

        msg = {
            "assets_ids": asset_ids,
            "type": "market",
        }

        await self.ws.send(json.dumps(msg))
        print(f"[PM-WS] Subscribed to {len(asset_ids)} assets", flush=True)

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
