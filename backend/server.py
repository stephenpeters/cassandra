#!/usr/bin/env python3
"""
WebSocket server for the Polymarket Whale Tracker.
Provides real-time data feeds to the frontend.
"""

import asyncio
import json
import signal
import time
from datetime import datetime
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import secrets

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the same directory as server.py
_backend_dir = Path(__file__).parent
load_dotenv(_backend_dir / ".env")

# =============================================================================
# SECURITY: API Key Authentication
# =============================================================================

# Environment: "production" requires API_KEY, "development" allows auto-generation
ENV = os.getenv("ENV", "development").lower()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    if ENV == "production":
        # In production, require API_KEY to be explicitly set
        print("[Security] FATAL: API_KEY environment variable not set.")
        print("[Security] Set API_KEY in .env or environment before starting in production.")
        import sys
        sys.exit(1)
    else:
        # In development, generate a random key for convenience
        API_KEY = secrets.token_urlsafe(32)
        print(f"[Security] No API_KEY set. Generated temporary key: {API_KEY}")
        print("[Security] Set API_KEY in .env for persistent authentication")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key for protected endpoints"""
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Include X-API-Key header."
        )
    return api_key


# =============================================================================
# PYDANTIC MODELS FOR INPUT VALIDATION
# =============================================================================

class TradingModeRequest(BaseModel):
    mode: str = Field(..., pattern="^(paper|live)$")


class KillSwitchRequest(BaseModel):
    activate: bool
    reason: str = "Manual"


class EnabledAssetsRequest(BaseModel):
    assets: List[str] = Field(..., min_length=1, max_length=5)


class ConfigUpdateRequest(BaseModel):
    max_position_usd: Optional[float] = Field(None, ge=10, le=10000)
    max_daily_volume_usd: Optional[float] = Field(None, ge=100, le=100000)
    max_consecutive_losses: Optional[int] = Field(None, ge=1, le=20)
    daily_loss_limit_usd: Optional[float] = Field(None, ge=50, le=10000)
    max_slippage_pct: Optional[float] = Field(None, ge=0.1, le=10)
    require_manual_confirm: Optional[bool] = None
    min_signal_confidence: Optional[float] = Field(None, ge=0.5, le=1.0)


class OrderConfirmRequest(BaseModel):
    order_id: str


class OrderRejectRequest(BaseModel):
    order_id: str
    reason: str = "Manual rejection"


class TestOrderRequest(BaseModel):
    """Manual test order - for testing live trading"""
    symbol: str = Field(..., pattern="^(BTC|ETH|SOL|XRP|DOGE)$")
    side: str = Field(..., pattern="^(UP|DOWN)$")
    amount_usd: float = Field(..., ge=1.0, le=500.0)  # $1-$500 for manual orders


from config import (
    CryptoExchangeAPI,
    CRYPTO_WHALE_WALLETS,
    DEFAULT_CONFIG,
)
from data_feeds import (
    BinanceFeed,
    WhaleTracker,
    MomentumCalculator,
    Polymarket15MinFeed,
    fetch_historical_candles,
    Trade,
    OHLCV,
    OrderBook,
    WhaleTrade,
)
from trading import TradingEngine
from live_trading import (
    LiveTradingEngine,
    LiveTradingConfig,
    TradingMode,
    CLOB_AVAILABLE,
)
from trade_ledger import TradeLedger
from market_data_store import MarketDataStore, PriceSnapshot, MarketTradeRecord
from whale_following import (
    WhaleFollowingStrategy,
    WhaleFollowConfig,
    WhaleTradeDetector,
    WhaleMarketBias,
    backtest_whale_following,
    fetch_whale_market_history,
    fetch_whale_positions_pnl,
)
from whale_websocket import (
    WhaleWebSocketDetector,
    WhaleTradeEvent,
    WhaleBiasUpdate,
    get_whale_ws_detector,
)
from pm_websocket import (
    PMWebSocketClient,
    PMPriceUpdate,
    get_pm_ws_client,
)
from retry import connection_monitor
from time_sync import (
    start_sync_loop,
    get_synced_timestamp,
    get_sync_status,
    sync_with_polymarket,
    sync_with_ntp,
)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Polymarket Whale Tracker API",
    description="Real-time crypto prices, whale trades, and momentum signals",
    version="1.0.0",
)

# CORS - Restrict to known origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Data feeds
binance_feed: BinanceFeed = None
whale_tracker: WhaleTracker = None
momentum_calc: MomentumCalculator = None
polymarket_feed: Polymarket15MinFeed = None
paper_trading: TradingEngine = None
live_trading: LiveTradingEngine = None
trade_ledger: TradeLedger = None
whale_detector: WhaleTradeDetector = None
whale_ws_detector: WhaleWebSocketDetector = None  # WebSocket-based whale detection (~5-15s latency)
pm_ws_client: PMWebSocketClient = None  # Real-time Polymarket prices (replaces 2s polling)

# WebSocket clients
ws_clients: Set[WebSocket] = set()

# Background tasks
background_tasks: list[asyncio.Task] = []


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

async def prefill_historical_candles():
    """
    Fetch historical candles from Binance REST API to pre-fill the candle buffer.
    This allows indicators (RSI, ADX, Supertrend) to calculate immediately.
    """
    if not binance_feed:
        return

    print("[Server] Prefilling historical candles for indicators...")

    for symbol in binance_feed.symbols:
        try:
            # Fetch last 100 1-minute candles (enough for RSI-14, ADX-14, etc.)
            candles = await fetch_historical_candles(symbol.upper(), interval="1m", limit=100)

            if candles:
                # Convert to OHLCV objects and store
                ohlcv_list = []
                for c in candles:
                    ohlcv = OHLCV(
                        timestamp=c["time"],
                        open=c["open"],
                        high=c["high"],
                        low=c["low"],
                        close=c["close"],
                        volume=c["volume"],
                    )
                    ohlcv_list.append(ohlcv)

                binance_feed.candles[symbol] = ohlcv_list
                print(f"[Server] Loaded {len(ohlcv_list)} candles for {symbol.upper()}")

        except Exception as e:
            print(f"[Server] Error loading candles for {symbol}: {e}")

    print("[Server] Historical candles loaded")


@app.on_event("startup")
async def startup():
    """Initialize data feeds on startup"""
    global binance_feed, whale_tracker, momentum_calc, polymarket_feed, paper_trading, trade_ledger, live_trading, whale_detector, whale_ws_detector, pm_ws_client

    # Initialize Binance feed with crypto symbols
    symbols = [v["binance"] for v in CryptoExchangeAPI.SYMBOLS.values()]
    binance_feed = BinanceFeed(symbols)

    # Set up callbacks to broadcast to WebSocket clients
    binance_feed.on_trade = lambda sym, trade: asyncio.create_task(
        broadcast({"type": "trade", "symbol": sym.upper(), "data": {
            "time": trade.timestamp // 1000,
            "price": trade.price,
            "size": trade.size,
            "side": trade.side,
        }})
    )

    binance_feed.on_candle = lambda sym, candle: asyncio.create_task(
        broadcast({"type": "candle", "symbol": sym.upper(), "data": candle.to_dict()})
    )

    binance_feed.on_orderbook = lambda sym, book: asyncio.create_task(
        broadcast({"type": "orderbook", "symbol": sym.upper(), "data": {
            "mid": book.mid_price,
            "spread": book.spread,
            "imbalance": book.imbalance,
            "bids": [[l.price, l.size] for l in book.bids[:5]],
            "asks": [[l.price, l.size] for l in book.asks[:5]],
        }})
    )

    # Initialize whale tracker
    whale_tracker = WhaleTracker()
    whale_tracker.on_whale_trade = lambda trade: asyncio.create_task(
        broadcast({"type": "whale_trade", "data": trade.to_dict()})
    )

    # Initialize Polymarket 15-min feed
    polymarket_feed = Polymarket15MinFeed()
    polymarket_feed.on_market_update = lambda mkt: asyncio.create_task(
        broadcast({"type": "market_update", "data": mkt.to_dict()})
    )
    polymarket_feed.on_market_trade = lambda trade: asyncio.create_task(
        broadcast({"type": "market_trade", "data": trade.to_dict()})
    )

    # Initialize momentum calculator
    momentum_calc = MomentumCalculator(binance_feed)

    # Initialize trade ledger for persistent storage
    trade_ledger = TradeLedger(db_path="trades.db")
    print(f"[Server] Trade ledger initialized: trades.db")

    # Initialize market data store for rolling 24-hour data
    market_data_store = MarketDataStore(db_path="market_data.db")
    print(f"[Server] Market data store initialized: market_data.db")

    # Initialize live trading engine FIRST
    # This creates its internal paper_engine which we'll use as the shared instance
    # IMPORTANT: This eliminates the dual-engine state drift bug
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    live_config = LiveTradingConfig.from_env()
    live_trading = LiveTradingEngine(
        private_key=private_key,
        config=live_config,
        data_dir=".",
        ledger=trade_ledger,
    )

    # Use live_trading's internal paper_engine as the shared paper trading instance
    # This ensures signals and positions are synchronized with live order execution
    paper_trading = live_trading.paper_engine

    # Set up paper trading callbacks for broadcasting
    paper_trading.on_signal = lambda sig: asyncio.create_task(
        broadcast({"type": "paper_signal", "data": sig.to_dict()})
    )
    paper_trading.on_trade = lambda trade: asyncio.create_task(
        broadcast({"type": "paper_trade", "data": trade.to_dict()})
    )
    paper_trading.on_position_open = lambda pos: asyncio.create_task(
        broadcast({"type": "paper_position", "data": pos.to_dict()})
    )
    paper_trading.on_alert = lambda title, msg: asyncio.create_task(
        live_trading._send_alert(title, msg)
    )

    # Set up live trading callbacks
    live_trading.on_order = lambda order: asyncio.create_task(
        broadcast({"type": "live_order", "data": order.to_dict()})
    )
    live_trading.on_fill = lambda order: asyncio.create_task(
        broadcast({"type": "live_fill", "data": order.to_dict()})
    )
    live_trading.on_alert = lambda title, msg: asyncio.create_task(
        broadcast({"type": "live_alert", "data": {"title": title, "message": msg}})
    )

    # Trading mode is controlled by live_trading.config.mode
    print(f"[Server] Live trading initialized in {live_trading.config.mode.value} mode")

    # Initialize whale trade detector
    global whale_detector

    def on_whale_bias_detected(bias: WhaleMarketBias):
        """Handle whale bias detection - forward to paper trading"""
        if paper_trading and bias.bias != "NEUTRAL":
            # Get current Polymarket price
            current_price = 0.5  # Default
            if polymarket_feed:
                markets = polymarket_feed.get_current_markets()
                for m in markets:
                    if m.get("symbol") == bias.symbol:
                        current_price = m.get("up_price", 0.5)
                        break

            paper_trading.process_whale_bias(
                whale_name=bias.whale_name,
                symbol=bias.symbol,
                bias=bias.bias,
                bias_confidence=bias.bias_confidence,
                market_start=bias.market_start,
                market_end=bias.market_end,
                detection_latency_sec=bias.detection_latency_sec,
                polymarket_price=current_price,
            )

        # Broadcast to frontend
        asyncio.create_task(
            broadcast({"type": "whale_bias", "data": bias.to_dict()})
        )

    whale_detector = WhaleTradeDetector(
        whales=["gabagool22"],  # Track gabagool22 by default
        poll_interval_sec=1.0,  # Poll every second
        on_bias_detected=on_whale_bias_detected,
    )
    print("[Server] Whale trade detector initialized")

    # Initialize WebSocket-based whale detector for Account88888
    # This has ~5-15s latency vs ~39s for polling
    global whale_ws_detector

    def on_ws_whale_trade(event: WhaleTradeEvent):
        """Handle real-time whale trade from WebSocket"""
        print(f"[WhaleWS] {event.whale_name} {event.side} {event.outcome} on {event.symbol}: "
              f"${event.usd_value:.2f} (latency: {event.detection_latency_ms}ms)")

        # Broadcast to frontend
        asyncio.create_task(
            broadcast({"type": "whale_ws_trade", "data": event.to_dict()})
        )

    def on_ws_bias_update(bias: WhaleBiasUpdate):
        """Handle bias update from WebSocket whale detector"""
        print(f"[WhaleWS] {bias.whale_name} bias on {bias.symbol}: {bias.bias} "
              f"(confidence: {bias.confidence:.0%}, trades: {bias.num_trades})")

        # Forward to paper trading if bias is actionable
        if paper_trading and bias.bias != "NEUTRAL":
            # Get current Polymarket price
            current_price = 0.5
            if polymarket_feed:
                markets = polymarket_feed.get_active_markets()
                market = markets.get(bias.symbol, {})
                current_price = market.get("price", 0.5)
                market_start = market.get("start_time", 0)
                market_end = market.get("end_time", 0)

                if market_start and market_end:
                    paper_trading.process_whale_bias(
                        whale_name=bias.whale_name,
                        symbol=bias.symbol,
                        bias=bias.bias,
                        bias_confidence=bias.confidence,
                        market_start=market_start,
                        market_end=market_end,
                        detection_latency_sec=bias.first_trade_elapsed,
                        polymarket_price=current_price,
                    )

        # Broadcast to frontend
        asyncio.create_task(
            broadcast({"type": "whale_ws_bias", "data": bias.to_dict()})
        )

    whale_ws_detector = get_whale_ws_detector()
    whale_ws_detector.on_whale_trade = on_ws_whale_trade
    whale_ws_detector.on_bias_update = on_ws_bias_update
    print("[Server] WebSocket whale detector initialized (targeting Account88888)")

    # Initialize Polymarket WebSocket for real-time prices
    global pm_ws_client

    def on_pm_price_update(update: PMPriceUpdate):
        """Handle real-time price update from Polymarket WebSocket"""
        # Broadcast to frontend for instant chart updates
        asyncio.create_task(
            broadcast({
                "type": "pm_price",
                "data": update.to_dict()
            })
        )

    pm_ws_client = get_pm_ws_client()
    pm_ws_client.on_price_update = on_pm_price_update
    print("[Server] Polymarket WebSocket client initialized for real-time prices")

    if not CLOB_AVAILABLE:
        print("[Server] WARNING: py-clob-client not installed - live trading disabled")

    # Pre-fill historical candles for indicator calculations (RSI, ADX, Supertrend)
    await prefill_historical_candles()

    # Start background tasks
    background_tasks.append(asyncio.create_task(binance_feed.connect()))
    background_tasks.append(asyncio.create_task(whale_tracker.start(
        poll_interval=DEFAULT_CONFIG.whale_poll_interval_sec
    )))
    background_tasks.append(asyncio.create_task(polymarket_feed.start(poll_interval=2.0)))
    background_tasks.append(asyncio.create_task(broadcast_momentum_loop()))
    background_tasks.append(asyncio.create_task(broadcast_markets_loop()))
    background_tasks.append(asyncio.create_task(paper_trading_loop()))
    background_tasks.append(asyncio.create_task(start_sync_loop()))
    background_tasks.append(asyncio.create_task(whale_detector.start()))
    background_tasks.append(asyncio.create_task(whale_ws_detector.start()))
    background_tasks.append(asyncio.create_task(whale_ws_market_feed_loop()))
    background_tasks.append(asyncio.create_task(pm_ws_client.start()))
    background_tasks.append(asyncio.create_task(pm_ws_market_feed_loop()))

    # Do initial time sync
    success, msg = await sync_with_polymarket()
    if not success:
        success, msg = await sync_with_ntp()
    print(f"[Server] Time sync: {msg}")

    # Load historical markets from file
    _load_historical_markets()

    # Register connections for health monitoring
    connection_monitor.register_connection("binance_ws")
    connection_monitor.register_connection("whale_tracker")
    connection_monitor.register_connection("polymarket_api")
    connection_monitor.register_connection("polymarket_trades")
    connection_monitor.register_connection("pm_ws")
    connection_monitor.register_connection("whale_ws")
    connection_monitor.register_connection("clob_client")
    print("[Server] Connection health monitoring initialized")

    # Start connection health monitoring loop
    background_tasks.append(asyncio.create_task(connection_health_loop()))

    # Start market data snapshot collection loop (V2)
    background_tasks.append(asyncio.create_task(snapshot_collection_loop()))

    print("[Server] Started all data feeds")


@app.on_event("shutdown")
async def shutdown():
    """
    Clean up on shutdown with graceful trading halt.

    This ensures:
    1. Kill switch is activated to halt any live trading
    2. Paper trading state is saved
    3. All background tasks are properly cancelled
    4. All WebSocket connections are closed
    """
    print("[Server] Initiating graceful shutdown...")

    # Activate kill switch for live trading if active
    if live_trading and live_trading.config.mode == "live":
        print("[Server] Activating kill switch for graceful shutdown")
        await live_trading.activate_kill_switch(reason="Server shutdown")

    # Save paper trading state
    if paper_trading:
        paper_trading._save_state()
        print("[Server] Paper trading state saved")

    # Save historical markets
    _save_historical_markets()
    print("[Server] Historical markets saved")

    # Stop data feeds
    if binance_feed:
        await binance_feed.stop()
    if whale_tracker:
        whale_tracker.stop()
    if polymarket_feed:
        polymarket_feed.stop()
    if whale_detector:
        await whale_detector.stop()
    if whale_ws_detector:
        await whale_ws_detector.stop()
    if pm_ws_client:
        pm_ws_client.stop()

    # Cancel background tasks
    for task in background_tasks:
        task.cancel()

    # Close WebSocket connections
    for client in list(ws_clients):
        try:
            await client.close()
        except Exception:
            pass

    print("[Server] Shutdown complete")


# ============================================================================
# BROADCAST HELPERS
# ============================================================================

async def broadcast(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if not ws_clients:
        return

    data = json.dumps(message)
    disconnected = set()

    # Create a copy to avoid "Set changed size during iteration" error
    clients_snapshot = list(ws_clients)

    for ws in clients_snapshot:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.add(ws)

    # Clean up disconnected clients
    for ws in disconnected:
        ws_clients.discard(ws)


async def broadcast_momentum_loop():
    """Periodically broadcast momentum signals (reduced frequency for memory)"""
    while True:
        await asyncio.sleep(3)  # Every 3 seconds (reduced from 1s for memory)

        if momentum_calc and ws_clients:
            signals = momentum_calc.get_all_signals()
            await broadcast({
                "type": "momentum",
                "data": signals,
            })


async def connection_health_loop():
    """
    Periodically check connection health and log warnings for stale connections.

    This loop runs every 30 seconds and:
    1. Checks if any connections are stale (no successful operation in 2 minutes)
    2. Logs warnings for unhealthy connections
    3. Can trigger reconnection callbacks if configured
    """
    while True:
        await asyncio.sleep(30)  # Check every 30 seconds

        try:
            status = connection_monitor.get_status()

            # Check for unhealthy connections
            unhealthy = []
            for name, info in status.items():
                if not info.get("is_healthy", True):
                    unhealthy.append(f"{name} (age: {info.get('last_success_age_sec', 'N/A')}s)")

            if unhealthy:
                print(f"[HealthCheck] WARNING: Unhealthy connections: {', '.join(unhealthy)}", flush=True)

                # Try to trigger reconnection for stale connections
                await connection_monitor.check_and_reconnect()

        except Exception as e:
            print(f"[HealthCheck] Error in health check loop: {e}", flush=True)


async def snapshot_collection_loop():
    """
    Collect price snapshots every 30 seconds for rolling 24-hour storage.

    V2 Phase 3: Data collection for historical analysis.
    """
    SYMBOLS = ["BTC", "ETH", "SOL"]

    while True:
        await asyncio.sleep(30)  # Every 30 seconds

        if not polymarket_feed or not binance_feed or not market_data_store:
            continue

        try:
            now = int(time.time())
            timing = polymarket_feed.get_next_market_time()

            # Only collect during active market windows
            if not timing.get("is_open"):
                continue

            for symbol in SYMBOLS:
                market = polymarket_feed.active_markets.get(symbol)
                if not market:
                    continue

                # Get current Binance price
                binance_price = binance_feed.get_current_price(f"{symbol}USDT")
                if not binance_price:
                    continue

                # Get momentum data for volume/orderbook
                momo = momentum_calc.get_signal(f"{symbol}USDT") if momentum_calc else None

                # Create snapshot
                snapshot = PriceSnapshot(
                    id=f"{symbol}_{now}",
                    timestamp=now,
                    symbol=symbol,
                    binance_price=binance_price,
                    pm_up_price=market.price,
                    pm_down_price=1.0 - market.price,
                    market_start=market.start_time,
                    market_end=market.end_time,
                    elapsed_sec=now - market.start_time,
                    volume_delta_usd=momo.get("volume_delta", 0) if momo else None,
                    orderbook_imbalance=momo.get("orderbook_imbalance", 0) if momo else None,
                )

                market_data_store.record_snapshot(snapshot)

        except Exception as e:
            print(f"[Snapshot] Error collecting snapshots: {e}", flush=True)


_markets_loop_counter = 0

async def broadcast_markets_loop():
    """Periodically broadcast active 15-min markets and timing"""
    global _markets_loop_counter
    while True:
        await asyncio.sleep(2)  # Every 2 seconds
        _markets_loop_counter += 1

        if polymarket_feed and ws_clients:
            # Only include chart_data every other cycle (4s) to reduce memory
            include_chart = (_markets_loop_counter % 2 == 0)
            data = {
                "active": polymarket_feed.get_active_markets(),
                "timing": polymarket_feed.get_next_market_time(),
                "trades": polymarket_feed.get_recent_trades(limit=20),
            }
            if include_chart:
                data["chart_data"] = get_chart_data_for_markets()

            await broadcast({
                "type": "markets_15m",
                "data": data,
            })


async def whale_ws_market_feed_loop():
    """
    Feed active market data to the WebSocket whale detector.

    The whale detector needs token IDs to subscribe to the correct
    Polymarket WebSocket channels for real-time trade detection.
    """
    last_window = 0

    while True:
        await asyncio.sleep(5)  # Check every 5 seconds

        if not polymarket_feed or not whale_ws_detector:
            continue

        try:
            # Get active markets
            active_markets = polymarket_feed.get_active_markets()

            if not active_markets:
                continue

            # Check if we're in a new window
            current_window = min(
                m.get("start_time", 0) for m in active_markets.values()
            ) if active_markets else 0

            # Only update detector when window changes or on first run
            if current_window != last_window:
                # Format markets for the WebSocket detector
                ws_markets = {}
                for symbol, market in active_markets.items():
                    ws_markets[symbol] = {
                        "start_time": market.get("start_time", 0),
                        "end_time": market.get("end_time", 0),
                        "up_token_id": market.get("up_token_id", market.get("token_id", "")),
                        "down_token_id": market.get("down_token_id", ""),
                    }

                whale_ws_detector.set_active_markets(ws_markets)
                last_window = current_window
                print(f"[WhaleWS Feed] Updated markets: {list(ws_markets.keys())}")

        except Exception as e:
            print(f"[WhaleWS Feed] Error: {e}")


async def pm_ws_market_feed_loop():
    """
    Feed active market token IDs to the Polymarket WebSocket client.

    This subscribes to real-time price updates for active 15-min markets,
    giving us instant price changes instead of 2-second polling.
    """
    last_window = 0
    subscribed_markets: set[str] = set()

    while True:
        await asyncio.sleep(5)  # Check every 5 seconds

        if not polymarket_feed or not pm_ws_client:
            continue

        try:
            # Get active markets
            active_markets = polymarket_feed.get_active_markets()

            if not active_markets:
                continue

            # Check if we're in a new window
            current_window = min(
                m.get("start_time", 0) for m in active_markets.values()
            ) if active_markets else 0

            # Get set of current market symbols
            current_symbols = set(active_markets.keys())

            # Check for new markets to subscribe
            new_markets = current_symbols - subscribed_markets

            if new_markets or current_window != last_window:
                for symbol in current_symbols:
                    market = active_markets.get(symbol, {})
                    up_token = market.get("up_token_id", market.get("token_id", ""))
                    down_token = market.get("down_token_id", "")

                    if up_token or down_token:
                        await pm_ws_client.subscribe_market(symbol, up_token, down_token)

                subscribed_markets = current_symbols
                last_window = current_window
                print(f"[PM-WS Feed] Subscribed to markets: {list(current_symbols)}")

        except Exception as e:
            print(f"[PM-WS Feed] Error: {e}")


# Track which market windows we've recorded open prices for
_recorded_opens: set[str] = set()  # "SYMBOL_market_start"

# Track market windows for resolution (even without positions)
_tracked_windows: dict[str, dict] = {}  # "SYMBOL_market_start" -> { symbol, start, end, open_price }
_resolved_windows: set[str] = set()  # Windows we've already resolved

# ============================================================================
# MARKET WINDOW CHART DATA TRACKING
# ============================================================================

# Configuration
CHART_UPDATE_INTERVAL_SEC = int(os.getenv("CHART_UPDATE_INTERVAL_SEC", "3"))  # Increased from 2 for memory

# Chart data storage: { "SYMBOL_market_start": { "symbol": str, "start_price": float, "data": [...] } }
_chart_data: dict[str, dict] = {}
_last_chart_update: dict[str, int] = {}  # "SYMBOL" -> last update timestamp


def fetch_binance_price_at_sync(symbol: str, timestamp: int) -> float | None:
    """
    Fetch the Binance price for a symbol at a specific timestamp.
    Uses Binance klines API to get the 1-minute candle starting at the timestamp.
    Returns the OPEN price of that candle (the price at market start).
    """
    import requests
    binance_symbol = f"{symbol}USDT".upper()

    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": binance_symbol,
            "interval": "1m",
            "startTime": timestamp * 1000,
            "limit": 1,
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                # Kline format: [open_time, open, high, low, close, ...]
                # Use OPEN price (index 1) as the "price to beat" at market start
                return float(data[0][1])
    except Exception as e:
        print(f"[Chart] Could not fetch historical price for {symbol} at {timestamp}: {e}", flush=True)

    return None


def get_chart_data_for_markets() -> dict[str, dict]:
    """Get chart data for all active market windows"""
    result = {}
    now = int(datetime.now().timestamp())

    for key, data in _chart_data.items():
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            symbol = parts[0]
            try:
                market_start = int(parts[1])
                # Only include if window is still active (within 15 min)
                if now < market_start + 900:
                    result[symbol] = data
            except ValueError:
                pass

    return result


def record_chart_datapoint(
    symbol: str,
    market_start: int,
    binance_price: float,
    up_price: float,
    start_price: float,
):
    """Record a data point for the market window chart"""
    key = f"{symbol}_{market_start}"
    now = int(datetime.now().timestamp())

    if key not in _chart_data:
        _chart_data[key] = {
            "symbol": symbol,
            "start_price": start_price,
            "data": [],
        }

    # Add data point
    _chart_data[key]["data"].append({
        "time": now,
        "binancePrice": round(binance_price, 2),
        "upPrice": round(up_price, 3),
        "downPrice": round(1 - up_price, 3),
    })

    # Keep last 300 points (reduced from 500 for memory - still covers ~10min at 2s intervals)
    if len(_chart_data[key]["data"]) > 300:
        _chart_data[key]["data"] = _chart_data[key]["data"][-300:]


def cleanup_old_chart_data():
    """Remove chart data for old market windows"""
    now = int(datetime.now().timestamp())
    # Keep data for windows that ended less than 5 minutes ago (reduced from 30min for memory)
    cutoff = now - 300

    keys_to_remove = []
    for key in _chart_data:
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            try:
                market_start = int(parts[1])
                # Assume 15-min window, so end is start + 900
                if market_start + 900 < cutoff:
                    keys_to_remove.append(key)
            except ValueError:
                pass

    for key in keys_to_remove:
        del _chart_data[key]


# ============================================================================
# HISTORICAL MARKET DATA STORAGE
# ============================================================================

# Store last N resolved markets for analysis
HISTORICAL_MARKET_LIMIT = int(os.getenv("HISTORICAL_MARKET_LIMIT", "10"))

# List of resolved markets with all data
_historical_markets: list[dict] = []


def store_resolved_market(
    symbol: str,
    market_start: int,
    market_end: int,
    binance_open: float,
    binance_close: float,
    resolution: str,
):
    """
    Store a resolved market for historical analysis.

    Keeps the last HISTORICAL_MARKET_LIMIT markets with:
    - Symbol, timing, resolution
    - Binance open/close prices
    - Price movement details
    - Any signals generated during the window
    - Chart data if available
    """
    global _historical_markets

    # Calculate price movement
    price_change = binance_close - binance_open
    price_change_pct = (price_change / binance_open) * 100 if binance_open > 0 else 0

    # Get chart data for this window if available
    chart_key = f"{symbol}_{market_start}"
    chart_data_copy = None
    if chart_key in _chart_data:
        chart_data_copy = _chart_data[chart_key].copy()

    # Get any signals generated during this window
    signals_in_window = []
    if paper_trading:
        for sig in paper_trading.recent_signals:
            if sig.symbol == symbol and market_start <= sig.timestamp <= market_end:
                signals_in_window.append(sig.to_dict())

    market_record = {
        "symbol": symbol,
        "market_start": market_start,
        "market_end": market_end,
        "start_time_str": datetime.fromtimestamp(market_start).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time_str": datetime.fromtimestamp(market_end).strftime("%Y-%m-%d %H:%M:%S"),
        "binance_open": round(binance_open, 2),
        "binance_close": round(binance_close, 2),
        "price_change": round(price_change, 2),
        "price_change_pct": round(price_change_pct, 4),
        "resolution": resolution,
        "signals": signals_in_window,
        "chart_data": chart_data_copy,
        "recorded_at": int(datetime.now().timestamp()),
    }

    # Avoid duplicates (same symbol + market_start)
    _historical_markets = [
        m for m in _historical_markets
        if not (m["symbol"] == symbol and m["market_start"] == market_start)
    ]

    # Add new record at beginning
    _historical_markets.insert(0, market_record)

    # Keep only last N markets
    if len(_historical_markets) > HISTORICAL_MARKET_LIMIT:
        _historical_markets = _historical_markets[:HISTORICAL_MARKET_LIMIT]

    # Reduce logging - only log if something significant changed
    # print(f"[Historical] Stored {symbol} market: {resolution} "
    #       f"({price_change_pct:+.3f}%) | Total stored: {len(_historical_markets)}")

    # Persist to file for recovery after restart
    _save_historical_markets()


def _save_historical_markets():
    """Save historical markets to JSON file"""
    try:
        with open("historical_markets.json", "w") as f:
            json.dump(_historical_markets, f, indent=2)
    except Exception as e:
        print(f"[Historical] Error saving: {e}")


def _load_historical_markets():
    """Load historical markets from JSON file"""
    global _historical_markets
    try:
        if os.path.exists("historical_markets.json"):
            with open("historical_markets.json", "r") as f:
                _historical_markets = json.load(f)
            print(f"[Historical] Loaded {len(_historical_markets)} markets from file")
    except Exception as e:
        print(f"[Historical] Error loading: {e}")


def get_historical_markets(symbol: str = None, limit: int = 10) -> list[dict]:
    """Get historical markets, optionally filtered by symbol"""
    markets = _historical_markets

    if symbol:
        markets = [m for m in markets if m["symbol"] == symbol.upper()]

    return markets[:limit]


async def paper_trading_loop():
    """
    Latency arbitrage paper trading loop.

    Exploits the 30-60 second gap between Binance price movements and
    Polymarket price adjustments. Runs every second to detect opportunities.

    Strategy:
    1. Record Binance price at each market window open
    2. Monitor current Binance price vs open
    3. Calculate implied UP probability from price move
    4. Compare to Polymarket UP price
    5. Execute when edge exceeds threshold (default 5%)
    """
    global _recorded_opens, _tracked_windows, _resolved_windows

    print("[PTLoop] Paper trading loop started", flush=True)
    iteration = 0

    while True:
        try:
            await asyncio.sleep(1)  # Check every second for latency opportunities
            iteration += 1

            now = int(datetime.now().timestamp())

            # Always broadcast paper account state (even when no markets active)
            if ws_clients and paper_trading and iteration % 5 == 0:
                await broadcast({
                    "type": "paper_account",
                    "data": paper_trading.get_account_summary(),
                })

            if not polymarket_feed or not binance_feed:
                if iteration % 10 == 0:
                    print(f"[PTLoop] Waiting for feeds: polymarket={polymarket_feed is not None}, binance={binance_feed is not None}", flush=True)
                continue

            # Get active 15-minute markets from Polymarket
            active_markets = polymarket_feed.get_active_markets()
            if not active_markets:
                continue

            # Debug: log active markets every 30 seconds
            if now % 30 == 0:
                print(f"[PTLoop] Processing {len(active_markets)} markets: {list(active_markets.keys())}", flush=True)

            # Determine if paper trading is active for signal generation
            paper_trading_active = paper_trading and paper_trading.config.enabled

            for symbol, market_data in active_markets.items():
                market_start = market_data.get("start_time", 0)
                market_end = market_data.get("end_time", 0)
                is_active = market_data.get("is_active", True)

                if not market_start or not market_end or not is_active:
                    print(f"[Debug] {symbol} skipped: start={market_start}, end={market_end}, active={is_active}", flush=True)
                    continue

                # Get current Binance price
                binance_sym = f"{symbol}usdt".lower()
                candles = binance_feed.candles.get(binance_sym, [])
                if not candles:
                    continue

                current_price = candles[-1].close

                # Record the open price at market window start (within first 10 seconds)
                elapsed = now - market_start
                open_key = f"{symbol}_{market_start}"

                if elapsed <= 10 and open_key not in _recorded_opens:
                    # Track window for chart data (always)
                    _tracked_windows[open_key] = {
                        "symbol": symbol,
                        "start": market_start,
                        "end": market_end,
                        "open_price": current_price,
                    }
                    _recorded_opens.add(open_key)
                    print(f"[Chart] {symbol} window open recorded: ${current_price:.2f}", flush=True)

                    # Also record for paper trading if active
                    if paper_trading_active:
                        paper_trading.record_window_open(symbol, market_start, current_price)

                # Clean up old keys (windows that ended)
                _recorded_opens = {k for k in _recorded_opens if int(k.split("_")[1]) > now - 1800}

                # Get current Polymarket UP price
                polymarket_up_price = market_data.get("price", 0.5)

                # Record chart data point at configured interval (always, not just when paper trading)
                last_update = _last_chart_update.get(symbol, 0)
                if now - last_update >= CHART_UPDATE_INTERVAL_SEC:
                    # Track window if not already tracked (allows late joining)
                    if open_key not in _tracked_windows:
                        # Fetch the actual price at market start from Binance history
                        historical_open = fetch_binance_price_at_sync(symbol, market_start)
                        actual_open_price = historical_open if historical_open else current_price

                        _tracked_windows[open_key] = {
                            "symbol": symbol,
                            "start": market_start,
                            "end": market_end,
                            "open_price": actual_open_price,
                        }
                        if historical_open:
                            print(f"[Chart] {symbol} late join: fetched historical open ${actual_open_price:.2f} (elapsed: {elapsed}s)", flush=True)
                        else:
                            print(f"[Chart] {symbol} late join: using current ${current_price:.2f} as fallback (elapsed: {elapsed}s)", flush=True)

                        # Also record for paper trading on late join (critical for latency strategy)
                        if paper_trading_active and open_key not in _recorded_opens:
                            paper_trading.record_window_open(symbol, market_start, actual_open_price)
                            _recorded_opens.add(open_key)
                            print(f"[PT] {symbol} late window open recorded: ${actual_open_price:.2f}", flush=True)

                    # Get start price from tracked windows
                    window_info = _tracked_windows.get(open_key, {})
                    start_price = window_info.get("open_price", current_price)
                    record_chart_datapoint(
                        symbol=symbol,
                        market_start=market_start,
                        binance_price=current_price,
                        up_price=polymarket_up_price,
                        start_price=start_price,
                    )
                    _last_chart_update[symbol] = now
                    # Reduce logging - only log every 15 seconds per symbol
                    if now % 15 == 0:
                        print(f"[Chart] {symbol} recorded: Binance=${current_price:.2f}, UP={polymarket_up_price:.1%}", flush=True)

                # Clean up old chart data periodically
                if now % 60 == 0:
                    cleanup_old_chart_data()

                # Paper trading signal generation (only when enabled)
                if not paper_trading_active:
                    continue

                # Get momentum data for enrichment
                momentum = {}
                if momentum_calc:
                    signals = momentum_calc.get_all_signals()
                    momentum = signals.get(f"{symbol}USDT", {})

                # Check for latency arbitrage opportunity
                signal = paper_trading.process_latency_opportunity(
                    symbol=symbol,
                    binance_current=current_price,
                    polymarket_up_price=polymarket_up_price,
                    market_start=market_start,
                    market_end=market_end,
                    momentum=momentum,
                )

                if signal:
                    print(f"[Latency] {symbol} SIGNAL: {signal.signal.value} | "
                          f"Edge: {signal.edge:.1%} | "
                          f"Binance: ${signal.momentum.get('binance_current', 0):.2f} | "
                          f"PM UP: {polymarket_up_price:.1%}", flush=True)

                    # Broadcast the signal to all clients
                    await broadcast({
                        "type": "paper_signal",
                        "data": signal.to_dict(),
                    })

                    # =========================================================
                    # LIVE TRADING: Send order to Polymarket if in live mode
                    # =========================================================
                    if live_trading and paper_trading.trading_mode == "live":
                        # Get token_ids from active markets
                        active_markets = polymarket_feed.get_active_markets() if polymarket_feed else {}
                        market_info = active_markets.get(symbol, {})
                        up_token_id = market_info.get("up_token_id") or market_info.get("token_id", "")
                        down_token_id = market_info.get("down_token_id", "")

                        if up_token_id:
                            try:
                                order = await live_trading.process_signal(
                                    signal=signal,
                                    token_id=up_token_id,
                                    current_price=polymarket_up_price,
                                    down_token_id=down_token_id,
                                )
                                if order:
                                    print(f"[LIVE] {symbol} ORDER PLACED: {order.id} | "
                                          f"Status: {order.status} | "
                                          f"Size: ${order.size_usd:.2f}", flush=True)
                            except Exception as e:
                                print(f"[LIVE] {symbol} ORDER ERROR: {e}", flush=True)
                        else:
                            print(f"[LIVE] {symbol} SKIPPED: No token_id available", flush=True)

            # =====================================================================
            # RESOLVE ALL MARKET WINDOWS (for historical tracking)
            # =====================================================================
            for window_key, window_info in list(_tracked_windows.items()):
                if window_key in _resolved_windows:
                    continue

                if now >= window_info["end"]:
                    hist_symbol = window_info["symbol"]
                    binance_sym = f"{hist_symbol}usdt".lower()
                    candles = binance_feed.candles.get(binance_sym, [])

                    if candles:
                        close_price = candles[-1].close
                        open_price = window_info["open_price"]
                        resolution = "UP" if close_price > open_price else "DOWN"

                        # Store for historical analysis (even if no position was taken)
                        store_resolved_market(
                            symbol=hist_symbol,
                            market_start=window_info["start"],
                            market_end=window_info["end"],
                            binance_open=open_price,
                            binance_close=close_price,
                            resolution=resolution,
                        )

                        _resolved_windows.add(window_key)

                        # Note: Alerts are only sent from paper_trading.resolve_market when we have a position
                        # No alert for markets where no position was taken

            # Clean up old tracked/resolved windows
            cutoff = now - 3600  # 1 hour
            _tracked_windows = {k: v for k, v in _tracked_windows.items() if v["end"] > cutoff}
            _resolved_windows = {k for k in _resolved_windows if int(k.split("_")[1]) > cutoff}

            # Check for market resolutions (positions that need settling)
            for position in list(paper_trading.account.positions):
                if now >= position.market_end:
                    # Get Binance prices for resolution - IMPORTANT: use lowercase key
                    binance_sym = f"{position.symbol}usdt".lower()
                    candles = binance_feed.candles.get(binance_sym, [])

                    # Get the open price we recorded
                    open_key = f"{position.symbol}_{position.market_start}"
                    open_price = paper_trading._binance_opens.get(open_key)

                    # Get current price as close price (market has ended)
                    close_price = None
                    if candles:
                        close_price = candles[-1].close

                    # Fallback: if no open price recorded, use first candle after market start
                    if not open_price and candles:
                        for c in candles:
                            candle_time = c.timestamp // 1000
                            if candle_time >= position.market_start:
                                open_price = c.open
                                print(f"[Resolution] {position.symbol} using candle fallback for open: ${open_price:.2f}", flush=True)
                                break

                    if open_price and close_price:
                        resolution = "UP" if close_price > open_price else "DOWN"
                        is_win = position.side == resolution
                        print(f"[Resolution] {position.symbol} RESOLVED: {resolution} | "
                              f"Open: ${open_price:.2f} -> Close: ${close_price:.2f} | "
                              f"Position: {position.side} | Result: {'WIN' if is_win else 'LOSS'}", flush=True)

                        paper_trading.resolve_market(
                            symbol=position.symbol,
                            market_start=position.market_start,
                            market_end=position.market_end,
                            binance_open=open_price,
                            binance_close=close_price,
                        )

                        # Store resolved market for historical analysis
                        store_resolved_market(
                            symbol=position.symbol,
                            market_start=position.market_start,
                            market_end=position.market_end,
                            binance_open=open_price,
                            binance_close=close_price,
                            resolution=resolution,
                        )
                    else:
                        # Log why resolution failed
                        print(f"[Resolution] {position.symbol} FAILED: "
                              f"open_price={open_price}, close_price={close_price}, "
                              f"candles_count={len(candles)}, key={binance_sym}", flush=True)

            # =====================================================================
            # RESOLVE LIVE TRADING POSITIONS
            # =====================================================================
            if live_trading and paper_trading.trading_mode == "live":
                for position in list(live_trading.open_positions):
                    if now >= position.market_end:
                        binance_sym = f"{position.symbol}usdt".lower()
                        candles = binance_feed.candles.get(binance_sym, [])

                        # Get open/close prices
                        open_key = f"{position.symbol}_{position.market_start}"
                        open_price = paper_trading._binance_opens.get(open_key)
                        close_price = candles[-1].close if candles else None

                        if open_price and close_price:
                            resolution = "UP" if close_price > open_price else "DOWN"
                            is_win = position.side == resolution

                            print(f"[LIVE Resolution] {position.symbol} RESOLVED: {resolution} | "
                                  f"Open: ${open_price:.2f} -> Close: ${close_price:.2f} | "
                                  f"Position: {position.side} | Result: {'WIN' if is_win else 'LOSS'}", flush=True)

                            try:
                                await live_trading.resolve_position(
                                    symbol=position.symbol,
                                    market_start=position.market_start,
                                    binance_open=open_price,
                                    binance_close=close_price,
                                )
                            except Exception as e:
                                print(f"[LIVE Resolution] ERROR resolving {position.symbol}: {e}", flush=True)
                        else:
                            print(f"[LIVE Resolution] {position.symbol} FAILED: "
                                  f"open_price={open_price}, close_price={close_price}", flush=True)

        except Exception as e:
            print(f"[PTLoop] ERROR in iteration {iteration}: {e}", flush=True)
            import traceback
            traceback.print_exc()


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Basic status"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/health")
async def health_check():
    """
    Production health check endpoint.
    Returns detailed status for monitoring systems.
    """
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": 0,
        "components": {
            "binance_feed": "unknown",
            "polymarket_feed": "unknown",
            "paper_trading": "unknown",
            "websocket_clients": 0,
        }
    }

    # Check Binance feed
    if binance_feed:
        has_data = any(binance_feed.candles.values())
        status["components"]["binance_feed"] = "healthy" if has_data else "degraded"
    else:
        status["components"]["binance_feed"] = "not_initialized"

    # Check Polymarket feed
    if polymarket_feed:
        status["components"]["polymarket_feed"] = "healthy"
    else:
        status["components"]["polymarket_feed"] = "not_initialized"

    # Check paper trading
    if paper_trading:
        status["components"]["paper_trading"] = "healthy"
        status["paper_trading"] = {
            "enabled": paper_trading.config.enabled,
            "enabled_assets": paper_trading.config.enabled_assets,
            "balance": paper_trading.account.balance,
            "open_positions": len(paper_trading.account.positions),
            "total_trades": paper_trading.account.total_trades,
            "trading_halted": paper_trading.account.trading_halted,
        }
    else:
        status["components"]["paper_trading"] = "not_initialized"

    # Check live trading
    if live_trading:
        mode = live_trading.config.mode.value
        kill_switch = live_trading.kill_switch_active
        circuit_breaker = live_trading.circuit_breaker.triggered

        if kill_switch or circuit_breaker:
            status["components"]["live_trading"] = "halted"
        else:
            status["components"]["live_trading"] = "healthy"

        status["live_trading"] = {
            "mode": mode,
            "kill_switch_active": kill_switch,
            "circuit_breaker_triggered": circuit_breaker,
            "circuit_breaker_reason": live_trading.circuit_breaker.reason,
            "enabled_assets": live_trading.config.enabled_assets,
            "open_positions": len(live_trading.open_positions),
            "clob_connected": live_trading.clob_client is not None,
        }
    else:
        status["components"]["live_trading"] = "not_initialized"

    # Check whale detector (polling-based)
    if whale_detector:
        status["components"]["whale_detector"] = "healthy" if whale_detector._running else "stopped"
        status["whale_detector"] = {
            "running": whale_detector._running,
            "whales_tracked": whale_detector.whales,
            "polls_made": whale_detector.polls_made,
            "trades_detected": whale_detector.trades_detected,
            "biases_generated": whale_detector.biases_generated,
        }
    else:
        status["components"]["whale_detector"] = "not_initialized"

    # Check WebSocket whale detector (low-latency)
    if whale_ws_detector:
        status["components"]["whale_ws_detector"] = "healthy" if whale_ws_detector._running else "stopped"
        active_markets = whale_ws_detector._active_markets
        status["whale_ws_detector"] = {
            "running": whale_ws_detector._running,
            "target_wallet": "Account88888",
            "active_markets": list(active_markets.keys()) if active_markets else [],
            "subscribed_tokens": len(whale_ws_detector.get_token_to_symbol()),
            "latency_target_ms": "5000-15000",
        }
    else:
        status["components"]["whale_ws_detector"] = "not_initialized"

    # WebSocket clients
    status["components"]["websocket_clients"] = len(ws_clients)

    # Connection health monitoring
    try:
        conn_status = connection_monitor.get_status()
        status["connections"] = conn_status

        # Check for unhealthy connections
        unhealthy_conns = [
            name for name, info in conn_status.items()
            if not info.get("is_healthy", True)
        ]
        if unhealthy_conns:
            status["components"]["connection_health"] = "degraded"
            status["unhealthy_connections"] = unhealthy_conns
        else:
            status["components"]["connection_health"] = "healthy"
    except Exception as e:
        status["components"]["connection_health"] = "error"
        status["connection_health_error"] = str(e)

    # Overall status
    unhealthy = [k for k, v in status["components"].items() if v not in ("healthy", "degraded") and k != "websocket_clients"]
    if unhealthy:
        status["status"] = "unhealthy"

    return status


@app.get("/api/symbols")
async def get_symbols():
    """Get available crypto symbols"""
    return {
        "symbols": list(CryptoExchangeAPI.SYMBOLS.keys()),
        "polymarket_pairs": {
            "BTC": "btcusdt",
            "ETH": "ethusdt",
            "SOL": "solusdt",
            "XRP": "xrpusdt",
            "DOGE": "dogeusdt",
        }
    }


@app.get("/api/whales")
async def get_whales():
    """Get tracked whale wallets"""
    return {
        "whales": [
            {
                "name": name,
                "address": info["address"][:16] + "..." if len(info["address"]) > 16 else info["address"],
                "strategy": info.get("strategy", "Unknown"),
                "focus": info.get("focus", []),
            }
            for name, info in CRYPTO_WHALE_WALLETS.items()
        ]
    }


@app.get("/api/whale-trades")
async def get_whale_trades(limit: int = 50):
    """Get recent whale trades"""
    if not whale_tracker:
        return {"trades": []}
    return {"trades": whale_tracker.get_recent_trades(limit)}


# =============================================================================
# WHALE FOLLOWING STRATEGY ENDPOINTS
# =============================================================================

@app.get("/api/whale-following/backtest/{whale_name}")
async def backtest_whale(whale_name: str, days: int = 7, position_size: float = 25.0):
    """
    Backtest the whale following strategy.

    Fetches historical trades from the specified whale on 15-min markets
    and simulates following their trades.

    Args:
        whale_name: Name of whale (e.g., "gabagool22")
        days: Number of days to backtest (default 7)
        position_size: USD per trade (default $25)
    """
    try:
        result = await backtest_whale_following(
            whale_name=whale_name,
            days=days,
            position_size=position_size,
        )
        return result.to_dict()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Backtest failed: {str(e)}"}, status_code=500)


@app.get("/api/whale-following/history/{whale_name}")
async def get_whale_history(whale_name: str, limit: int = 50):
    """
    Get a whale's recent 15-min market activity.

    Returns trades grouped by market with position information.
    """
    try:
        markets = await fetch_whale_market_history(whale_name, limit)
        return {
            "whale_name": whale_name,
            "markets_count": len(markets),
            "markets": markets,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/whale-following/positions/{whale_name}")
async def get_whale_positions_pnl(whale_name: str):
    """
    Get a whale's REAL position P&L from Polymarket.

    This fetches actual settled positions and realized P&L rather than
    simulated outcomes. Much more accurate than the trade-based backtest.

    Returns:
        Real performance metrics including total P&L, win rate, and ROI
    """
    try:
        result = await fetch_whale_positions_pnl(whale_name)
        return result
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Failed to fetch positions: {str(e)}"}, status_code=500)


@app.get("/api/whale-following/leaderboard")
async def get_whale_leaderboard():
    """
    Get performance leaderboard for tracked whales.

    Returns stats on which whales to follow based on recent performance.
    """
    leaderboard = []

    for name, info in CRYPTO_WHALE_WALLETS.items():
        try:
            # Quick backtest for each whale
            result = await backtest_whale_following(
                whale_name=name,
                days=3,  # Last 3 days
                position_size=25.0,
            )

            if result.total_trades > 0:
                leaderboard.append({
                    "name": name,
                    "strategy": info.get("strategy", "Unknown"),
                    "trades": result.total_trades,
                    "win_rate": result.win_rate,
                    "roi_pct": result.roi_pct,
                    "total_pnl": result.total_pnl,
                })
        except Exception:
            continue

    # Sort by ROI
    leaderboard.sort(key=lambda x: x["roi_pct"], reverse=True)

    return {
        "period_days": 3,
        "whales": leaderboard,
    }


@app.get("/api/whale-detector/status")
async def get_whale_detector_status():
    """
    Get real-time whale trade detector status.

    Returns detection stats and active market biases.
    """
    if not whale_detector:
        raise HTTPException(status_code=503, detail="Whale detector not initialized")

    stats = whale_detector.get_stats()
    active_biases = whale_detector.get_active_biases()

    return {
        **stats,
        "active_biases": [b.to_dict() for b in active_biases],
    }


@app.get("/api/candles/{symbol}")
async def get_candles(symbol: str, interval: str = "1m", limit: int = 500):
    """Get historical OHLCV candles"""
    # Map symbol to Binance format
    symbol_map = {v["binance"]: k for k, v in CryptoExchangeAPI.SYMBOLS.items()}
    binance_symbol = None

    for k, v in CryptoExchangeAPI.SYMBOLS.items():
        if k.upper() == symbol.upper():
            binance_symbol = v["binance"].upper()
            break

    if not binance_symbol:
        return JSONResponse({"error": f"Unknown symbol: {symbol}"}, status_code=400)

    candles = await fetch_historical_candles(binance_symbol, interval, limit)
    return {"symbol": symbol.upper(), "interval": interval, "candles": candles}


@app.get("/api/momentum")
async def get_momentum():
    """Get current momentum signals for all symbols"""
    if not momentum_calc:
        return {"signals": {}}
    return {"signals": momentum_calc.get_all_signals()}


@app.get("/api/orderbook/{symbol}")
async def get_orderbook(symbol: str):
    """Get current order book for symbol"""
    if not binance_feed:
        return {"error": "Feed not initialized"}

    binance_symbol = None
    for k, v in CryptoExchangeAPI.SYMBOLS.items():
        if k.upper() == symbol.upper():
            binance_symbol = v["binance"]
            break

    if not binance_symbol:
        return JSONResponse({"error": f"Unknown symbol: {symbol}"}, status_code=400)

    book = binance_feed.orderbooks.get(binance_symbol)
    if not book:
        return {"error": "Order book not available"}

    return {
        "symbol": symbol.upper(),
        "mid": book.mid_price,
        "spread": book.spread,
        "imbalance": book.imbalance,
        "bids": [[l.price, l.size] for l in book.bids],
        "asks": [[l.price, l.size] for l in book.asks],
    }


@app.get("/api/debug/ws-candles")
async def debug_ws_candles():
    """Debug endpoint to check WebSocket candle state"""
    if not binance_feed:
        return {"error": "Feed not initialized"}

    result = {}
    for sym in ["btcusdt", "ethusdt", "solusdt", "xrpusdt", "dogeusdt"]:
        candles = binance_feed.candles.get(sym, [])
        result[sym] = {
            "count": len(candles),
            "last": candles[-1].to_dict() if candles else None,
        }
    return {
        "candles": result,
        "chart_data_keys": list(_chart_data.keys()),
        "tracked_windows": list(_tracked_windows.keys()),
    }


@app.get("/api/time-sync")
async def get_time_sync():
    """
    Get time synchronization status.

    Returns current sync state including:
    - offset_ms: Difference between local and server time
    - last_sync: When last sync occurred
    - source: Time source (polymarket, google, etc.)
    - status: synced, sync_failed, not_synced
    - drift_ms: Absolute clock drift detected
    """
    status = get_sync_status()
    return {
        **status,
        "synced_timestamp": get_synced_timestamp(),
    }


@app.post("/api/time-sync/refresh")
async def refresh_time_sync():
    """Force a time sync refresh"""
    success, msg = await sync_with_polymarket()
    if not success:
        success, msg = await sync_with_ntp()

    return {
        "success": success,
        "message": msg,
        **get_sync_status(),
    }


@app.get("/api/markets-15m")
async def get_markets_15m():
    """Get active 15-minute markets and timing"""
    if not polymarket_feed:
        return {"active": {}, "timing": {}, "trades": [], "chart_data": {}}

    return {
        "active": polymarket_feed.get_active_markets(),
        "timing": polymarket_feed.get_next_market_time(),
        "trades": polymarket_feed.get_recent_trades(limit=50),
        "chart_data": get_chart_data_for_markets(),
    }


@app.get("/api/markets-15m/{symbol}")
async def get_market_15m_symbol(symbol: str):
    """Get 15-minute market for a specific symbol"""
    if not polymarket_feed:
        return {"error": "Feed not initialized"}

    market = polymarket_feed.active_markets.get(symbol.upper())
    if not market:
        return {"error": f"No active market for {symbol}"}

    return {
        "market": market.to_dict(),
        "trades": polymarket_feed.get_recent_trades(symbol=symbol.upper(), limit=50),
    }


@app.get("/api/markets-15m-trades")
async def get_market_trades(symbol: str = None, limit: int = 50):
    """Get recent trades on 15-minute markets"""
    if not polymarket_feed:
        return {"trades": []}

    return {
        "trades": polymarket_feed.get_recent_trades(symbol=symbol.upper() if symbol else None, limit=limit),
    }


@app.get("/api/markets-15m/history")
async def get_market_history(symbol: str = None, limit: int = 10):
    """
    Get historical resolved markets for analysis.

    Returns the last N resolved 15-minute markets with:
    - BTC/crypto price at open and close
    - Resolution (UP/DOWN)
    - Price change percentage
    - Any signals that were generated
    - Chart data if available

    Args:
        symbol: Filter by symbol (BTC, ETH, etc.)
        limit: Max markets to return (default 10, max 50)

    Example response:
    {
      "markets": [
        {
          "symbol": "BTC",
          "market_start": 1704067200,
          "start_time_str": "2024-01-01 00:00:00",
          "binance_open": 42500.00,
          "binance_close": 42520.00,
          "price_change_pct": 0.047,
          "resolution": "UP",
          "signals": [...],
          "chart_data": {...}
        }
      ]
    }
    """
    limit = min(limit, 50)  # Cap at 50
    markets = get_historical_markets(symbol=symbol, limit=limit)

    return {
        "markets": markets,
        "total_stored": len(_historical_markets),
        "limit_configured": HISTORICAL_MARKET_LIMIT,
    }


# ============================================================================
# MARKET CONFIGURATION ENDPOINTS (V2)
# ============================================================================

# Supported market timeframes (15m is active, others planned)
MARKET_TIMEFRAMES = {
    "15m": {"duration_sec": 900, "slug_pattern": "{sym}-updown-15m-{ts}", "active": True},
    "1h": {"duration_sec": 3600, "slug_pattern": "{sym}-updown-1h-{ts}", "active": False},
    "4h": {"duration_sec": 14400, "slug_pattern": "{sym}-updown-4h-{ts}", "active": False},
    "1d": {"duration_sec": 86400, "slug_pattern": "{sym}-updown-1d-{ts}", "active": False},
}

# Symbols with viable volume for trading
VIABLE_SYMBOLS = {
    "BTC": {"volume_24h": 165000, "tier": "primary"},
    "ETH": {"volume_24h": 52000, "tier": "primary"},
    "SOL": {"volume_24h": 15000, "tier": "secondary"},
    "XRP": {"volume_24h": 8000, "tier": "secondary"},
    "DOGE": {"volume_24h": 5000, "tier": "secondary"},
}


class MarketConfigUpdate(BaseModel):
    enabled_symbols: Optional[List[str]] = Field(None, min_length=1, max_length=5)
    enabled_timeframes: Optional[List[str]] = Field(None, min_length=1, max_length=4)


@app.get("/api/markets/config")
async def get_markets_config():
    """
    Get current market configuration.

    Returns enabled symbols, timeframes, and their settings.
    """
    # Get enabled symbols from paper trading config
    enabled_symbols = paper_trading.config.enabled_assets if paper_trading else ["BTC", "ETH"]

    return {
        "enabled_symbols": enabled_symbols,
        "enabled_timeframes": ["15m"],  # Only 15m is currently active
        "available_symbols": VIABLE_SYMBOLS,
        "available_timeframes": MARKET_TIMEFRAMES,
    }


@app.post("/api/markets/config", dependencies=[Depends(verify_api_key)])
async def update_markets_config(config: MarketConfigUpdate):
    """
    Update market configuration.

    Allows enabling/disabling symbols and timeframes (when available).
    Requires API key authentication.
    """
    if not paper_trading:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    updates = {}

    if config.enabled_symbols:
        # Validate symbols
        for sym in config.enabled_symbols:
            if sym not in VIABLE_SYMBOLS:
                raise HTTPException(status_code=400, detail=f"Invalid symbol: {sym}")
        updates["enabled_assets"] = config.enabled_symbols

    if config.enabled_timeframes:
        # Validate timeframes - only 15m is currently active
        for tf in config.enabled_timeframes:
            if tf not in MARKET_TIMEFRAMES:
                raise HTTPException(status_code=400, detail=f"Invalid timeframe: {tf}")
            if not MARKET_TIMEFRAMES[tf]["active"]:
                raise HTTPException(status_code=400, detail=f"Timeframe {tf} not yet available on Polymarket")

    if updates:
        paper_trading.update_config(updates)

    return {
        "success": True,
        "config": {
            "enabled_symbols": paper_trading.config.enabled_assets,
            "enabled_timeframes": ["15m"],
        }
    }


@app.get("/api/markets/available")
async def get_available_markets():
    """
    Check which markets are currently available on Polymarket.

    Returns a grid of Symbol x Timeframe availability.
    """
    available = {}

    for sym in VIABLE_SYMBOLS:
        available[sym] = {}
        for tf, tf_config in MARKET_TIMEFRAMES.items():
            # Check if market exists for this symbol/timeframe
            if tf == "15m" and polymarket_feed:
                market = polymarket_feed.active_markets.get(sym)
                available[sym][tf] = {
                    "exists": market is not None,
                    "active": tf_config["active"],
                    "volume_24h": VIABLE_SYMBOLS[sym]["volume_24h"],
                    "current_market": market.to_dict() if market else None,
                }
            else:
                available[sym][tf] = {
                    "exists": False,
                    "active": tf_config["active"],
                    "volume_24h": VIABLE_SYMBOLS[sym]["volume_24h"],
                    "current_market": None,
                }

    return {
        "symbols": list(VIABLE_SYMBOLS.keys()),
        "timeframes": list(MARKET_TIMEFRAMES.keys()),
        "availability": available,
    }


# ============================================================================
# HISTORY/DATA ENDPOINTS (V2 Phase 3)
# ============================================================================

@app.get("/api/history/snapshots")
async def get_history_snapshots(
    symbol: Optional[str] = None,
    market_start: Optional[int] = None,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100,
):
    """
    Get price snapshots for historical analysis.

    Args:
        symbol: Filter by symbol (BTC, ETH, SOL)
        market_start: Filter by specific 15-min market window
        since: Filter after this Unix timestamp
        until: Filter before this Unix timestamp
        limit: Max results (default 100, max 1000)

    Returns:
        List of price snapshots with Binance and Polymarket prices.
    """
    if not market_data_store:
        return {"error": "Market data store not initialized", "snapshots": []}

    limit = min(limit, 1000)
    snapshots = market_data_store.get_snapshots(
        symbol=symbol,
        market_start=market_start,
        since=since,
        until=until,
        limit=limit,
    )

    return {
        "snapshots": [s.to_dict() for s in snapshots],
        "count": len(snapshots),
    }


@app.get("/api/history/trades")
async def get_history_trades(
    symbol: Optional[str] = None,
    market_start: Optional[int] = None,
    side: Optional[str] = None,
    min_usd: Optional[float] = None,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100,
):
    """
    Get market trades for historical analysis.

    Args:
        symbol: Filter by symbol (BTC, ETH, SOL)
        market_start: Filter by specific 15-min market window
        side: Filter by "UP" or "DOWN"
        min_usd: Minimum trade size in USD
        since: Filter after this Unix timestamp
        until: Filter before this Unix timestamp
        limit: Max results (default 100, max 1000)

    Returns:
        List of market trades.
    """
    if not market_data_store:
        return {"error": "Market data store not initialized", "trades": []}

    limit = min(limit, 1000)
    trades = market_data_store.get_trades(
        symbol=symbol,
        market_start=market_start,
        side=side,
        min_usd=min_usd,
        since=since,
        until=until,
        limit=limit,
    )

    return {
        "trades": [t.to_dict() for t in trades],
        "count": len(trades),
    }


@app.get("/api/history/market-analysis/{market_start}")
async def get_market_analysis(market_start: int, symbol: str = "BTC"):
    """
    Get analysis for a specific market window.

    Args:
        market_start: The 15-min window start timestamp
        symbol: Symbol to analyze (default BTC)

    Returns:
        Analysis with price evolution, trade volume, etc.
    """
    if not market_data_store:
        return {"error": "Market data store not initialized"}

    return market_data_store.get_market_analysis(market_start=market_start, symbol=symbol)


@app.get("/api/history/stats")
async def get_history_stats():
    """
    Get market data store statistics.

    Returns:
        Snapshot count, trade count, time range, etc.
    """
    if not market_data_store:
        return {"error": "Market data store not initialized"}

    return market_data_store.get_stats()


# ============================================================================
# PAPER TRADING ENDPOINTS
# ============================================================================

@app.get("/api/paper-trading/account")
async def get_paper_account():
    """Get paper trading account summary"""
    if not paper_trading:
        return {"error": "Paper trading not initialized"}
    return paper_trading.get_account_summary()


@app.get("/api/paper-trading/config")
async def get_paper_config():
    """Get paper trading configuration"""
    if not paper_trading:
        return {"error": "Paper trading not initialized"}
    return paper_trading.get_config()


@app.post("/api/paper-trading/config")
async def update_paper_config(config: dict):
    """Update paper trading configuration"""
    if not paper_trading:
        return {"error": "Paper trading not initialized"}

    try:
        paper_trading.update_config(**config)
        return {"status": "ok", "config": paper_trading.get_config()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/paper-trading/toggle")
async def toggle_paper_trading():
    """Toggle paper trading on/off"""
    if not paper_trading:
        return {"error": "Paper trading not initialized"}

    paper_trading.config.enabled = not paper_trading.config.enabled
    paper_trading._save_state()
    return {"enabled": paper_trading.config.enabled}


@app.post("/api/paper-trading/reset")
async def reset_paper_account():
    """Reset paper trading account to initial state"""
    if not paper_trading:
        return {"error": "Paper trading not initialized"}

    paper_trading.reset_account()
    return {"status": "ok", "account": paper_trading.get_account_summary()}


@app.post("/api/paper-trading/factory-reset")
async def factory_reset_paper_trading():
    """Reset paper trading to factory defaults (account + all settings)"""
    if not paper_trading:
        return {"error": "Paper trading not initialized"}

    paper_trading.reset_to_defaults()
    return {
        "status": "ok",
        "account": paper_trading.get_account_summary(),
        "config": paper_trading.get_config(),
    }


@app.get("/api/paper-trading/positions")
async def get_paper_positions():
    """Get open paper trading positions"""
    if not paper_trading:
        return {"positions": []}
    return {"positions": paper_trading.get_positions()}


@app.get("/api/paper-trading/trades")
async def get_paper_trades(limit: int = 50):
    """Get paper trading trade history"""
    if not paper_trading:
        return {"trades": []}
    return {"trades": paper_trading.get_trade_history(limit)}


@app.get("/api/paper-trading/signals")
async def get_paper_signals(limit: int = 20):
    """Get recent paper trading signals"""
    if not paper_trading:
        return {"signals": []}
    return {"signals": paper_trading.get_recent_signals(limit)}


# ============================================================================
# LIVE TRADING ENDPOINTS (Protected by API Key)
# ============================================================================

@app.get("/api/live-trading/status")
async def get_live_status(_: str = Depends(verify_api_key)):
    """Get live trading status and mode (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}
    return live_trading.get_status()


@app.get("/api/live-trading/config")
async def get_live_config(_: str = Depends(verify_api_key)):
    """Get live trading configuration (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}
    return live_trading.config.to_dict()


@app.post("/api/live-trading/config")
async def update_live_config(
    request: ConfigUpdateRequest,
    _: str = Depends(verify_api_key)
):
    """Update live trading configuration (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}

    try:
        # Only update fields that were provided
        updates = request.model_dump(exclude_none=True)
        live_trading.update_config(**updates)
        return {"status": "ok", "config": live_trading.config.to_dict()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/live-trading/mode")
async def set_trading_mode(
    request: TradingModeRequest,
    _: str = Depends(verify_api_key)
):
    """
    Set trading mode (requires API key):
    - paper: Real signals, simulated execution (no real money)
    - live: Real money trades via Polymarket CLOB (CAUTION!)
    """
    if not live_trading:
        return {"error": "Live trading not initialized"}

    allowance_warning = None

    # Safety checks for live mode
    if request.mode == "live":
        if not CLOB_AVAILABLE:
            return JSONResponse({"error": "py-clob-client not installed - cannot enable live mode"}, status_code=400)
        if not os.getenv("POLYMARKET_PRIVATE_KEY"):
            return JSONResponse({"error": "POLYMARKET_PRIVATE_KEY not set - cannot enable live mode"}, status_code=400)

        # Initialize CLOB client if not already done
        clob_ok, clob_msg = live_trading.ensure_clob_initialized()
        if not clob_ok:
            return JSONResponse({"error": f"Failed to initialize CLOB: {clob_msg}"}, status_code=400)

        # Check token allowances - warn but don't block (user can set after entering live mode)
        is_approved, message, _ = live_trading.check_token_allowances()
        if not is_approved:
            allowance_warning = f"Token allowances not set. You must set allowances before placing orders."

    live_trading.set_mode(request.mode)

    # Sync trading mode to paper trading engine for alerts
    if paper_trading:
        paper_trading.trading_mode = request.mode

    # Broadcast updated live trading status to all WebSocket clients
    await broadcast({
        "type": "live_status",
        "data": live_trading.get_status(),
    })

    response = {"status": "ok", "mode": request.mode}
    if request.mode == "live" and allowance_warning:
        response["warning"] = allowance_warning
    return response


@app.post("/api/live-trading/kill-switch")
async def toggle_kill_switch(
    request: KillSwitchRequest,
    _: str = Depends(verify_api_key)
):
    """Activate or deactivate the kill switch (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}

    if request.activate:
        await live_trading.activate_kill_switch(request.reason)
        return {"status": "ok", "kill_switch": True, "reason": request.reason}
    else:
        await live_trading.deactivate_kill_switch()
        return {"status": "ok", "kill_switch": False}


@app.get("/api/live-trading/positions")
async def get_live_positions(_: str = Depends(verify_api_key)):
    """Get open live trading positions (requires API key)"""
    if not live_trading:
        return {"positions": []}
    return {"positions": live_trading.get_positions()}


@app.get("/api/live-trading/orders")
async def get_live_orders(limit: int = 50, _: str = Depends(verify_api_key)):
    """Get live trading order history (requires API key)"""
    if not live_trading:
        return {"orders": []}
    return {"orders": live_trading.get_order_history(limit)}


@app.get("/api/live-trading/circuit-breaker")
async def get_circuit_breaker(_: str = Depends(verify_api_key)):
    """Get circuit breaker status (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}
    return live_trading.circuit_breaker.to_dict()


@app.post("/api/live-trading/circuit-breaker/reset")
async def reset_circuit_breaker(_: str = Depends(verify_api_key)):
    """Reset circuit breaker (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}

    live_trading.circuit_breaker.triggered = False
    live_trading.circuit_breaker.reason = ""
    live_trading._save_state()
    return {"status": "ok", "circuit_breaker": live_trading.circuit_breaker.to_dict()}


@app.post("/api/live-trading/refresh-credentials")
async def refresh_api_credentials(_: str = Depends(verify_api_key)):
    """
    Refresh Polymarket API credentials.
    Call this if getting 'invalid signature' errors (requires API key).
    """
    if not live_trading:
        return {"error": "Live trading not initialized"}

    if not live_trading.clob_client:
        # Try to initialize first
        ok, msg = live_trading.ensure_clob_initialized()
        if not ok:
            return {"error": f"CLOB not available: {msg}"}

    success = live_trading.refresh_api_credentials()
    if success:
        return {"status": "ok", "message": "API credentials refreshed successfully"}
    else:
        return JSONResponse(
            {"error": "Failed to refresh credentials - check logs"},
            status_code=500
        )


@app.post("/api/live-trading/enabled-assets")
async def set_enabled_assets(
    request: EnabledAssetsRequest,
    _: str = Depends(verify_api_key)
):
    """Set which assets are enabled for live trading (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}

    # Validate assets
    valid_assets = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
    invalid = [a for a in request.assets if a.upper() not in valid_assets]
    if invalid:
        return JSONResponse({"error": f"Invalid assets: {invalid}"}, status_code=400)

    live_trading.config.enabled_assets = [a.upper() for a in request.assets]
    live_trading._save_state()
    return {"status": "ok", "enabled_assets": live_trading.config.enabled_assets}


@app.get("/api/live-trading/wallet")
async def get_wallet_balance(_: str = Depends(verify_api_key)):
    """Get wallet USDC balance (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}

    # Initialize CLOB if needed
    clob_ok, clob_msg = live_trading.ensure_clob_initialized()
    if not clob_ok:
        return {"error": f"CLOB not available: {clob_msg}"}

    return live_trading.get_wallet_balance()


@app.get("/api/live-trading/allowances")
async def check_allowances(_: str = Depends(verify_api_key)):
    """
    Check if token allowances are set for Polymarket (requires API key).

    Per Polymarket docs: Allowances must be set ONCE per wallet before trading.
    This checks if they're already set.
    """
    if not live_trading:
        return {"error": "Live trading not initialized"}

    # Initialize CLOB if needed
    clob_ok, clob_msg = live_trading.ensure_clob_initialized()
    if not clob_ok:
        return {"approved": False, "message": f"CLOB not available: {clob_msg}"}

    is_approved, message, details = live_trading.check_token_allowances()
    return {
        "approved": is_approved,
        "message": message,
        **details,
    }


@app.post("/api/live-trading/allowances")
async def set_allowances(_: str = Depends(verify_api_key)):
    """
    Set token allowances for Polymarket exchange contracts (requires API key).

    This approves USDC and conditional tokens for the exchange.
    Only needs to be done ONCE per wallet - costs gas.
    """
    if not live_trading:
        return {"error": "Live trading not initialized"}

    # Initialize CLOB if needed
    clob_ok, clob_msg = live_trading.ensure_clob_initialized()
    if not clob_ok:
        return JSONResponse({"error": f"CLOB not available: {clob_msg}"}, status_code=400)

    success, message = live_trading.set_allowances()
    if not success:
        return JSONResponse({"error": message}, status_code=400)

    return {"status": "ok", "message": message}


@app.post("/api/live-trading/test-order")
async def place_test_order(
    request: TestOrderRequest,
    _: str = Depends(verify_api_key)
):
    """
    Place a manual test order (requires API key).

    Use this to test if live trading is working.
    Amount is limited to $1-$100 for safety.
    """
    if not live_trading:
        return JSONResponse({"error": "Live trading not initialized"}, status_code=400)

    if live_trading.config.mode.value != "live":
        return JSONResponse({"error": "Not in live mode - switch to live mode first"}, status_code=400)

    # Initialize CLOB if needed
    clob_ok, clob_msg = live_trading.ensure_clob_initialized()
    if not clob_ok:
        return JSONResponse({"error": f"CLOB not available: {clob_msg}"}, status_code=400)

    # Get current active market for the symbol
    if not polymarket_feed:
        return JSONResponse({"error": "Polymarket feed not available"}, status_code=400)

    active_markets = polymarket_feed.get_active_markets()
    market_info = active_markets.get(request.symbol)

    if not market_info:
        return JSONResponse({"error": f"No active market for {request.symbol}"}, status_code=400)

    # Get the token_id based on side
    token_id = market_info.get("up_token_id" if request.side == "UP" else "down_token_id")
    if not token_id:
        return JSONResponse({"error": f"Token ID not found for {request.symbol} {request.side}"}, status_code=400)

    # Create the order
    from live_trading import LiveOrder
    import time
    import uuid

    order = LiveOrder(
        id=f"test-{uuid.uuid4().hex[:8]}",
        symbol=request.symbol,
        side=request.side,
        direction="BUY",  # Test orders are always buys
        token_id=token_id,
        size_usd=request.amount_usd,
        price=0.50,  # Market order at ~50%
        order_type="FOK",  # Fill-or-kill for immediate execution
        status="pending",
        created_at=int(time.time()),
    )

    # Execute the order
    try:
        await live_trading._execute_order(order)
        live_trading.order_history.append(order)

        # Check if order actually succeeded
        if order.status == "failed":
            live_trading._save_state()
            return JSONResponse({
                "error": order.error or "Order execution failed",
                "order": order.to_dict()
            }, status_code=400)

        # Create a position to track this order
        from live_trading import LivePosition

        # Get market timing from active market
        market_start = market_info.get("market_start", int(time.time()))
        market_end = market_info.get("market_end", market_start + 900)  # 15 min default

        # Calculate fill details
        fill_price = order.fill_price or 0.50
        size = (request.amount_usd / fill_price) if fill_price > 0 else 0

        position = LivePosition(
            symbol=request.symbol,
            side=request.side,
            token_id=token_id,
            size=size,
            avg_entry_price=fill_price,
            cost_basis_usd=request.amount_usd,
            market_start=market_start,
            market_end=market_end,
            entry_orders=[order.id],
        )
        live_trading.open_positions.append(position)
        live_trading._save_state()

        logger.info(f"[LiveTrading] Created position: {request.symbol} {request.side} ${request.amount_usd}")

        return {
            "status": "ok",
            "order": order.to_dict(),
            "position": position.to_dict(),
            "message": f"Test order placed: {request.symbol} {request.side} ${request.amount_usd}"
        }
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "order": order.to_dict() if order else None
        }, status_code=500)


@app.post("/api/live-trading/confirm-order")
async def confirm_pending_order(
    request: OrderConfirmRequest,
    _: str = Depends(verify_api_key)
):
    """Confirm a pending order (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}

    # Find the pending order
    for order in live_trading.order_history:
        if order.id == request.order_id and order.status == "pending_confirmation":
            await live_trading._execute_order(order)
            return {"status": "ok", "order": order.to_dict()}

    return JSONResponse({"error": f"Order not found or not pending: {request.order_id}"}, status_code=404)


@app.post("/api/live-trading/reject-order")
async def reject_pending_order(
    request: OrderRejectRequest,
    _: str = Depends(verify_api_key)
):
    """Reject a pending order (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}

    # Find the pending order
    for order in live_trading.order_history:
        if order.id == request.order_id and order.status == "pending_confirmation":
            order.status = "rejected"
            order.error = request.reason
            live_trading._save_state()
            return {"status": "ok", "order": order.to_dict()}

    return JSONResponse({"error": f"Order not found or not pending: {request.order_id}"}, status_code=404)


class CancelOrderRequest(BaseModel):
    order_id: str


@app.post("/api/live-trading/cancel-order")
async def cancel_order(
    request: CancelOrderRequest,
    _: str = Depends(verify_api_key)
):
    """Cancel a specific order (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}

    success, message = live_trading.cancel_order(request.order_id)

    if success:
        return {"status": "ok", "message": message}
    else:
        return JSONResponse({"error": message}, status_code=400)


@app.post("/api/live-trading/cancel-all-orders")
async def cancel_all_orders(
    _: str = Depends(verify_api_key)
):
    """Cancel all open orders (requires API key)"""
    if not live_trading:
        return {"error": "Live trading not initialized"}

    success, message = live_trading.cancel_all_orders()

    if success:
        return {"status": "ok", "message": message}
    else:
        return JSONResponse({"error": message}, status_code=400)


# ============================================================================
# TRADE LEDGER API ENDPOINTS
# ============================================================================

@app.get("/api/trades")
async def get_trades(
    mode: Optional[str] = None,
    symbol: Optional[str] = None,
    from_date: Optional[int] = None,
    to_date: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Get recent trades from the ledger.

    Args:
        mode: Filter by "paper" or "live"
        symbol: Filter by symbol (BTC, ETH, etc.)
        from_date: Filter trades after this Unix timestamp
        to_date: Filter trades before this Unix timestamp
        limit: Max results (default 50)
        offset: Skip first N results
    """
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    trades = trade_ledger.get_trades(
        mode=mode,
        symbol=symbol,
        since=from_date,
        until=to_date,
        limit=limit,
        offset=offset,
    )
    return {"trades": [t.to_dict() for t in trades]}


@app.get("/api/trades/stats")
async def get_trade_stats(
    mode: Optional[str] = None,
    symbol: Optional[str] = None,
):
    """Get aggregate trade statistics"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    return trade_ledger.get_stats(mode=mode, symbol=symbol)


@app.get("/api/trades/daily")
async def get_daily_stats(
    days: int = 30,
    mode: Optional[str] = None,
):
    """Get daily P&L breakdown"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    return {"daily": trade_ledger.get_daily_stats(days=days, mode=mode)}


@app.get("/api/trades/by-symbol")
async def get_symbol_stats(
    mode: Optional[str] = None,
):
    """Get per-symbol breakdown"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    return {"symbols": trade_ledger.get_symbol_stats(mode=mode)}


@app.get("/api/trades/by-checkpoint")
async def get_checkpoint_stats(
    mode: Optional[str] = None,
):
    """Get per-checkpoint breakdown (to see which checkpoints are most profitable)"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    return {"checkpoints": trade_ledger.get_checkpoint_stats(mode=mode)}


@app.get("/api/trades/export/csv")
async def export_trades_csv(
    mode: Optional[str] = None,
    _: str = Depends(verify_api_key),
):
    """Export trades to CSV (requires API key)"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    import tempfile
    from fastapi.responses import FileResponse

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        filepath = f.name

    count = trade_ledger.export_csv(filepath, mode=mode)
    return FileResponse(
        filepath,
        media_type="text/csv",
        filename=f"trades_{mode or 'all'}.csv",
        headers={"X-Trade-Count": str(count)}
    )


# ============================================================================
# ACCOUNT & TRANSACTION ENDPOINTS
# ============================================================================

class CreateAccountRequest(BaseModel):
    name: str
    initial_balance: float = Field(0, ge=0)


class TransactionRequest(BaseModel):
    account: str
    amount: float = Field(..., gt=0)
    description: str
    notes: Optional[str] = None


@app.get("/api/accounts")
async def list_accounts():
    """List all trading accounts"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}
    return {"accounts": trade_ledger.list_accounts()}


@app.get("/api/accounts/{name}")
async def get_account(name: str):
    """Get account information"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    info = trade_ledger.get_account_info(name)
    if not info:
        return JSONResponse({"error": f"Account not found: {name}"}, status_code=404)

    return info


@app.post("/api/accounts")
async def create_account(
    request: CreateAccountRequest,
    _: str = Depends(verify_api_key),
):
    """Create a new trading account (requires API key)"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    success = trade_ledger.create_account(request.name, request.initial_balance)
    if not success:
        return JSONResponse({"error": f"Account already exists: {request.name}"}, status_code=400)

    return {"status": "ok", "account": trade_ledger.get_account_info(request.name)}


@app.get("/api/accounts/{name}/pnl")
async def get_account_pnl(name: str):
    """Get P&L summary for an account"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    return trade_ledger.get_pnl_summary(name)


@app.get("/api/accounts/{name}/transactions")
async def get_account_transactions(
    name: str,
    type: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """
    Get transactions for an account.

    Args:
        name: Account name
        type: Filter by transaction type (trade, credit, debit, fee, adjustment)
        from_date: Filter transactions after this date (ISO format or Unix timestamp)
        to_date: Filter transactions before this date (ISO format or Unix timestamp)
        limit: Max results (default 100)
        offset: Skip first N results

    Examples:
        /api/accounts/paper/transactions?from_date=2024-01-01&to_date=2024-01-31
        /api/accounts/paper/transactions?from_date=1704067200&to_date=1706745600
    """
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    txs = trade_ledger.get_transactions(
        account=name,
        type=type,
        since=from_date,
        until=to_date,
        limit=limit,
        offset=offset,
    )
    return {"transactions": [t.to_dict() for t in txs]}


@app.post("/api/accounts/{name}/credit")
async def credit_account(
    name: str,
    request: TransactionRequest,
    _: str = Depends(verify_api_key),
):
    """Credit (deposit) to an account (requires API key)"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    tx = trade_ledger.record_credit(
        account=name,
        amount=request.amount,
        description=request.description,
        notes=request.notes,
    )

    if not tx:
        return JSONResponse({"error": "Failed to record credit"}, status_code=500)

    return {"status": "ok", "transaction": tx.to_dict()}


@app.post("/api/accounts/{name}/debit")
async def debit_account(
    name: str,
    request: TransactionRequest,
    _: str = Depends(verify_api_key),
):
    """Debit (withdraw) from an account (requires API key)"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    # Check sufficient balance
    balance = trade_ledger.get_account_balance(name)
    if request.amount > balance:
        return JSONResponse(
            {"error": f"Insufficient balance: ${balance:.2f} < ${request.amount:.2f}"},
            status_code=400
        )

    tx = trade_ledger.record_debit(
        account=name,
        amount=request.amount,
        description=request.description,
        notes=request.notes,
    )

    if not tx:
        return JSONResponse({"error": "Failed to record debit"}, status_code=500)

    return {"status": "ok", "transaction": tx.to_dict()}


class AdjustmentRequest(BaseModel):
    account: str
    amount: float  # Can be positive or negative
    description: str
    notes: Optional[str] = None


@app.post("/api/accounts/{name}/adjustment")
async def adjust_account(
    name: str,
    request: AdjustmentRequest,
    _: str = Depends(verify_api_key),
):
    """Record a manual P&L adjustment (requires API key)"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    tx = trade_ledger.record_adjustment(
        account=name,
        amount=request.amount,
        description=request.description,
        notes=request.notes,
    )

    if not tx:
        return JSONResponse({"error": "Failed to record adjustment"}, status_code=500)

    return {"status": "ok", "transaction": tx.to_dict()}


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for real-time data.

    Clients receive:
    - trade: Real-time trades from Binance
    - candle: Updated 1-minute candles
    - orderbook: Order book updates
    - momentum: Momentum signals (every second)
    - whale_trade: New whale trades on Polymarket
    """
    await ws.accept()
    ws_clients.add(ws)
    print(f"[WS] Client connected. Total: {len(ws_clients)}")

    try:
        # Send initial data
        await ws.send_json({
            "type": "init",
            "whales": [
                {"name": k, "address": v["address"][:16] + "..."}
                for k, v in CRYPTO_WHALE_WALLETS.items()
                if v["address"].startswith("0x") and len(v["address"]) > 10
            ],
            "symbols": list(CryptoExchangeAPI.SYMBOLS.keys()),
            "paper_trading": paper_trading.get_account_summary() if paper_trading else None,
            "live_trading": live_trading.get_status() if live_trading else None,
        })

        # Keep connection alive
        while True:
            try:
                # Wait for client messages (heartbeat or commands)
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                msg = json.loads(data)

                # Handle client commands
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})

                elif msg.get("type") == "get_candles":
                    symbol = msg.get("symbol", "BTCUSDT")
                    candles = binance_feed.get_recent_candles(symbol.lower(), 100)
                    await ws.send_json({
                        "type": "candles_snapshot",
                        "symbol": symbol,
                        "candles": candles,
                    })

            except asyncio.TimeoutError:
                # Send keepalive ping
                await ws.send_json({"type": "ping"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        ws_clients.discard(ws)
        print(f"[WS] Client disconnected. Total: {len(ws_clients)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the server"""
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
