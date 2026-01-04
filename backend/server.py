#!/usr/bin/env python3
"""
WebSocket server for the Polymarket Whale Tracker.
Provides real-time data feeds to the frontend.
"""

import asyncio
import json
import signal
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

# =============================================================================
# SECURITY: API Key Authentication
# =============================================================================

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    # Generate a random key if not set (will be logged on startup)
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
from paper_trading import PaperTradingEngine
from live_trading import (
    LiveTradingEngine,
    LiveTradingConfig,
    TradingMode,
    CLOB_AVAILABLE,
)
from trade_ledger import TradeLedger

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
paper_trading: PaperTradingEngine = None
live_trading: LiveTradingEngine = None
trade_ledger: TradeLedger = None

# WebSocket clients
ws_clients: Set[WebSocket] = set()

# Background tasks
background_tasks: list[asyncio.Task] = []


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize data feeds on startup"""
    global binance_feed, whale_tracker, momentum_calc, polymarket_feed, paper_trading, trade_ledger

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

    # Initialize paper trading engine
    paper_trading = PaperTradingEngine(data_dir=".", ledger=trade_ledger)
    paper_trading.on_signal = lambda sig: asyncio.create_task(
        broadcast({"type": "paper_signal", "data": sig.to_dict()})
    )
    paper_trading.on_trade = lambda trade: asyncio.create_task(
        broadcast({"type": "paper_trade", "data": trade.to_dict()})
    )
    paper_trading.on_position_open = lambda pos: asyncio.create_task(
        broadcast({"type": "paper_position", "data": pos.to_dict()})
    )

    # Initialize live trading engine
    # Private key loaded from environment for security
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    live_config = LiveTradingConfig.from_env()
    live_trading = LiveTradingEngine(
        private_key=private_key,
        config=live_config,
        data_dir=".",
        ledger=trade_ledger,
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

    print(f"[Server] Live trading initialized in {live_trading.config.mode.value} mode")
    if not CLOB_AVAILABLE:
        print("[Server] WARNING: py-clob-client not installed - live trading disabled")

    # Start background tasks
    background_tasks.append(asyncio.create_task(binance_feed.connect()))
    background_tasks.append(asyncio.create_task(whale_tracker.start(
        poll_interval=DEFAULT_CONFIG.whale_poll_interval_sec
    )))
    background_tasks.append(asyncio.create_task(polymarket_feed.start(poll_interval=2.0)))
    background_tasks.append(asyncio.create_task(broadcast_momentum_loop()))
    background_tasks.append(asyncio.create_task(broadcast_markets_loop()))
    background_tasks.append(asyncio.create_task(paper_trading_loop()))

    print("[Server] Started all data feeds")


@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown"""
    if binance_feed:
        await binance_feed.stop()
    if whale_tracker:
        whale_tracker.stop()
    if polymarket_feed:
        polymarket_feed.stop()

    for task in background_tasks:
        task.cancel()

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

    for ws in ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.add(ws)

    # Clean up disconnected clients
    for ws in disconnected:
        ws_clients.discard(ws)


async def broadcast_momentum_loop():
    """Periodically broadcast momentum signals"""
    while True:
        await asyncio.sleep(1)  # Every second

        if momentum_calc and ws_clients:
            signals = momentum_calc.get_all_signals()
            await broadcast({
                "type": "momentum",
                "data": signals,
            })


async def broadcast_markets_loop():
    """Periodically broadcast active 15-min markets and timing"""
    while True:
        await asyncio.sleep(2)  # Every 2 seconds

        if polymarket_feed and ws_clients:
            await broadcast({
                "type": "markets_15m",
                "data": {
                    "active": polymarket_feed.get_active_markets(),
                    "timing": polymarket_feed.get_next_market_time(),
                    "trades": polymarket_feed.get_recent_trades(limit=20),
                },
            })


# Track which market windows we've recorded open prices for
_recorded_opens: set[str] = set()  # "SYMBOL_market_start"


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
    global _recorded_opens

    while True:
        await asyncio.sleep(1)  # Check every second for latency opportunities

        if not paper_trading or not paper_trading.config.enabled:
            continue

        if not polymarket_feed or not binance_feed:
            continue

        now = int(datetime.now().timestamp())

        # Get active 15-minute markets from Polymarket
        active_markets = polymarket_feed.get_active_markets()
        if not active_markets:
            continue

        for symbol, market_data in active_markets.items():
            market_start = market_data.get("start_time", 0)
            market_end = market_data.get("end_time", 0)
            is_active = market_data.get("is_active", True)

            if not market_start or not market_end or not is_active:
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
                paper_trading.record_window_open(symbol, market_start, current_price)
                _recorded_opens.add(open_key)
                print(f"[Latency] {symbol} window open recorded: ${current_price:.2f}")

            # Clean up old keys (windows that ended)
            _recorded_opens = {k for k in _recorded_opens if int(k.split("_")[1]) > now - 1800}

            # Get current Polymarket UP price
            polymarket_up_price = market_data.get("price", 0.5)

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
                      f"PM UP: {polymarket_up_price:.1%}")

                # Broadcast the signal to all clients
                await broadcast({
                    "type": "paper_signal",
                    "data": signal.to_dict(),
                })

        # Check for market resolutions (positions that need settling)
        for position in list(paper_trading.account.positions):
            if now >= position.market_end:
                # Get Binance prices for resolution
                binance_sym = f"{position.symbol}usdt"
                candles = binance_feed.candles.get(binance_sym, [])

                if len(candles) >= 2:
                    # Get the open price we recorded
                    open_key = f"{position.symbol}_{position.market_start}"
                    open_price = paper_trading._binance_opens.get(open_key)

                    # Fallback: try to find from candles
                    if not open_price:
                        for c in candles:
                            candle_time = c.timestamp // 1000
                            if candle_time >= position.market_start:
                                open_price = c.open
                                break

                    # Get close price
                    close_price = candles[-1].close
                    for c in candles:
                        candle_time = c.timestamp // 1000
                        if candle_time >= position.market_end:
                            close_price = c.close
                            break

                    if open_price and close_price:
                        resolution = "UP" if close_price > open_price else "DOWN"
                        print(f"[Latency] {position.symbol} RESOLVED: {resolution} | "
                              f"Open: ${open_price:.2f} -> Close: ${close_price:.2f} | "
                              f"Position: {position.side}")

                        paper_trading.resolve_market(
                            symbol=position.symbol,
                            market_start=position.market_start,
                            market_end=position.market_end,
                            binance_open=open_price,
                            binance_close=close_price,
                        )

        # Broadcast paper trading state periodically
        if ws_clients and paper_trading:
            await broadcast({
                "type": "paper_account",
                "data": paper_trading.get_account_summary(),
            })


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

    # WebSocket clients
    status["components"]["websocket_clients"] = len(ws_clients)

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


@app.get("/api/markets-15m")
async def get_markets_15m():
    """Get active 15-minute markets and timing"""
    if not polymarket_feed:
        return {"active": {}, "timing": {}, "trades": []}

    return {
        "active": polymarket_feed.get_active_markets(),
        "timing": polymarket_feed.get_next_market_time(),
        "trades": polymarket_feed.get_recent_trades(limit=50),
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

    # Safety checks for live mode
    if request.mode == "live":
        if not CLOB_AVAILABLE:
            return JSONResponse({"error": "py-clob-client not installed - cannot enable live mode"}, status_code=400)
        if not os.getenv("POLYMARKET_PRIVATE_KEY"):
            return JSONResponse({"error": "POLYMARKET_PRIVATE_KEY not set - cannot enable live mode"}, status_code=400)

        # Check token allowances are set (per Polymarket best practices)
        is_approved, message, _ = live_trading.check_token_allowances()
        if not is_approved:
            return JSONResponse({
                "error": f"Token allowances not set: {message}. "
                         "Call POST /api/live-trading/allowances first."
            }, status_code=400)

    live_trading.set_mode(request.mode)
    return {"status": "ok", "mode": request.mode}


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

    success, message = live_trading.set_allowances()
    if not success:
        return JSONResponse({"error": message}, status_code=400)

    return {"status": "ok", "message": message}


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


# ============================================================================
# TRADE LEDGER API ENDPOINTS
# ============================================================================

@app.get("/api/trades")
async def get_trades(
    mode: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Get recent trades from the ledger"""
    if not trade_ledger:
        return {"error": "Trade ledger not initialized"}

    trades = trade_ledger.get_trades(
        mode=mode,
        symbol=symbol,
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
