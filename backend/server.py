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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import (
    CryptoExchangeAPI,
    CRYPTO_WHALE_WALLETS,
    DEFAULT_CONFIG,
)
from data_feeds import (
    BinanceFeed,
    WhaleTracker,
    MomentumCalculator,
    fetch_historical_candles,
    Trade,
    OHLCV,
    OrderBook,
    WhaleTrade,
)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Polymarket Whale Tracker API",
    description="Real-time crypto prices, whale trades, and momentum signals",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Data feeds
binance_feed: BinanceFeed = None
whale_tracker: WhaleTracker = None
momentum_calc: MomentumCalculator = None

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
    global binance_feed, whale_tracker, momentum_calc

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

    # Initialize momentum calculator
    momentum_calc = MomentumCalculator(binance_feed)

    # Start background tasks
    background_tasks.append(asyncio.create_task(binance_feed.connect()))
    background_tasks.append(asyncio.create_task(whale_tracker.start(
        poll_interval=DEFAULT_CONFIG.whale_poll_interval_sec
    )))
    background_tasks.append(asyncio.create_task(broadcast_momentum_loop()))

    print("[Server] Started all data feeds")


@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown"""
    if binance_feed:
        await binance_feed.stop()
    if whale_tracker:
        whale_tracker.stop()

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


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


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
