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
    Polymarket15MinFeed,
    fetch_historical_candles,
    Trade,
    OHLCV,
    OrderBook,
    WhaleTrade,
)
from paper_trading import PaperTradingEngine

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
polymarket_feed: Polymarket15MinFeed = None
paper_trading: PaperTradingEngine = None

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
    global binance_feed, whale_tracker, momentum_calc, polymarket_feed, paper_trading

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

    # Initialize paper trading engine
    paper_trading = PaperTradingEngine(data_dir=".")
    paper_trading.on_signal = lambda sig: asyncio.create_task(
        broadcast({"type": "paper_signal", "data": sig.to_dict()})
    )
    paper_trading.on_trade = lambda trade: asyncio.create_task(
        broadcast({"type": "paper_trade", "data": trade.to_dict()})
    )
    paper_trading.on_position_open = lambda pos: asyncio.create_task(
        broadcast({"type": "paper_position", "data": pos.to_dict()})
    )

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
