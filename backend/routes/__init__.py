"""
Route modules for the Polymarket Whale Tracker API.

This package organizes API endpoints into logical groups:
- whales: Whale tracking and analysis
- trading: Paper and live trading
- history: Historical data and analytics
- markets: Market data and configuration
- accounts: Account management
"""

from .whales import router as whales_router
from .trading import router as trading_router
from .history import router as history_router
from .markets import router as markets_router
from .accounts import router as accounts_router

__all__ = [
    "whales_router",
    "trading_router",
    "history_router",
    "markets_router",
    "accounts_router",
]
