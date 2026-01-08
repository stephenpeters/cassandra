"""
Shared dependencies for route modules.

This module provides access to global state and shared utilities.
"""

import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# =============================================================================
# SECURITY: API Key Authentication
# =============================================================================

ENV = os.getenv("ENV", "development").lower()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    if ENV == "production":
        import sys
        print("[Security] FATAL: API_KEY environment variable not set.")
        sys.exit(1)
    else:
        API_KEY = secrets.token_urlsafe(32)
        print(f"[Security] Generated temporary key: {API_KEY}")

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
# GLOBAL STATE ACCESSORS
# =============================================================================

# These will be set by server.py at startup
_state = {
    "whale_tracker": None,
    "whale_detector": None,
    "whale_ws_detector": None,
    "binance_feed": None,
    "polymarket_feed": None,
    "momentum_calc": None,
    "paper_trading": None,
    "live_trading": None,
    "trade_ledger": None,
    "market_data_store": None,
}


def set_state(key: str, value):
    """Set a global state value (called from server.py)"""
    _state[key] = value


def get_whale_tracker():
    return _state["whale_tracker"]


def get_whale_detector():
    return _state["whale_detector"]


def get_whale_ws_detector():
    return _state["whale_ws_detector"]


def get_binance_feed():
    return _state["binance_feed"]


def get_polymarket_feed():
    return _state["polymarket_feed"]


def get_momentum_calc():
    return _state["momentum_calc"]


def get_paper_trading():
    return _state["paper_trading"]


def get_live_trading():
    return _state["live_trading"]


def get_trade_ledger():
    return _state["trade_ledger"]


def get_market_data_store():
    return _state["market_data_store"]
