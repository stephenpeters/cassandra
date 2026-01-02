"""
Configuration for the Polymarket Whale Tracker
Contains whale wallets, API endpoints, and trading parameters.
"""

from dataclasses import dataclass
from typing import Optional

# ============================================================================
# WHALE WALLET REGISTRY
# ============================================================================

# Known whale wallets - sources from research and Polymarket leaderboard
WHALE_WALLETS = {
    # Verified crypto market traders
    "gabagool22": {
        "address": "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d",
        "strategy": "News sniping, asymmetric scalping",
        "focus": ["crypto", "politics"],
        "notes": "Top performer on crypto up/down markets",
    },

    # French whale (Théo) - 11 connected accounts
    # Made $85M on 2024 election, $30M in total bets
    "Fredi9999": {
        "address": "0x1f2d",  # Partial - full address needed from chain
        "strategy": "Political high-stakes",
        "focus": ["politics"],
        "notes": "Part of Théo cluster, $22M+ lifetime earnings",
    },
    "Theo4": {
        "address": "0x5668",  # Partial
        "strategy": "High-frequency political",
        "focus": ["politics"],
        "notes": "$12M in 3 days on Trump election bets",
    },
    "PrincessCaro": {
        "address": "0x8119",  # Partial
        "strategy": "Political",
        "focus": ["politics"],
        "notes": "Part of Théo cluster",
    },

    # Account88888 / JaneStreetIndia - 15-min crypto specialist
    # $360K+ profit, 23/25 profitable days
    "Account88888": {
        "address": "0x",  # Need to extract from Polymarket
        "strategy": "Sequential entry arbitrage on 15-min crypto",
        "focus": ["crypto"],
        "notes": "Suspected Jane Street quant bot, 92% win rate",
    },

    # Add more whales as discovered
    # Use PolyTrack, Polymarket Analytics, or on-chain analysis
}

# Crypto-focused whales only (for the tracker)
CRYPTO_WHALE_WALLETS = {
    k: v for k, v in WHALE_WALLETS.items()
    if "crypto" in v.get("focus", [])
}

# ============================================================================
# API ENDPOINTS
# ============================================================================

class PolymarketAPI:
    # REST APIs
    DATA_API = "https://data-api.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    # WebSocket endpoints
    WS_MARKET = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    WS_USER = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    # Useful endpoints
    TRADES = f"{DATA_API}/trades"
    MARKETS = f"{GAMMA_API}/markets"
    EVENTS = f"{GAMMA_API}/events"
    ORDERBOOK = f"{CLOB_API}/book"
    PRICES = f"{CLOB_API}/prices"


class CryptoExchangeAPI:
    """Real-time crypto price feeds for momentum detection"""

    # WebSocket streams
    BINANCE_WS = "wss://stream.binance.com:9443/ws"
    BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"
    COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
    BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"

    # REST APIs (for OHLCV)
    BINANCE_REST = "https://api.binance.com/api/v3"
    COINBASE_REST = "https://api.exchange.coinbase.com"

    # Symbols to track
    SYMBOLS = {
        "BTC": {"binance": "btcusdt", "coinbase": "BTC-USD"},
        "ETH": {"binance": "ethusdt", "coinbase": "ETH-USD"},
        "SOL": {"binance": "solusdt", "coinbase": "SOL-USD"},
        "XRP": {"binance": "xrpusdt", "coinbase": "XRP-USD"},
        "DOGE": {"binance": "dogeusdt", "coinbase": "DOGE-USD"},
    }


# ============================================================================
# TRADING PARAMETERS
# ============================================================================

@dataclass
class TradingConfig:
    # Polling/update intervals
    whale_poll_interval_sec: float = 3.0
    price_update_interval_ms: int = 100
    orderbook_update_interval_ms: int = 500

    # Whale copy trading
    min_whale_trade_size: float = 100.0  # Minimum USD to trigger alert
    significant_trade_size: float = 1000.0  # Highlighted in UI
    large_trade_size: float = 5000.0  # Major alert

    # Momentum detection thresholds
    volume_delta_threshold: float = 5_000_000  # $5M imbalance
    liquidation_cascade_threshold: float = 1_000_000  # $1M in 500ms
    price_lead_threshold: float = 0.0005  # 0.05% cross-exchange divergence

    # Risk parameters
    max_position_pct: float = 0.05  # 5% of portfolio per trade
    max_daily_loss_pct: float = 0.05  # Stop at 5% daily loss


DEFAULT_CONFIG = TradingConfig()


# ============================================================================
# VPN CONFIGURATION (for Singapore/geo-restricted access)
# ============================================================================

class VPNConfig:
    """
    Options for programmatic VPN control from Singapore:

    1. ExpressVPN CLI (recommended if you have subscription):
       - Install: brew install expressvpn
       - Connect: expressvpn connect "USA - New York"
       - Disconnect: expressvpn disconnect
       - Python wrapper: pip install expressvpn-python

    2. Docker with ExpressVPN (most automatable):
       - Image: superdanio/expressvpn
       - API endpoint: http://localhost:5000/api/status
       - Route other containers through VPN container

    3. Mullvad VPN (alternative with better API):
       - Has official CLI with better automation support
       - mullvad connect/disconnect

    4. Proxy alternatives (no VPN needed):
       - Use a US-based proxy server
       - Route only Polymarket traffic through proxy
       - Cheaper and more reliable for API-only access
    """

    # ExpressVPN CLI commands
    EXPRESSVPN_CONNECT = "expressvpn connect 'USA - New York'"
    EXPRESSVPN_DISCONNECT = "expressvpn disconnect"
    EXPRESSVPN_STATUS = "expressvpn status"

    # Docker API (if using superdanio/expressvpn)
    DOCKER_VPN_API = "http://localhost:5000/api"

    @staticmethod
    def check_vpn_needed() -> bool:
        """Check if VPN is needed by testing Polymarket API"""
        import requests
        try:
            resp = requests.get(
                f"{PolymarketAPI.DATA_API}/trades?limit=1",
                timeout=5
            )
            return resp.status_code != 200
        except:
            return True


# ============================================================================
# MARKET FILTERING
# ============================================================================

# Keywords for 15-minute crypto markets
CRYPTO_15M_KEYWORDS = [
    "up or down",
    "15m",
    "15-min",
    "15 minute",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "solana",
    "sol",
    "xrp",
    "doge",
]

# Slugs for 15-minute markets (more reliable filtering)
CRYPTO_15M_SLUG_PATTERNS = [
    "btc-updown-15m",
    "eth-updown-15m",
    "sol-updown-15m",
    "xrp-updown-15m",
    "doge-updown-15m",
]

def is_crypto_15m_market(market: dict) -> bool:
    """Check if market is a 15-minute crypto up/down market"""
    slug = market.get("slug", "").lower()
    title = market.get("title", "").lower()

    # Check slug patterns first (most reliable)
    for pattern in CRYPTO_15M_SLUG_PATTERNS:
        if pattern in slug:
            return True

    # Fallback to keyword matching
    return any(kw in title for kw in CRYPTO_15M_KEYWORDS)
