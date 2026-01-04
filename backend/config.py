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
    # =========================================================================
    # TOP CRYPTO 15-MIN TRADERS (by recent volume)
    # =========================================================================

    "gabagool22": {
        "address": "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d",
        "strategy": "News sniping, momentum. Fast entry <1min on strong momo, 3-5min on weaker",
        "focus": ["crypto", "politics"],
        "notes": "Top performer on crypto up/down markets",
    },

    "Account88888": {
        "address": "0x7f69983eb28245bba0d5083502a78744a8f66162",
        "strategy": "Sequential entry arbitrage, waits 5-7min for confirmation",
        "focus": ["crypto"],
        "notes": "$324K profit in 25 days, suspected Jane Street quant",
    },

    "PurpleThunderBicycleMountain": {
        "address": "0x589222a5124a96765443b97a3498d89ffd824ad2",
        "strategy": "Multi-asset crypto 15-min",
        "focus": ["crypto"],
        "notes": "High volume: SOL, XRP, BTC markets, 95 trades",
    },

    "updateupdate": {
        "address": "0xd0d6053c3c37e727402d84c14069780d360993aa",
        "strategy": "BTC-focused 15-min",
        "focus": ["crypto"],
        "notes": "35 trades, BTC specialist",
    },

    "soratin": {
        "address": "0x60aaaafa018e46cee26ead8afc8e5506d53b0df0",
        "strategy": "Multi-asset crypto 15-min",
        "focus": ["crypto"],
        "notes": "ETH, BTC, XRP - 45 trades",
    },

    "BoshBashBish": {
        "address": "0x29bc82f761749e67fa00d62896bc6855097b683c",
        "strategy": "XRP and BTC 15-min",
        "focus": ["crypto"],
        "notes": "24 trades",
    },

    "distinct-baguette": {
        "address": "0xe00740bce98a594e265f3c07ff0b55a55ad47e4f",
        "strategy": "Cross-market arbitrage",
        "focus": ["crypto"],
        "notes": "$50K+ arb profits, SOL/XRP focus",
    },

    "jdkfbsbdskjbfjksq": {
        "address": "0xb1198968d144997367fe23c14f3730348075680b",
        "strategy": "High-frequency BTC 15-min",
        "focus": ["crypto"],
        "notes": "128 trades - very active",
    },

    "b1gGambler": {
        "address": "0x8ac1eaed0399f8332f05d4e6e3c0db1eb8cf15fb",
        "strategy": "BTC 15-min",
        "focus": ["crypto"],
        "notes": "34 trades",
    },

    "ProfitGoblin": {
        "address": "0x2a97fb7142ef565b4b1bc823c4a62f5b1fe07eaf",
        "strategy": "BTC 15-min",
        "focus": ["crypto"],
        "notes": "33 trades",
    },

    # =========================================================================
    # POLITICAL WHALES (Théo cluster)
    # =========================================================================

    "Fredi9999": {
        "address": "0x1f2d",  # Partial - needs full address
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
