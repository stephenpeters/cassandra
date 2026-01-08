"""
Tests for REST API Endpoints (server.py).

Tests cover:
- Trading API endpoints
- Live trading API endpoints
- WebSocket message handling
- Error handling
"""
import pytest
import secrets
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Test API key for authentication
TEST_API_KEY = secrets.token_urlsafe(32)


# We need to mock the global state before importing server
@pytest.fixture(autouse=True)
def mock_globals():
    """Mock global state for server tests."""
    with patch.dict(os.environ, {
        "POLYMARKET_PRIVATE_KEY": "",
        "API_KEY": TEST_API_KEY,  # Set test API key
    }):
        yield


@pytest.fixture
def auth_headers():
    """Return auth headers for protected endpoints."""
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def mock_paper_trading():
    """Create mock paper trading engine."""
    mock = MagicMock()
    mock.trading_mode = "paper"
    mock.config = MagicMock()
    mock.config.enabled = True
    mock.config.to_dict = MagicMock(return_value={"enabled": True})
    mock.account = MagicMock()
    mock.account.balance = 1000
    mock.account.positions = []
    mock.account.trade_history = []
    mock.get_account_summary = MagicMock(return_value={
        "balance": 1000,
        "total_pnl": 0,
        "positions": [],
        "recent_trades": [],
        "win_rate": 0.0,
    })
    mock.get_positions = MagicMock(return_value=[])
    mock.get_config = MagicMock(return_value={"enabled": True})
    return mock


@pytest.fixture
def mock_live_trading():
    """Create mock live trading engine."""
    mock = MagicMock()
    mock.config = MagicMock()
    mock.config.mode = MagicMock()
    mock.config.mode.value = "paper"
    mock.config.enabled_assets = ["BTC", "ETH"]
    mock.config.to_dict = MagicMock(return_value={"mode": "paper"})
    mock.kill_switch_active = False
    mock.circuit_breaker = MagicMock()
    mock.circuit_breaker.triggered = False
    mock.circuit_breaker.reason = ""
    mock.circuit_breaker.to_dict = MagicMock(return_value={"triggered": False})
    mock.open_positions = []
    mock.order_history = []
    mock.clob_client = None
    mock.get_status = MagicMock(return_value={
        "mode": "paper",
        "kill_switch_active": False,
        "circuit_breaker": {"triggered": False},
        "open_positions": 0,
        "enabled_assets": ["BTC", "ETH"],
    })
    mock.get_positions = MagicMock(return_value=[])
    mock.get_order_history = MagicMock(return_value=[])
    mock.ensure_clob_initialized = MagicMock(return_value=(False, "Not configured"))
    mock.check_token_allowances = MagicMock(return_value=(False, "Not configured", {}))
    mock.get_wallet_balance = MagicMock(return_value={"error": "Not configured"})
    return mock


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, mock_paper_trading, mock_live_trading):
        """Test root endpoint returns status."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/")
            assert response.status_code == 200
            # Actual root returns {"status": "ok", "timestamp": ...}
            assert "status" in response.json()

    def test_health_endpoint(self, mock_paper_trading, mock_live_trading):
        """Test health endpoint returns status."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] in ["healthy", "degraded", "unhealthy"]


class TestPaperTradingAPIEndpoints:
    """Tests for paper trading API endpoints."""

    def test_get_paper_trading_account(self, mock_paper_trading, mock_live_trading):
        """Test GET /api/paper-trading/account."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/api/paper-trading/account")
            assert response.status_code == 200
            data = response.json()
            assert "balance" in data

    def test_get_paper_trading_positions(self, mock_paper_trading, mock_live_trading):
        """Test GET /api/paper-trading/positions."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/api/paper-trading/positions")
            assert response.status_code == 200
            assert "positions" in response.json()

    def test_get_paper_trading_config(self, mock_paper_trading, mock_live_trading):
        """Test GET /api/paper-trading/config."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/api/paper-trading/config")
            assert response.status_code == 200


class TestLiveTradingAPIEndpoints:
    """Tests for live trading API endpoints - require X-API-Key header."""

    def test_get_live_trading_status(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test GET /api/live-trading/status with auth."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/api/live-trading/status", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert "mode" in data

    def test_get_live_trading_status_no_auth(self, mock_paper_trading, mock_live_trading):
        """Test GET /api/live-trading/status without auth returns 403."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/api/live-trading/status")
            assert response.status_code == 403

    def test_get_live_trading_config(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test GET /api/live-trading/config with auth."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/api/live-trading/config", headers=auth_headers)
            assert response.status_code == 200

    def test_get_live_trading_positions(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test GET /api/live-trading/positions with auth."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/api/live-trading/positions", headers=auth_headers)
            assert response.status_code == 200
            assert "positions" in response.json()

    def test_get_live_trading_orders(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test GET /api/live-trading/orders with auth."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/api/live-trading/orders", headers=auth_headers)
            assert response.status_code == 200
            assert "orders" in response.json()

    def test_get_circuit_breaker_status(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test GET /api/live-trading/circuit-breaker with auth."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.get("/api/live-trading/circuit-breaker", headers=auth_headers)
            assert response.status_code == 200

    def test_post_kill_switch_activate(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test POST /api/live-trading/kill-switch to activate with auth."""
        # activate_kill_switch is async, so use AsyncMock
        mock_live_trading.activate_kill_switch = AsyncMock()
        mock_live_trading.deactivate_kill_switch = AsyncMock()

        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.post(
                "/api/live-trading/kill-switch",
                json={"activate": True, "reason": "Test"},
                headers=auth_headers
            )
            assert response.status_code == 200

    def test_post_mode_change_to_paper(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test POST /api/live-trading/mode to switch to paper with auth."""
        mock_live_trading.set_mode = MagicMock()

        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.post(
                "/api/live-trading/mode",
                json={"mode": "paper"},
                headers=auth_headers
            )
            assert response.status_code == 200

    def test_post_mode_change_to_live_without_clob(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test POST /api/live-trading/mode fails when CLOB unavailable."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY), \
             patch("server.CLOB_AVAILABLE", False):
            from server import app
            client = TestClient(app)

            response = client.post(
                "/api/live-trading/mode",
                json={"mode": "live"},
                headers=auth_headers
            )
            # Should fail due to missing CLOB
            assert response.status_code == 400

    def test_post_enabled_assets(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test POST /api/live-trading/enabled-assets with auth."""
        mock_live_trading._save_state = MagicMock()

        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.post(
                "/api/live-trading/enabled-assets",
                json={"assets": ["BTC", "ETH", "SOL"]},
                headers=auth_headers
            )
            assert response.status_code == 200

    def test_post_enabled_assets_invalid(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test POST /api/live-trading/enabled-assets with invalid asset."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.post(
                "/api/live-trading/enabled-assets",
                json={"assets": ["INVALID_COIN"]},
                headers=auth_headers
            )
            assert response.status_code == 400


class TestMarketDataEndpoints:
    """Tests for market data endpoints."""

    def test_get_candles(self, mock_paper_trading, mock_live_trading):
        """Test GET /api/candles/{symbol}."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY), \
             patch("server.binance_feed") as mock_feed:
            mock_feed.candles = {"btcusdt": []}
            from server import app
            client = TestClient(app)

            response = client.get("/api/candles/BTC")
            assert response.status_code == 200

    def test_get_momentum(self, mock_paper_trading, mock_live_trading):
        """Test GET /api/momentum."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY), \
             patch("server.momentum_calc") as mock_calc:
            mock_calc.get_all_signals = MagicMock(return_value={})
            from server import app
            client = TestClient(app)

            response = client.get("/api/momentum")
            assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""

    def test_not_initialized_error(self, mock_paper_trading):
        """Test endpoints handle not-initialized state."""
        with patch("server.paper_trading", None), \
             patch("server.live_trading", None), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            # Use correct endpoint path
            response = client.get("/api/paper-trading/account")
            # Should return 500 or error response, not crash
            assert response.status_code in [200, 500]

    def test_invalid_json_body(self, mock_paper_trading, mock_live_trading, auth_headers):
        """Test endpoints handle invalid JSON."""
        with patch("server.paper_trading", mock_paper_trading), \
             patch("server.live_trading", mock_live_trading), \
             patch("server.API_KEY", TEST_API_KEY):
            from server import app
            client = TestClient(app)

            response = client.post(
                "/api/live-trading/mode",
                content="not valid json",
                headers={**auth_headers, "Content-Type": "application/json"}
            )
            assert response.status_code == 422  # Validation error
