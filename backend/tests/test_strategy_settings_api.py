"""
Integration tests for Strategy Settings API.

These tests ensure:
1. Strategy enable/disable works and persists
2. Strategy markets can be updated
3. Strategy settings can be updated
4. Config file is properly saved after changes
5. API returns correct responses and status codes
6. Invalid requests are properly rejected

Prevents regressions like:
- Frontend calling wrong URL (relative vs absolute)
- Wrong HTTP method (query params vs JSON body)
- Config not saving to file
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config = {
        "strategies": {
            "sniper": {
                "enabled": True,
                "description": "Test sniper",
                "markets": ["BTC"],
                "settings": {
                    "min_price": 0.75,
                    "max_price": 0.98,
                    "min_elapsed_sec": 600
                }
            },
            "copy_trading": {
                "enabled": True,
                "description": "Test copy trading",
                "markets": ["BTC", "ETH"],
                "settings": {
                    "position_multiplier": 0.5
                },
                "traders": []
            },
            "latency_gap": {
                "enabled": False,
                "description": "Test latency gap",
                "markets": ["BTC"],
                "settings": {
                    "min_edge_pct": 5.0
                }
            },
            "dip_arb": {
                "enabled": False,
                "description": "Test dip arb",
                "markets": ["BTC"],
                "settings": {
                    "dip_threshold": 0.15
                }
            },
            "latency_arb": {
                "enabled": False,
                "description": "Test latency arb",
                "markets": ["BTC"],
                "settings": {
                    "min_move_pct": 0.3
                }
            }
        },
        "global_settings": {
            "max_daily_loss_pct": 10,
            "max_daily_trades": 20
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def api_key():
    """Test API key."""
    return "test_api_key_12345"


@pytest.fixture
def auth_headers(api_key):
    """Auth headers for API requests."""
    return {"X-API-Key": api_key}


@pytest.fixture
def test_client(temp_config_file, api_key):
    """Create a test client with mocked dependencies."""
    # Patch environment and strategy manager before importing server
    with patch.dict(os.environ, {"API_KEY": api_key, "ENVIRONMENT": "test"}):
        with patch('strategy_manager._strategy_manager', None):
            # Import fresh to pick up patches
            import importlib
            import strategy_manager
            importlib.reload(strategy_manager)

            # Create strategy manager with temp config
            strategy_manager._strategy_manager = strategy_manager.StrategyManager(temp_config_file)

            # Now import server (it will use our patched strategy manager)
            import server

            # Mock the data feeds and other dependencies
            server.binance_feed = MagicMock()
            server.polymarket_feed = MagicMock()
            server.momentum_calc = MagicMock()
            server.whale_tracker = MagicMock()
            server.paper_trading = MagicMock()
            server.live_trading = MagicMock()
            server.sniper_strategy = MagicMock()
            server.dip_arb_strategy = MagicMock()
            server.latency_arb_strategy = MagicMock()

            client = TestClient(server.app)
            yield client, temp_config_file


class TestStrategyEnableEndpoint:
    """Test POST /api/strategies/{name}/enable"""

    def test_enable_strategy(self, test_client, auth_headers):
        """Can enable a disabled strategy."""
        client, config_path = test_client

        # Disable first
        response = client.post(
            "/api/strategies/latency_gap/enable?enabled=false",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["enabled"] == False

        # Enable
        response = client.post(
            "/api/strategies/latency_gap/enable?enabled=true",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["strategy"] == "latency_gap"
        assert data["enabled"] == True

    def test_disable_strategy(self, test_client, auth_headers):
        """Can disable an enabled strategy."""
        client, config_path = test_client

        response = client.post(
            "/api/strategies/sniper/enable?enabled=false",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] == False

    def test_enable_persists_to_file(self, test_client, auth_headers):
        """Enable/disable changes are saved to config file."""
        client, config_path = test_client

        # Disable sniper
        response = client.post(
            "/api/strategies/sniper/enable?enabled=false",
            headers=auth_headers
        )
        assert response.status_code == 200

        # Read config file and verify
        with open(config_path) as f:
            config = json.load(f)

        assert config["strategies"]["sniper"]["enabled"] == False

    def test_enable_invalid_strategy_returns_404(self, test_client, auth_headers):
        """Enabling non-existent strategy returns 404."""
        client, _ = test_client

        response = client.post(
            "/api/strategies/nonexistent/enable?enabled=true",
            headers=auth_headers
        )
        assert response.status_code == 404

    def test_enable_requires_auth(self, test_client):
        """Enable endpoint requires API key."""
        client, _ = test_client

        response = client.post("/api/strategies/sniper/enable?enabled=true")
        assert response.status_code in [401, 403]  # Either unauthorized or forbidden


class TestStrategyMarketsEndpoint:
    """Test POST /api/strategies/{name}/markets"""

    def test_set_markets_with_json_body(self, test_client, auth_headers):
        """Markets endpoint accepts JSON body (not query params)."""
        client, config_path = test_client

        response = client.post(
            "/api/strategies/sniper/markets",
            headers={**auth_headers, "Content-Type": "application/json"},
            json=["BTC", "ETH", "SOL"]
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert set(data["markets"]) == {"BTC", "ETH", "SOL"}

    def test_set_markets_persists_to_file(self, test_client, auth_headers):
        """Market changes are saved to config file."""
        client, config_path = test_client

        response = client.post(
            "/api/strategies/sniper/markets",
            headers={**auth_headers, "Content-Type": "application/json"},
            json=["ETH", "SOL"]
        )
        assert response.status_code == 200

        # Read config file and verify
        with open(config_path) as f:
            config = json.load(f)

        assert set(config["strategies"]["sniper"]["markets"]) == {"ETH", "SOL"}

    def test_set_markets_empty_list(self, test_client, auth_headers):
        """Can set empty markets list."""
        client, _ = test_client

        response = client.post(
            "/api/strategies/sniper/markets",
            headers={**auth_headers, "Content-Type": "application/json"},
            json=[]
        )
        assert response.status_code == 200
        assert response.json()["markets"] == []

    def test_set_markets_requires_json_body(self, test_client, auth_headers):
        """Markets endpoint rejects query params (must use JSON body)."""
        client, _ = test_client

        # This is the bug we caught - sending as query params instead of JSON body
        response = client.post(
            "/api/strategies/sniper/markets?markets=BTC&markets=ETH",
            headers=auth_headers
        )
        # Should fail because no body provided
        assert response.status_code == 422  # Unprocessable Entity

    def test_set_markets_invalid_strategy_returns_404(self, test_client, auth_headers):
        """Setting markets for non-existent strategy returns 404."""
        client, _ = test_client

        response = client.post(
            "/api/strategies/nonexistent/markets",
            headers={**auth_headers, "Content-Type": "application/json"},
            json=["BTC"]
        )
        assert response.status_code == 404


class TestStrategySettingsEndpoint:
    """Test POST /api/strategies/{name}/settings"""

    def test_update_settings(self, test_client, auth_headers):
        """Can update strategy settings."""
        client, _ = test_client

        response = client.post(
            "/api/strategies/sniper/settings",
            headers={**auth_headers, "Content-Type": "application/json"},
            json={"min_price": 0.80, "max_price": 0.95}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["settings"]["min_price"] == 0.80
        assert data["settings"]["max_price"] == 0.95

    def test_update_settings_persists_to_file(self, test_client, auth_headers):
        """Settings changes are saved to config file."""
        client, config_path = test_client

        response = client.post(
            "/api/strategies/sniper/settings",
            headers={**auth_headers, "Content-Type": "application/json"},
            json={"min_elapsed_sec": 500}
        )
        assert response.status_code == 200

        # Read config file and verify
        with open(config_path) as f:
            config = json.load(f)

        assert config["strategies"]["sniper"]["settings"]["min_elapsed_sec"] == 500

    def test_update_settings_partial(self, test_client, auth_headers):
        """Partial settings update preserves other settings."""
        client, config_path = test_client

        # Update only one setting
        response = client.post(
            "/api/strategies/sniper/settings",
            headers={**auth_headers, "Content-Type": "application/json"},
            json={"min_price": 0.70}
        )
        assert response.status_code == 200

        # Verify other settings are preserved
        with open(config_path) as f:
            config = json.load(f)

        settings = config["strategies"]["sniper"]["settings"]
        assert settings["min_price"] == 0.70
        assert settings["max_price"] == 0.98  # Original value preserved
        assert settings["min_elapsed_sec"] == 600  # Original value preserved

    def test_update_settings_invalid_strategy_returns_404(self, test_client, auth_headers):
        """Updating settings for non-existent strategy returns 404."""
        client, _ = test_client

        response = client.post(
            "/api/strategies/nonexistent/settings",
            headers={**auth_headers, "Content-Type": "application/json"},
            json={"foo": "bar"}
        )
        assert response.status_code == 404


class TestGetStrategiesEndpoint:
    """Test GET /api/strategies"""

    def test_get_all_strategies(self, test_client, auth_headers):
        """Can retrieve all strategies."""
        client, _ = test_client

        response = client.get("/api/strategies")
        assert response.status_code == 200
        data = response.json()

        assert "strategies" in data
        strategies = data["strategies"]

        # Verify expected strategies exist
        assert "sniper" in strategies
        assert "copy_trading" in strategies
        assert "latency_gap" in strategies

    def test_get_strategy_by_name(self, test_client, auth_headers):
        """Can retrieve a specific strategy."""
        client, _ = test_client

        response = client.get("/api/strategies/sniper")
        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "sniper"
        assert "enabled" in data
        assert "markets" in data
        assert "settings" in data


class TestFullSaveFlow:
    """Test the complete save flow as frontend does it."""

    def test_full_save_flow(self, test_client, auth_headers):
        """
        Test the complete save flow that frontend uses:
        1. Enable/disable
        2. Set markets
        3. Update settings
        """
        client, config_path = test_client

        # Step 1: Disable strategy
        response = client.post(
            "/api/strategies/sniper/enable?enabled=false",
            headers=auth_headers
        )
        assert response.status_code == 200

        # Step 2: Set markets (JSON body, not query params!)
        response = client.post(
            "/api/strategies/sniper/markets",
            headers={**auth_headers, "Content-Type": "application/json"},
            json=["BTC", "ETH"]
        )
        assert response.status_code == 200

        # Step 3: Update settings
        response = client.post(
            "/api/strategies/sniper/settings",
            headers={**auth_headers, "Content-Type": "application/json"},
            json={"min_price": 0.80, "min_elapsed_sec": 540}
        )
        assert response.status_code == 200

        # Verify everything persisted to file
        with open(config_path) as f:
            config = json.load(f)

        sniper = config["strategies"]["sniper"]
        assert sniper["enabled"] == False
        assert set(sniper["markets"]) == {"BTC", "ETH"}
        assert sniper["settings"]["min_price"] == 0.80
        assert sniper["settings"]["min_elapsed_sec"] == 540

    def test_save_all_strategies(self, test_client, auth_headers):
        """Test saving all strategies in sequence (as frontend does)."""
        client, config_path = test_client

        strategies = [
            ("latency_gap", False, ["BTC"], {"min_edge_pct": 3.0}),
            ("copy_trading", True, ["BTC", "ETH"], {"position_multiplier": 0.3}),
            ("sniper", True, ["BTC"], {"min_price": 0.75}),
            ("dip_arb", False, ["BTC"], {"dip_threshold": 0.10}),
            ("latency_arb", False, ["BTC", "ETH"], {"min_move_pct": 0.5}),
        ]

        for name, enabled, markets, settings in strategies:
            # Enable
            r = client.post(
                f"/api/strategies/{name}/enable?enabled={str(enabled).lower()}",
                headers=auth_headers
            )
            assert r.status_code == 200, f"Enable {name} failed: {r.text}"

            # Markets
            r = client.post(
                f"/api/strategies/{name}/markets",
                headers={**auth_headers, "Content-Type": "application/json"},
                json=markets
            )
            assert r.status_code == 200, f"Markets {name} failed: {r.text}"

            # Settings
            r = client.post(
                f"/api/strategies/{name}/settings",
                headers={**auth_headers, "Content-Type": "application/json"},
                json=settings
            )
            assert r.status_code == 200, f"Settings {name} failed: {r.text}"

        # Verify all saved correctly
        with open(config_path) as f:
            config = json.load(f)

        for name, enabled, markets, settings in strategies:
            strat = config["strategies"][name]
            assert strat["enabled"] == enabled, f"{name} enabled mismatch"
            assert set(strat["markets"]) == set(markets), f"{name} markets mismatch"
            for key, value in settings.items():
                assert strat["settings"][key] == value, f"{name} {key} mismatch"


class TestCORSHeaders:
    """Test CORS headers are properly set."""

    def test_options_request_returns_cors_headers(self, test_client):
        """OPTIONS request returns proper CORS headers."""
        client, _ = test_client

        response = client.options(
            "/api/strategies/sniper/enable",
            headers={"Origin": "http://localhost:3000"}
        )

        # Should allow the origin
        assert response.headers.get("access-control-allow-origin") in [
            "http://localhost:3000", "*"
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
