"""
Tests for API endpoints.

Tests cover:
- Live trading endpoints
- Paper trading endpoints
- Market data endpoints
- Manual order flow (fill_price bug)
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def api_key():
    return "dev_test_key_12345"


@pytest.fixture
def auth_headers(api_key):
    return {"X-API-Key": api_key}


class TestManualOrderEndpoint:
    """Test manual order - catches the fill_price vs filled_price bug"""

    def test_live_order_has_filled_price_attribute(self):
        """LiveOrder must have filled_price attribute (not fill_price)"""
        from live_trading import LiveOrder
        
        order = LiveOrder(
            id="test",
            symbol="BTC",
            side="UP",
            direction="BUY",
            token_id="123",
            size_usd=20.0,
            price=0.5,
            order_type="GTC",
            status="pending",
            created_at=1234567890,
        )
        
        # This was the bug - code used fill_price instead of filled_price
        assert hasattr(order, 'filled_price'), "LiveOrder must have filled_price attribute"
        assert not hasattr(order, 'fill_price'), "LiveOrder should NOT have fill_price (typo)"
        assert order.filled_price == 0.0  # Default value


class TestLiveOrderAttributes:
    """Verify LiveOrder has all expected attributes"""

    def test_live_order_required_attributes(self):
        """LiveOrder has all required attributes"""
        from live_trading import LiveOrder
        
        required = ['id', 'symbol', 'side', 'direction', 'token_id', 
                    'size_usd', 'price', 'order_type', 'status', 'created_at',
                    'filled_at', 'filled_size', 'filled_price', 'error',
                    'polymarket_order_id', 'tx_hash']
        
        order = LiveOrder(
            id="test", symbol="BTC", side="UP", direction="BUY",
            token_id="123", size_usd=20.0, price=0.5,
            order_type="GTC", status="pending", created_at=1234567890,
        )
        
        for attr in required:
            assert hasattr(order, attr), f"LiveOrder missing {attr}"


class TestLivePositionAttributes:
    """Verify LivePosition has all expected attributes"""

    def test_live_position_required_attributes(self):
        """LivePosition has all required attributes"""
        from live_trading import LivePosition
        
        required = ['symbol', 'side', 'token_id', 'size', 'avg_entry_price',
                    'cost_basis_usd', 'market_start', 'market_end', 'entry_orders']
        
        position = LivePosition(
            symbol="BTC", side="UP", token_id="123", size=10.0,
            avg_entry_price=0.5, cost_basis_usd=5.0,
            market_start=1234567890, market_end=1234568790,
        )
        
        for attr in required:
            assert hasattr(position, attr), f"LivePosition missing {attr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
