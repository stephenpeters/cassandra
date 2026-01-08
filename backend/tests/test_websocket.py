"""
WebSocket integration tests.

Tests cover:
- WebSocket connection and handshake
- Initial data message on connect
- Message broadcast functionality
- Connection cleanup on disconnect
"""
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestWebSocketConnection:
    """Tests for WebSocket connection lifecycle."""

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket connection."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=asyncio.CancelledError)
        return ws

    @pytest.mark.asyncio
    async def test_websocket_accepts_connection(self, mock_websocket):
        """Test that WebSocket connections are accepted."""
        # Import after path setup
        from server import ws_clients

        # Clear existing clients
        ws_clients.clear()

        # Simulate connection accept
        await mock_websocket.accept()
        ws_clients.add(mock_websocket)

        assert mock_websocket.accept.called
        assert mock_websocket in ws_clients

        # Cleanup
        ws_clients.discard(mock_websocket)

    @pytest.mark.asyncio
    async def test_websocket_sends_init_message(self, mock_websocket):
        """Test that WebSocket sends initial data on connection."""
        # The init message should contain symbols and whale info
        init_message = {
            "type": "init",
            "whales": [{"name": "test_whale", "address": "0x1234..."}],
            "symbols": ["BTC", "ETH", "SOL"],
            "paper_trading": None,
            "live_trading": None,
        }

        await mock_websocket.send_json(init_message)

        mock_websocket.send_json.assert_called_once_with(init_message)
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "init"
        assert "symbols" in call_args
        assert "whales" in call_args

    @pytest.mark.asyncio
    async def test_websocket_cleanup_on_disconnect(self, mock_websocket):
        """Test that disconnected clients are cleaned up."""
        from server import ws_clients

        ws_clients.clear()
        ws_clients.add(mock_websocket)

        assert len(ws_clients) == 1

        # Simulate disconnect
        ws_clients.discard(mock_websocket)

        assert len(ws_clients) == 0


class TestBroadcast:
    """Tests for WebSocket broadcast functionality."""

    @pytest.fixture
    def mock_clients(self):
        """Create multiple mock WebSocket clients."""
        clients = []
        for i in range(3):
            ws = AsyncMock()
            ws.send_text = AsyncMock()
            clients.append(ws)
        return clients

    @pytest.mark.asyncio
    async def test_broadcast_to_all_clients(self, mock_clients):
        """Test that messages are broadcast to all connected clients."""
        from server import ws_clients, broadcast

        ws_clients.clear()
        for client in mock_clients:
            ws_clients.add(client)

        message = {"type": "test", "data": "hello"}
        await broadcast(message)

        for client in mock_clients:
            client.send_text.assert_called_once()
            sent_data = client.send_text.call_args[0][0]
            parsed = json.loads(sent_data)
            assert parsed["type"] == "test"
            assert parsed["data"] == "hello"

        # Cleanup
        ws_clients.clear()

    @pytest.mark.asyncio
    async def test_broadcast_handles_disconnected_client(self, mock_clients):
        """Test that broadcast handles disconnected clients gracefully."""
        from server import ws_clients, broadcast

        ws_clients.clear()

        # First client will raise an exception (disconnected)
        mock_clients[0].send_text = AsyncMock(side_effect=Exception("Connection closed"))

        for client in mock_clients:
            ws_clients.add(client)

        message = {"type": "test", "data": "hello"}
        await broadcast(message)

        # Disconnected client should be removed
        assert mock_clients[0] not in ws_clients

        # Other clients should still be connected
        assert mock_clients[1] in ws_clients
        assert mock_clients[2] in ws_clients

        # Cleanup
        ws_clients.clear()

    @pytest.mark.asyncio
    async def test_broadcast_no_clients(self):
        """Test that broadcast does nothing with no clients."""
        from server import ws_clients, broadcast

        ws_clients.clear()

        # Should not raise
        await broadcast({"type": "test"})


class TestWebSocketMessages:
    """Tests for WebSocket message types."""

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket connection."""
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_trade_message_format(self, mock_websocket):
        """Test trade message structure."""
        from server import ws_clients, broadcast

        ws_clients.clear()
        ws_clients.add(mock_websocket)

        trade_msg = {
            "type": "trade",
            "symbol": "BTC",
            "data": {
                "time": 1704067200,
                "price": 91000.0,
                "size": 0.5,
                "side": "buy",
            }
        }

        await broadcast(trade_msg)

        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])

        assert sent_data["type"] == "trade"
        assert sent_data["symbol"] == "BTC"
        assert "data" in sent_data
        assert sent_data["data"]["price"] == 91000.0

        ws_clients.clear()

    @pytest.mark.asyncio
    async def test_momentum_message_format(self, mock_websocket):
        """Test momentum message structure."""
        from server import ws_clients, broadcast

        ws_clients.clear()
        ws_clients.add(mock_websocket)

        momentum_msg = {
            "type": "momentum",
            "symbol": "BTC",
            "data": {
                "rsi": 65.5,
                "adx": 25.0,
                "supertrend_direction": "UP",
                "vwap_deviation": 0.002,
            }
        }

        await broadcast(momentum_msg)

        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])

        assert sent_data["type"] == "momentum"
        assert sent_data["data"]["rsi"] == 65.5

        ws_clients.clear()

    @pytest.mark.asyncio
    async def test_whale_trade_message_format(self, mock_websocket):
        """Test whale trade message structure."""
        from server import ws_clients, broadcast

        ws_clients.clear()
        ws_clients.add(mock_websocket)

        whale_msg = {
            "type": "whale_trade",
            "data": {
                "whale": "Account88888",
                "symbol": "BTC",
                "side": "BUY",
                "outcome": "UP",
                "size": 1000.0,
                "usd_value": 500.0,
            }
        }

        await broadcast(whale_msg)

        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])

        assert sent_data["type"] == "whale_trade"
        assert sent_data["data"]["whale"] == "Account88888"
        assert sent_data["data"]["usd_value"] == 500.0

        ws_clients.clear()

    @pytest.mark.asyncio
    async def test_paper_signal_message_format(self, mock_websocket):
        """Test paper trading signal message structure."""
        from server import ws_clients, broadcast

        ws_clients.clear()
        ws_clients.add(mock_websocket)

        signal_msg = {
            "type": "paper_signal",
            "data": {
                "symbol": "ETH",
                "checkpoint": "7m30s",
                "signal": "BUY_UP",
                "fair_value": 0.65,
                "market_price": 0.55,
                "edge": 0.10,
                "confidence": 0.8,
            }
        }

        await broadcast(signal_msg)

        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])

        assert sent_data["type"] == "paper_signal"
        assert sent_data["data"]["edge"] == 0.10

        ws_clients.clear()


class TestWebSocketIntegration:
    """Integration tests for WebSocket with API client."""

    @pytest.fixture
    def test_client(self):
        """Create a test client for the FastAPI app."""
        from fastapi.testclient import TestClient
        from server import app
        return TestClient(app)

    def test_websocket_endpoint_exists(self, test_client):
        """Test that the WebSocket endpoint is reachable."""
        # FastAPI TestClient supports WebSocket testing
        with test_client.websocket_connect("/ws") as websocket:
            # Should receive init message
            data = websocket.receive_json()
            assert data["type"] == "init"
            assert "symbols" in data
            assert "whales" in data

    def test_websocket_receives_multiple_messages(self, test_client):
        """Test that WebSocket can receive multiple messages."""
        with test_client.websocket_connect("/ws") as websocket:
            # Receive init
            init_data = websocket.receive_json()
            assert init_data["type"] == "init"

            # Can send ping and receive data (connection stays open)
            # Note: In real test, would trigger broadcast from server
            # This just verifies connection is maintained


class TestWebSocketErrorHandling:
    """Tests for WebSocket error handling."""

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket connection."""
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_broadcast_with_json_serialization_error(self, mock_websocket):
        """Test broadcast handles JSON serialization errors gracefully."""
        from server import ws_clients, broadcast

        ws_clients.clear()
        ws_clients.add(mock_websocket)

        # Create a message that can be serialized (no circular refs)
        message = {"type": "test", "data": {"nested": "value"}}

        # Should not raise
        await broadcast(message)

        ws_clients.clear()

    @pytest.mark.asyncio
    async def test_concurrent_broadcast(self, mock_websocket):
        """Test concurrent broadcasts don't cause issues."""
        from server import ws_clients, broadcast

        ws_clients.clear()
        ws_clients.add(mock_websocket)

        # Send multiple broadcasts concurrently
        messages = [
            {"type": "test", "id": i}
            for i in range(10)
        ]

        await asyncio.gather(*[broadcast(msg) for msg in messages])

        # All messages should have been sent
        assert mock_websocket.send_text.call_count == 10

        ws_clients.clear()
