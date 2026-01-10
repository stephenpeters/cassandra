"""
Tests for Strategy Manager.

Tests cover:
- Strategy enable/disable
- Market filtering per strategy
- Copy trading trader management
- Position sizing calculations
"""

import pytest
import json
import tempfile
import os

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_manager import (
    StrategyManager,
    StrategyConfig,
    TraderConfig,
    get_strategy_manager,
)


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing"""
    config = {
        "strategies": {
            "latency_gap": {
                "enabled": True,
                "description": "Test latency gap",
                "markets": ["BTC", "ETH"],
                "settings": {
                    "min_edge_pct": 5.0,
                    "checkpoints": ["7m30s", "9m"],
                },
            },
            "copy_trading": {
                "enabled": True,
                "description": "Test copy trading",
                "markets": ["BTC"],
                "settings": {
                    "scale_to_account": True,
                    "reference_balance": 10000,
                    "position_multiplier": 0.5,
                    "max_position_usd": 100,
                },
                "traders": [
                    {
                        "name": "TestTrader1",
                        "address": "0x1234567890abcdef",
                        "enabled": True,
                        "min_trade_size": 500,
                        "trust_score": 0.8,
                    },
                    {
                        "name": "TestTrader2",
                        "address": "0xabcdef1234567890",
                        "enabled": False,
                        "min_trade_size": 1000,
                        "trust_score": 0.6,
                    },
                ],
            },
        },
        "global_settings": {
            "enabled_markets": ["BTC", "ETH", "SOL"],
            "max_concurrent_positions": 3,
        },
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


class TestStrategyConfig:
    """Test StrategyConfig dataclass"""

    def test_is_enabled_for_market(self):
        """Test market filtering"""
        config = StrategyConfig(
            name="test",
            enabled=True,
            markets=["BTC", "ETH"],
        )

        assert config.is_enabled_for_market("BTC") is True
        assert config.is_enabled_for_market("btc") is True  # Case insensitive
        assert config.is_enabled_for_market("ETH") is True
        assert config.is_enabled_for_market("SOL") is False

    def test_disabled_strategy_not_enabled_for_any_market(self):
        """Disabled strategy returns False for all markets"""
        config = StrategyConfig(
            name="test",
            enabled=False,
            markets=["BTC", "ETH"],
        )

        assert config.is_enabled_for_market("BTC") is False
        assert config.is_enabled_for_market("ETH") is False


class TestStrategyManager:
    """Test StrategyManager class"""

    def test_load_config(self, temp_config_file):
        """Test loading config from file"""
        manager = StrategyManager(config_path=temp_config_file)

        assert "latency_gap" in manager.strategies
        assert "copy_trading" in manager.strategies
        assert manager.strategies["latency_gap"].enabled is True
        assert manager.strategies["latency_gap"].markets == ["BTC", "ETH"]

    def test_get_strategy(self, temp_config_file):
        """Test getting a specific strategy"""
        manager = StrategyManager(config_path=temp_config_file)

        strategy = manager.get_strategy("latency_gap")
        assert strategy is not None
        assert strategy.name == "latency_gap"

        # Non-existent strategy
        assert manager.get_strategy("nonexistent") is None

    def test_is_strategy_enabled(self, temp_config_file):
        """Test checking if strategy is enabled for market"""
        manager = StrategyManager(config_path=temp_config_file)

        assert manager.is_strategy_enabled("latency_gap", "BTC") is True
        assert manager.is_strategy_enabled("latency_gap", "ETH") is True
        assert manager.is_strategy_enabled("latency_gap", "SOL") is False

        # Copy trading only enabled for BTC
        assert manager.is_strategy_enabled("copy_trading", "BTC") is True
        assert manager.is_strategy_enabled("copy_trading", "ETH") is False

    def test_get_enabled_strategies(self, temp_config_file):
        """Test getting all enabled strategies for a market"""
        manager = StrategyManager(config_path=temp_config_file)

        btc_strategies = manager.get_enabled_strategies("BTC")
        assert "latency_gap" in btc_strategies
        assert "copy_trading" in btc_strategies

        eth_strategies = manager.get_enabled_strategies("ETH")
        assert "latency_gap" in eth_strategies
        assert "copy_trading" not in eth_strategies

    def test_enable_disable_strategy(self, temp_config_file):
        """Test enabling/disabling strategies"""
        manager = StrategyManager(config_path=temp_config_file)

        # Disable latency_gap
        manager.enable_strategy("latency_gap", False)
        assert manager.strategies["latency_gap"].enabled is False
        assert manager.is_strategy_enabled("latency_gap", "BTC") is False

        # Re-enable
        manager.enable_strategy("latency_gap", True)
        assert manager.strategies["latency_gap"].enabled is True

    def test_set_strategy_markets(self, temp_config_file):
        """Test setting strategy markets"""
        manager = StrategyManager(config_path=temp_config_file)

        manager.set_strategy_markets("latency_gap", ["SOL", "XRP"])
        assert manager.strategies["latency_gap"].markets == ["SOL", "XRP"]
        assert manager.is_strategy_enabled("latency_gap", "SOL") is True
        assert manager.is_strategy_enabled("latency_gap", "BTC") is False


class TestCopyTradingTraders:
    """Test copy trading trader management"""

    def test_get_copy_traders(self, temp_config_file):
        """Test getting all traders"""
        manager = StrategyManager(config_path=temp_config_file)

        traders = manager.get_copy_traders()
        assert len(traders) == 2
        assert traders[0].name == "TestTrader1"
        assert traders[1].name == "TestTrader2"

    def test_get_enabled_traders(self, temp_config_file):
        """Test getting only enabled traders"""
        manager = StrategyManager(config_path=temp_config_file)

        enabled = manager.get_enabled_traders()
        assert len(enabled) == 1
        assert enabled[0].name == "TestTrader1"

    def test_enable_disable_trader(self, temp_config_file):
        """Test enabling/disabling a trader"""
        manager = StrategyManager(config_path=temp_config_file)

        # Enable TestTrader2
        manager.enable_trader("TestTrader2", True)
        enabled = manager.get_enabled_traders()
        assert len(enabled) == 2

        # Disable TestTrader1
        manager.enable_trader("TestTrader1", False)
        enabled = manager.get_enabled_traders()
        assert len(enabled) == 1
        assert enabled[0].name == "TestTrader2"

    def test_add_trader(self, temp_config_file):
        """Test adding a new trader"""
        manager = StrategyManager(config_path=temp_config_file)

        new_trader = TraderConfig(
            name="NewTrader",
            address="0xnewaddress",
            enabled=True,
        )
        manager.add_trader(new_trader)

        traders = manager.get_copy_traders()
        assert len(traders) == 3
        assert traders[2].name == "NewTrader"

    def test_remove_trader(self, temp_config_file):
        """Test removing a trader"""
        manager = StrategyManager(config_path=temp_config_file)

        manager.remove_trader("TestTrader1")

        traders = manager.get_copy_traders()
        assert len(traders) == 1
        assert traders[0].name == "TestTrader2"


class TestPositionSizing:
    """Test position sizing calculations"""

    def test_copy_trading_position_scaling(self, temp_config_file):
        """Test position scaling for copy trading"""
        manager = StrategyManager(config_path=temp_config_file)

        # Whale bets $5000, our account is $1000, reference is $10000
        # scale_factor = 1000 / 10000 = 0.1
        # position = 5000 * 0.1 * 0.5 = $250
        # But max is $100, so should be $100
        size = manager.calculate_position_size(
            strategy="copy_trading",
            whale_bet_size=5000,
            account_balance=1000,
            max_position=500,
        )
        assert size == 100  # Capped by max_position_usd in settings

    def test_copy_trading_small_account(self, temp_config_file):
        """Test position scaling with small account"""
        manager = StrategyManager(config_path=temp_config_file)

        # Account is $500, reference is $10000
        # scale_factor = 500 / 10000 = 0.05
        # position = 1000 * 0.05 * 0.5 = $25
        size = manager.calculate_position_size(
            strategy="copy_trading",
            whale_bet_size=1000,
            account_balance=500,
            max_position=500,
        )
        assert size == 25

    def test_latency_gap_position_sizing(self, temp_config_file):
        """Test position sizing for latency gap"""
        manager = StrategyManager(config_path=temp_config_file)

        # Update settings to have position_size_pct
        manager.update_strategy_settings("latency_gap", {"position_size_pct": 10})

        # 10% of $1000 = $100
        size = manager.calculate_position_size(
            strategy="latency_gap",
            whale_bet_size=0,  # Not used for latency gap
            account_balance=1000,
            max_position=500,
        )
        assert size == 100

    def test_position_respects_max(self, temp_config_file):
        """Test that position size respects max_position"""
        manager = StrategyManager(config_path=temp_config_file)

        manager.update_strategy_settings("latency_gap", {"position_size_pct": 50})

        # 50% of $1000 = $500, but max is $200
        size = manager.calculate_position_size(
            strategy="latency_gap",
            whale_bet_size=0,
            account_balance=1000,
            max_position=200,
        )
        assert size == 200


class TestPersistence:
    """Test config persistence"""

    def test_save_and_reload(self, temp_config_file):
        """Test that changes are saved and reloaded correctly"""
        manager = StrategyManager(config_path=temp_config_file)

        # Make changes
        manager.enable_strategy("latency_gap", False)
        manager.set_strategy_markets("copy_trading", ["BTC", "ETH", "SOL"])
        manager.enable_trader("TestTrader2", True)

        # Create new manager to reload
        manager2 = StrategyManager(config_path=temp_config_file)

        assert manager2.strategies["latency_gap"].enabled is False
        assert manager2.strategies["copy_trading"].markets == ["BTC", "ETH", "SOL"]

        enabled_traders = manager2.get_enabled_traders()
        trader_names = [t.name for t in enabled_traders]
        assert "TestTrader2" in trader_names


class TestSniperStrategyManager:
    """Test sniper strategy management through StrategyManager"""

    @pytest.fixture
    def sniper_config_file(self):
        """Create a config file with sniper strategy"""
        config = {
            "strategies": {
                "sniper": {
                    "enabled": True,
                    "description": "Buy high-probability outcomes late in window",
                    "markets": ["BTC", "ETH"],
                    "settings": {
                        "min_price": 0.75,
                        "min_elapsed_sec": 600,
                        "position_size_pct": 2.0,
                        "max_position_usd": 100,
                    },
                },
            },
            "global_settings": {
                "enabled_markets": ["BTC", "ETH", "SOL", "XRP"],
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_sniper_strategy_loads(self, sniper_config_file):
        """Test sniper strategy loads from config"""
        manager = StrategyManager(config_path=sniper_config_file)

        assert "sniper" in manager.strategies
        assert manager.strategies["sniper"].enabled is True
        assert manager.strategies["sniper"].markets == ["BTC", "ETH"]

    def test_sniper_settings_load(self, sniper_config_file):
        """Test sniper settings are loaded correctly"""
        manager = StrategyManager(config_path=sniper_config_file)

        settings = manager.strategies["sniper"].settings
        assert settings["min_price"] == 0.75
        assert settings["min_elapsed_sec"] == 600
        assert settings["position_size_pct"] == 2.0
        assert settings["max_position_usd"] == 100

    def test_sniper_market_filtering(self, sniper_config_file):
        """Test sniper only applies to configured markets"""
        manager = StrategyManager(config_path=sniper_config_file)

        assert manager.is_strategy_enabled("sniper", "BTC") is True
        assert manager.is_strategy_enabled("sniper", "ETH") is True
        assert manager.is_strategy_enabled("sniper", "SOL") is False
        assert manager.is_strategy_enabled("sniper", "XRP") is False

    def test_sniper_enable_disable(self, sniper_config_file):
        """Test enabling/disabling sniper strategy"""
        manager = StrategyManager(config_path=sniper_config_file)

        manager.enable_strategy("sniper", False)
        assert manager.strategies["sniper"].enabled is False
        assert manager.is_strategy_enabled("sniper", "BTC") is False

        manager.enable_strategy("sniper", True)
        assert manager.strategies["sniper"].enabled is True

    def test_sniper_update_markets(self, sniper_config_file):
        """Test updating sniper markets"""
        manager = StrategyManager(config_path=sniper_config_file)

        manager.set_strategy_markets("sniper", ["BTC", "ETH", "SOL", "XRP"])
        assert manager.strategies["sniper"].markets == ["BTC", "ETH", "SOL", "XRP"]
        assert manager.is_strategy_enabled("sniper", "SOL") is True

    def test_sniper_update_settings(self, sniper_config_file):
        """Test updating sniper settings"""
        manager = StrategyManager(config_path=sniper_config_file)

        manager.update_strategy_settings("sniper", {
            "min_price": 0.80,
            "min_elapsed_sec": 480,
        })

        settings = manager.strategies["sniper"].settings
        assert settings["min_price"] == 0.80
        assert settings["min_elapsed_sec"] == 480
        # Other settings should remain unchanged
        assert settings["position_size_pct"] == 2.0

    def test_sniper_position_sizing(self, sniper_config_file):
        """Test sniper position size calculation"""
        manager = StrategyManager(config_path=sniper_config_file)

        # 2% of $5000 = $100, same as max_position_usd
        size = manager.calculate_position_size(
            strategy="sniper",
            whale_bet_size=0,  # Not used for sniper
            account_balance=5000,
            max_position=500,
        )
        assert size == 100  # Capped by max_position_usd

    def test_sniper_position_sizing_small_account(self, sniper_config_file):
        """Test sniper position sizing with small account"""
        manager = StrategyManager(config_path=sniper_config_file)

        # 2% of $1000 = $20, below max_position_usd
        size = manager.calculate_position_size(
            strategy="sniper",
            whale_bet_size=0,
            account_balance=1000,
            max_position=500,
        )
        assert size == 20

    def test_sniper_position_sizing_respects_global_max(self, sniper_config_file):
        """Test sniper respects global max_position"""
        manager = StrategyManager(config_path=sniper_config_file)

        # Update to aggressive settings
        manager.update_strategy_settings("sniper", {
            "position_size_pct": 10.0,
            "max_position_usd": 500,
        })

        # 10% of $1000 = $100, but global max is $50
        size = manager.calculate_position_size(
            strategy="sniper",
            whale_bet_size=0,
            account_balance=1000,
            max_position=50,  # Global max
        )
        assert size == 50

    def test_sniper_persistence(self, sniper_config_file):
        """Test sniper settings persist after reload"""
        manager = StrategyManager(config_path=sniper_config_file)

        # Make changes
        manager.enable_strategy("sniper", False)
        manager.set_strategy_markets("sniper", ["SOL", "DOGE"])
        manager.update_strategy_settings("sniper", {"min_price": 0.70})

        # Reload
        manager2 = StrategyManager(config_path=sniper_config_file)

        assert manager2.strategies["sniper"].enabled is False
        assert manager2.strategies["sniper"].markets == ["SOL", "DOGE"]
        assert manager2.strategies["sniper"].settings["min_price"] == 0.70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
