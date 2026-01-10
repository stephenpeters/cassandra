"""
Tests for the Sniper Strategy (trading.py).

Tests cover:
- Signal generation based on price thresholds
- Signal generation based on elapsed time
- Market filtering
- Position tracking (no duplicate entries)
- Edge cases (exactly at threshold)

Note: check_signal returns tuple[str, dict] or None, not just "UP"/"DOWN"
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading import SniperConfig, SniperStrategy


class TestSniperConfig:
    """Tests for SniperConfig."""

    def test_config_defaults(self):
        """Test default config values."""
        config = SniperConfig()
        assert config.enabled is True
        assert config.min_price == 0.75
        assert config.min_elapsed_sec == 600
        assert config.markets == ["BTC"]  # Changed: now just BTC for paper trading
        assert config.position_size_pct == 2.0
        assert config.max_position_usd == 100.0

    def test_config_custom_values(self):
        """Test custom config values."""
        config = SniperConfig(
            enabled=False,
            min_price=0.80,
            min_elapsed_sec=480,
            markets=["BTC", "SOL"],
            position_size_pct=5.0,
            max_position_usd=200.0,
        )
        assert config.enabled is False
        assert config.min_price == 0.80
        assert config.min_elapsed_sec == 480
        assert config.markets == ["BTC", "SOL"]
        assert config.position_size_pct == 5.0
        assert config.max_position_usd == 200.0

    def test_config_to_dict(self):
        """Test config serialization."""
        config = SniperConfig(min_price=0.80)
        data = config.to_dict()
        assert data["min_price"] == 0.80
        assert data["enabled"] is True
        assert "markets" in data


class TestSniperStrategy:
    """Tests for SniperStrategy signal generation."""

    @pytest.fixture
    def strategy(self):
        """Create a default sniper strategy with BTC and ETH."""
        return SniperStrategy(SniperConfig(markets=["BTC", "ETH"]))

    @pytest.fixture
    def aggressive_strategy(self):
        """Create an aggressive sniper strategy (lower thresholds)."""
        return SniperStrategy(SniperConfig(
            min_price=0.70,
            min_elapsed_sec=300,  # 5 minutes
        ))

    # -------------------------------------------------------------------------
    # Basic Signal Tests
    # -------------------------------------------------------------------------

    def test_no_signal_when_disabled(self, strategy):
        """No signal when strategy is disabled."""
        strategy.config.enabled = False
        signal = strategy.check_signal(
            symbol="BTC",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=700,
            market_start=1000000,
        )
        assert signal is None

    def test_no_signal_for_unlisted_market(self, strategy):
        """No signal for markets not in config.markets."""
        signal = strategy.check_signal(
            symbol="DOGE",  # Not in ["BTC", "ETH"]
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=700,
            market_start=1000000,
        )
        assert signal is None

    def test_no_signal_before_min_elapsed(self, strategy):
        """No signal before min_elapsed_sec is reached."""
        signal = strategy.check_signal(
            symbol="BTC",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=500,  # < 600 default
            market_start=1000000,
        )
        assert signal is None

    def test_signal_picks_higher_probability_side(self, strategy):
        """Signal picks the higher-probability side (no min_price gating)."""
        signal = strategy.check_signal(
            symbol="BTC",
            up_price=0.60,  # Higher side wins
            down_price=0.40,
            elapsed_sec=700,
            market_start=1000000,
        )
        # Sniper now signals on the higher side without min_price gating
        assert signal is not None
        assert signal[0] == "UP"

    # -------------------------------------------------------------------------
    # Positive Signal Tests
    # -------------------------------------------------------------------------

    def test_up_signal_when_conditions_met(self, strategy):
        """UP signal when up_price > down_price and elapsed >= min_elapsed."""
        signal = strategy.check_signal(
            symbol="BTC",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=700,
            market_start=1000000,
        )
        assert signal is not None
        assert signal[0] == "UP"
        assert "ev_pct" in signal[1]  # Contains EV info (for reporting)

    def test_down_signal_when_conditions_met(self, strategy):
        """DOWN signal when down_price > up_price."""
        signal = strategy.check_signal(
            symbol="ETH",
            up_price=0.20,
            down_price=0.80,
            elapsed_sec=700,
            market_start=1000001,
        )
        assert signal is not None
        assert signal[0] == "DOWN"

    def test_up_wins_when_prices_equal(self, strategy):
        """UP is returned when both sides are equal (tiebreaker)."""
        # When prices are equal, UP wins by convention
        signal = strategy.check_signal(
            symbol="BTC",
            up_price=0.50,
            down_price=0.50,
            elapsed_sec=700,
            market_start=1000002,
        )
        assert signal is not None
        assert signal[0] == "UP"

    # -------------------------------------------------------------------------
    # Edge Case Tests
    # -------------------------------------------------------------------------

    def test_signal_at_high_probability(self, strategy):
        """Signal triggers at high probability (75%)."""
        signal = strategy.check_signal(
            symbol="BTC",
            up_price=0.75,
            down_price=0.25,
            elapsed_sec=700,
            market_start=1000003,
        )
        assert signal is not None
        assert signal[0] == "UP"

    def test_signal_exactly_at_time_threshold(self, strategy):
        """Signal triggers at exactly min_elapsed_sec (inclusive)."""
        signal = strategy.check_signal(
            symbol="BTC",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=600,  # Exactly at threshold
            market_start=1000004,
        )
        assert signal is not None
        assert signal[0] == "UP"

    def test_signal_at_low_probability_still_works(self, strategy):
        """Signal works at any probability (no min_price gating)."""
        signal = strategy.check_signal(
            symbol="BTC",
            up_price=0.51,  # Just above 50%
            down_price=0.49,
            elapsed_sec=700,
            market_start=1000005,
        )
        # Sniper signals on higher side regardless of probability level
        assert signal is not None
        assert signal[0] == "UP"

    def test_no_signal_just_before_time_threshold(self, strategy):
        """No signal at 599 seconds (just below 600)."""
        signal = strategy.check_signal(
            symbol="BTC",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=599,
            market_start=1000006,
        )
        assert signal is None

    # -------------------------------------------------------------------------
    # Position Tracking Tests
    # -------------------------------------------------------------------------

    def test_no_duplicate_signal_same_window(self, strategy):
        """No signal after position already taken for same window."""
        market_start = 1000007

        # First signal should work
        signal1 = strategy.check_signal(
            symbol="BTC",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=700,
            market_start=market_start,
        )
        assert signal1 is not None
        assert signal1[0] == "UP"

        # Record the position
        strategy.record_position("BTC", market_start)

        # Second check should return None
        signal2 = strategy.check_signal(
            symbol="BTC",
            up_price=0.85,
            down_price=0.15,
            elapsed_sec=800,
            market_start=market_start,
        )
        assert signal2 is None

    def test_signal_allowed_for_different_window(self, strategy):
        """Signal allowed for different market window."""
        market_start_1 = 1000008
        market_start_2 = 1000009

        # Take position in first window
        signal1 = strategy.check_signal(
            symbol="BTC",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=700,
            market_start=market_start_1,
        )
        strategy.record_position("BTC", market_start_1)
        assert signal1 is not None
        assert signal1[0] == "UP"

        # Should still get signal for second window
        signal2 = strategy.check_signal(
            symbol="BTC",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=700,
            market_start=market_start_2,
        )
        assert signal2 is not None
        assert signal2[0] == "UP"

    def test_signal_allowed_for_different_symbol_same_window(self, strategy):
        """Signal allowed for different symbol in same time window."""
        market_start = 1000010

        # Take position in BTC
        signal1 = strategy.check_signal(
            symbol="BTC",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=700,
            market_start=market_start,
        )
        strategy.record_position("BTC", market_start)
        assert signal1 is not None
        assert signal1[0] == "UP"

        # Should still get signal for ETH
        signal2 = strategy.check_signal(
            symbol="ETH",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=700,
            market_start=market_start,
        )
        assert signal2 is not None
        assert signal2[0] == "UP"

    # -------------------------------------------------------------------------
    # Aggressive Strategy Tests
    # -------------------------------------------------------------------------

    def test_aggressive_strategy_triggers_earlier(self, aggressive_strategy):
        """Aggressive strategy triggers at 5 minutes (300s)."""
        signal = aggressive_strategy.check_signal(
            symbol="BTC",
            up_price=0.72,  # Above 0.70 threshold
            down_price=0.28,
            elapsed_sec=350,  # Above 300s threshold
            market_start=1000011,
        )
        assert signal is not None
        assert signal[0] == "UP"

    def test_aggressive_strategy_lower_price_threshold(self, aggressive_strategy):
        """Aggressive strategy triggers at 70c."""
        signal = aggressive_strategy.check_signal(
            symbol="BTC",
            up_price=0.70,  # At 0.70 threshold
            down_price=0.30,
            elapsed_sec=350,
            market_start=1000012,
        )
        assert signal is not None
        assert signal[0] == "UP"


class TestSniperStrategyIntegration:
    """Integration tests for sniper strategy with realistic scenarios."""

    def test_btc_late_window_high_probability(self):
        """Realistic BTC scenario: 80c UP price at 11 minutes."""
        strategy = SniperStrategy(SniperConfig())

        signal = strategy.check_signal(
            symbol="BTC",
            up_price=0.80,
            down_price=0.20,
            elapsed_sec=660,  # 11 minutes
            market_start=1704067200,  # Real timestamp
        )
        assert signal is not None
        assert signal[0] == "UP"

    def test_eth_reversal_detected_after_time_threshold(self):
        """Test that market reversals are detected once time threshold is met."""
        strategy = SniperStrategy(SniperConfig(markets=["ETH"]))

        # At 5 minutes, price shows 72c UP - but we don't trigger (too early)
        signal_early = strategy.check_signal(
            symbol="ETH",
            up_price=0.72,
            down_price=0.28,
            elapsed_sec=300,  # 5 minutes - too early
            market_start=1704067200,
        )
        assert signal_early is None

        # Market reverses - by 10 minutes, DOWN is now leading
        signal_late = strategy.check_signal(
            symbol="ETH",
            up_price=0.35,
            down_price=0.65,  # DOWN is now higher
            elapsed_sec=600,
            market_start=1704067200,
        )
        # Now signals DOWN because it's the higher side and time threshold met
        assert signal_late is not None
        assert signal_late[0] == "DOWN"

    def test_xrp_not_in_default_markets(self):
        """XRP signals are filtered by default config."""
        strategy = SniperStrategy(SniperConfig())

        signal = strategy.check_signal(
            symbol="XRP",
            up_price=0.85,
            down_price=0.15,
            elapsed_sec=700,
            market_start=1704067200,
        )
        assert signal is None

    def test_custom_markets_include_xrp(self):
        """Custom config can include XRP."""
        strategy = SniperStrategy(SniperConfig(
            markets=["BTC", "ETH", "XRP"]
        ))

        signal = strategy.check_signal(
            symbol="XRP",
            up_price=0.85,
            down_price=0.15,
            elapsed_sec=700,
            market_start=1704067200,
        )
        assert signal is not None
        assert signal[0] == "UP"


class TestSniperPositionSizing:
    """Tests for position size calculation logic."""

    def test_position_size_within_limits(self):
        """Position size respects both percentage and max USD."""
        config = SniperConfig(
            position_size_pct=2.0,
            max_position_usd=100.0,
        )

        # With $5000 balance: 2% = $100, capped at $100
        account_balance = 5000
        position_size = min(
            account_balance * (config.position_size_pct / 100),
            config.max_position_usd,
        )
        assert position_size == 100.0

    def test_position_size_percentage_dominates(self):
        """Percentage is limiting factor for small accounts."""
        config = SniperConfig(
            position_size_pct=2.0,
            max_position_usd=100.0,
        )

        # With $1000 balance: 2% = $20, below $100 cap
        account_balance = 1000
        position_size = min(
            account_balance * (config.position_size_pct / 100),
            config.max_position_usd,
        )
        assert position_size == 20.0

    def test_position_size_max_usd_dominates(self):
        """Max USD is limiting factor for large accounts."""
        config = SniperConfig(
            position_size_pct=5.0,
            max_position_usd=100.0,
        )

        # With $10000 balance: 5% = $500, capped at $100
        account_balance = 10000
        position_size = min(
            account_balance * (config.position_size_pct / 100),
            config.max_position_usd,
        )
        assert position_size == 100.0
