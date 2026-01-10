"""
Sniper Strategy - Buy high-probability outcomes late in the 15-min window.

Refactored to use the new Strategy base class interface.
Strategies ONLY produce signals - execution is handled by Executor layer.
"""

import logging
from dataclasses import dataclass, asdict, field
from typing import Optional

from strategy_base import (
    Strategy,
    MarketEvent,
    Signal,
    MarketType,
    StrategyRecommendation,
)

logger = logging.getLogger(__name__)


@dataclass
class SniperConfig:
    """
    Sniper Strategy Configuration.

    Entry Signal:
        IF: UP price >= min_price OR DOWN price >= min_price
        AND: UP price <= max_price AND DOWN price <= max_price (avoid -EV)
        AND: elapsed_seconds >= min_elapsed_sec
        AND: EV >= min_ev_pct
        THEN: Buy the leading side

    Backtest Results (75c+ after 10min):
        - Win rate: 100% (2/2 trades)
        - P&L: +$44 per $100 wagered

    EV Calculation:
        At 75c entry with 90% true prob: EV = +19.4%
        At 98c entry with 99% true prob: EV = -1% to +1% (fees eat profit)
        => Don't buy above 98c as EV becomes negative
    """
    enabled: bool = True
    min_price: float = 0.75  # Minimum probability to enter (75c)
    max_price: float = 0.98  # Maximum price - don't buy above this (EV negative)
    min_elapsed_sec: int = 600  # Minimum seconds into window (10 minutes)
    markets: list = field(default_factory=lambda: ["BTC"])  # Focus on BTC for paper trading
    position_size_pct: float = 2.0  # % of account per trade
    max_position_usd: float = 100.0  # Hard cap per position
    fee_rate: float = 0.02  # Estimated fee + slippage (2%)
    min_ev_pct: float = 3.0  # Minimum EV% required to trade
    assumed_win_rate: float = 0.90  # Assumed win rate for EV calc (late entry = high confidence)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SniperConfig":
        return cls(**data)


class SniperStrategy(Strategy):
    """
    Sniper Strategy implementation using the new Strategy base class.

    Buys the high-probability side (>=min_price) after min_elapsed_sec into the window.
    PM-only conditions - no Binance confirmation required.
    """

    name = "sniper"
    description = "Buy high-probability outcomes late in window (75c-98c, EV>3%)"

    def __init__(self, config: dict, market_type: Optional[MarketType] = None):
        """
        Initialize sniper strategy.

        Args:
            config: Configuration dict (will be converted to SniperConfig)
            market_type: Market timeframe type
        """
        super().__init__(config, market_type)

        # Convert dict config to SniperConfig
        if isinstance(config, dict):
            self.sniper_config = SniperConfig.from_dict(config)
        elif isinstance(config, SniperConfig):
            self.sniper_config = config
        else:
            self.sniper_config = SniperConfig()

    @classmethod
    def get_recommendations(cls) -> list[StrategyRecommendation]:
        """Return recommendations for each market type."""
        return [
            StrategyRecommendation(
                MarketType.FIFTEEN_MIN,
                recommended=True,
                reason="Optimized for 15-min windows. Entry after 10min gives 5min for price to converge."
            ),
            StrategyRecommendation(
                MarketType.ONE_HOUR,
                recommended=True,
                reason="Works well. Entry after 40min allows 20min convergence window."
            ),
            StrategyRecommendation(
                MarketType.FOUR_HOUR,
                recommended=False,
                reason="Long windows reduce edge - more time for reversals. Consider latency_arb instead."
            ),
            StrategyRecommendation(
                MarketType.DAILY,
                recommended=False,
                reason="Too much time for external events to affect outcome. Not recommended."
            ),
        ]

    def on_tick(self, event: MarketEvent) -> Optional[Signal]:
        """
        Check if sniper entry conditions are met.

        Called every second. Returns Signal if conditions are met, None otherwise.
        """
        if not self.sniper_config.enabled:
            return None

        if event.symbol not in self.sniper_config.markets:
            return None

        if event.elapsed_sec < self.sniper_config.min_elapsed_sec:
            return None

        # Check if we already took a position for this market window
        if self.has_position(event.symbol, event.market_start):
            return None

        # Check if either side meets the price range and EV threshold
        for side, price in [("UP", event.up_price), ("DOWN", event.down_price)]:
            if price >= self.sniper_config.min_price and price <= self.sniper_config.max_price:
                # Calculate EV for this entry
                ev_info = self.calculate_ev(price, self.sniper_config.assumed_win_rate)
                if ev_info["ev_pct"] >= self.sniper_config.min_ev_pct:
                    logger.info(
                        f"[Sniper] {event.symbol}: SIGNAL {side} @ {price:.1%} | EV={ev_info['ev_pct']:+.1f}%"
                    )

                    return Signal(
                        strategy=self.name,
                        symbol=event.symbol,
                        side=side,
                        entry_price=price,
                        reason=f"Sniper entry: {price:.1%} with EV {ev_info['ev_pct']:+.1f}%",
                        confidence=min(0.95, price),  # High confidence for high probability
                        metadata={
                            "ev_info": ev_info,
                            "elapsed_sec": event.elapsed_sec,
                            "min_elapsed_sec": self.sniper_config.min_elapsed_sec,
                        },
                        market_start=event.market_start,
                        market_end=event.market_end,
                        timestamp=event.timestamp,
                    )

        return None

    def get_status(self, event: MarketEvent) -> dict:
        """
        Get current strategy status for UI display.

        Called frequently to update the UI with strategy state.
        """
        if not self.sniper_config.enabled:
            return {
                "status": "disabled",
                "reason": "Strategy disabled",
                "symbol": event.symbol,
            }

        if event.symbol not in self.sniper_config.markets:
            return {
                "status": "skip",
                "reason": f"Not in markets {self.sniper_config.markets}",
                "symbol": event.symbol,
            }

        if event.elapsed_sec < self.sniper_config.min_elapsed_sec:
            remaining_wait = self.sniper_config.min_elapsed_sec - event.elapsed_sec
            return {
                "status": "waiting",
                "reason": f"Waiting {remaining_wait}s more",
                "symbol": event.symbol,
                "elapsed_sec": event.elapsed_sec,
                "min_elapsed_sec": self.sniper_config.min_elapsed_sec,
                "time_remaining": remaining_wait,
            }

        if self.has_position(event.symbol, event.market_start):
            return {
                "status": "position_taken",
                "reason": "Already have position",
                "symbol": event.symbol,
            }

        # Evaluate both sides
        evaluations = []
        for side, price in [("UP", event.up_price), ("DOWN", event.down_price)]:
            ev_info = self.calculate_ev(price, self.sniper_config.assumed_win_rate)
            eval_data = {
                "side": side,
                "price": price,
                "ev_pct": ev_info["ev_pct"],
                "in_range": self.sniper_config.min_price <= price <= self.sniper_config.max_price,
                "ev_ok": ev_info["ev_pct"] >= self.sniper_config.min_ev_pct,
            }
            evaluations.append(eval_data)

            # Check if this would trigger
            if eval_data["in_range"] and eval_data["ev_ok"]:
                return {
                    "status": "ready",
                    "reason": f"Would signal {side}",
                    "symbol": event.symbol,
                    "signal": side,
                    "entry_price": price,
                    "ev_pct": ev_info["ev_pct"],
                    "elapsed_sec": event.elapsed_sec,
                    "evaluations": evaluations,
                }

        # No signal - explain why
        best_eval = max(evaluations, key=lambda e: e["ev_pct"])
        if not best_eval["in_range"]:
            if best_eval["price"] < self.sniper_config.min_price:
                reason = f"Best price {best_eval['price']:.1%} < min {self.sniper_config.min_price:.1%}"
            else:
                reason = f"Best price {best_eval['price']:.1%} > max {self.sniper_config.max_price:.1%}"
        else:
            reason = f"EV {best_eval['ev_pct']:+.1f}% < min {self.sniper_config.min_ev_pct}%"

        return {
            "status": "no_signal",
            "reason": reason,
            "symbol": event.symbol,
            "elapsed_sec": event.elapsed_sec,
            "evaluations": evaluations,
        }

    def calculate_ev(
        self,
        entry_price: float,
        assumed_win_rate: Optional[float] = None
    ) -> dict:
        """
        Calculate expected value as a percentage.

        Args:
            entry_price: Price paid (e.g., 0.75 for 75c)
            assumed_win_rate: Override win rate estimate (default: use entry_price)

        Returns:
            dict with:
                - ev_pct: EV as percentage of investment
                - net_win: Profit per $1 if correct (after fees)
                - loss: Loss per $1 if incorrect
                - breakeven_win_rate: Win rate needed to break even
        """
        # Use entry price as win rate estimate if not provided
        win_rate = assumed_win_rate if assumed_win_rate is not None else entry_price

        # Calculate outcomes
        # If win: get $1, profit = 1 - entry_price, fee = profit * fee_rate
        profit = 1.0 - entry_price
        fee = profit * self.sniper_config.fee_rate
        net_win = profit - fee  # Net profit if correct

        # If lose: lose entry_price
        loss = entry_price

        # Expected value
        ev = win_rate * net_win - (1 - win_rate) * loss

        # EV as percentage of investment
        ev_pct = (ev / entry_price) * 100.0 if entry_price > 0 else 0.0

        # Breakeven win rate
        breakeven = loss / (net_win + loss) if (net_win + loss) > 0 else 1.0

        return {
            "ev_pct": round(ev_pct, 2),
            "net_win": round(net_win, 4),
            "loss": round(loss, 4),
            "breakeven_win_rate": round(breakeven * 100, 1),
            "entry_price": entry_price,
            "assumed_win_rate": round(win_rate * 100, 1),
            "fee_rate": self.sniper_config.fee_rate * 100
        }

    def get_position_size(self, account_balance: float) -> float:
        """Calculate position size based on config."""
        pct_based = account_balance * (self.sniper_config.position_size_pct / 100.0)
        return min(pct_based, self.sniper_config.max_position_usd)

    def validate_retry(
        self,
        side: str,
        current_price: float,
        elapsed_sec: int,
        original_price: float,
        max_price_move_pct: float = 5.0
    ) -> tuple[bool, str, Optional[dict]]:
        """
        Validate if a retry is still valid after an order failure.

        Args:
            side: "UP" or "DOWN" - the original signal side
            current_price: Current market price for that side
            elapsed_sec: Current elapsed seconds in the window
            original_price: Price when original signal was generated
            max_price_move_pct: Max acceptable price move for retry

        Returns:
            Tuple of (is_valid, reason, ev_info_if_valid)
        """
        if not self.sniper_config.enabled:
            return (False, "Strategy disabled", None)

        # Check if we've run out of time (within 30 seconds of market end)
        remaining = 900 - elapsed_sec
        if remaining < 30:
            return (False, f"Only {remaining}s remaining - too late", None)

        # Check if price is still in valid range
        if current_price < self.sniper_config.min_price:
            return (False, f"Price dropped below min ({current_price:.1%} < {self.sniper_config.min_price:.1%})", None)

        if current_price > self.sniper_config.max_price:
            return (False, f"Price exceeded max ({current_price:.1%} > {self.sniper_config.max_price:.1%})", None)

        # Check price hasn't moved too much from original signal
        price_move = abs(current_price - original_price) / original_price * 100
        if price_move > max_price_move_pct:
            return (False, f"Price moved {price_move:.1f}% from original - signal stale", None)

        # Check EV is still valid
        ev_info = self.calculate_ev(current_price, self.sniper_config.assumed_win_rate)
        if ev_info["ev_pct"] < self.sniper_config.min_ev_pct:
            return (False, f"EV dropped below min ({ev_info['ev_pct']:.1f}% < {self.sniper_config.min_ev_pct}%)", None)

        return (True, "Signal still valid", ev_info)
