"""
Trading Strategies Package

All strategies inherit from strategy_base.Strategy and ONLY produce signals.
Execution is handled by the Executor layer.
"""

from .sniper import SniperStrategy, SniperConfig

# Future strategies will be added here:
# from .dip_arb import DipArbStrategy, DipArbConfig
# from .latency_arb import LatencyArbStrategy, LatencyArbConfig
# from .market_maker import MarketMakerStrategy, MarketMakerConfig

__all__ = [
    "SniperStrategy",
    "SniperConfig",
]
