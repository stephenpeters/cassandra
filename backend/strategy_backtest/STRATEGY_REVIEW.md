# Request for Strategy Development Help

## The Problem

I'm building a trading system for **Polymarket's 15-minute crypto prediction markets**. These are binary markets that resolve based on whether BTC/ETH/SOL/etc goes UP or DOWN over a 15-minute window.

**Goal**: Develop a profitable automated trading strategy.

**Benchmark**: Top traders on the platform:
- **gabagool22**: +$529K lifetime, 62% win rate on BTC 15-min
- **Account88888**: +$446K lifetime, 45% win rate on BTC 15-min

I've collected data, backtested strategies, and measured latencies but haven't achieved consistent profitability. **I need fresh perspectives on what I might be missing.**

---

## Data Available

### Collected Dataset
- **20 BTC 15-minute markets** (January 5-6, 2026)
- **80,097 trades** with full details (wallet, size, price, timestamp, outcome)
- **Binance 1-second OHLCV** for each market window
- **Whale trade data** for Account88888, gabagool22, updateupdate

### Data Structure
Each market file contains:
```json
{
  "slug": "btc-updown-15m-1767640500",
  "trades": [{
    "proxyWallet": "0x...",
    "side": "BUY",
    "size": 69.68,
    "price": 0.50,
    "timestamp": 1767642237,
    "outcome": "Up"
  }],
  "binance": {
    "open": 98234.50,
    "close": 98312.00,
    "actual_resolution": "UP"
  }
}
```

---

## Strategies Tried

### Strategy 1: Latency Arbitrage

**Hypothesis**: Polymarket prices lag Binance by 30-60 seconds. Buy when Binance moves but Polymarket hasn't adjusted.

**Implementation**:
1. Record Binance price at window open
2. Monitor price movement every second
3. Calculate implied UP probability from % move
4. Compare to Polymarket price to find edge
5. Enter when edge > 5%

**Results**: ❌ **Negative**
- Most windows show "NO LATENCY GAP"
- When gaps exist, they're often < 5%
- Edge gets eroded by execution latency
- Win rate: ~45% (net negative after slippage)

**What might be wrong**:
- 5% edge threshold too high?
- Fixed checkpoints (3m, 6m, 9m) don't adapt to market conditions?
- Should I be looking at order flow instead of just price?

---

### Strategy 2: Whale Following

**Hypothesis**: Follow profitable traders' positions. They have edge, piggyback on it.

**Implementation**:
1. Track wallet addresses of top traders
2. Detect their trades via Polymarket API
3. Infer their directional bias (net UP or DOWN position)
4. Enter same direction before market closes

**Latency Measurements**:
| Detection Method | Latency | Source |
|------------------|---------|--------|
| API Polling (60s interval) | ~39 seconds | Measured |
| API Polling (2s interval) | ~1.2 seconds | Measured |
| WebSocket (prices only) | ~25ms | Measured, but NO wallet data |

**Backtest Results** (20 markets):

| Configuration | Win Rate | ROI |
|---------------|----------|-----|
| Follow ALL whales, 5s latency | 37% | **-9.9%** |
| Follow gabagool22 + Account88888 only, 5s | 53% | **+2.6%** |
| Follow gabagool22 + Account88888 only, 2s | 58% | **+13.3%** |

**Per-Whale Accuracy** (critical finding):
| Whale | Trades | Win Rate | Notes |
|-------|--------|----------|-------|
| gabagool22 | 8 | **62%** | Highly predictive |
| Account88888 | 11 | 45% | Volume contributor |
| updateupdate | 11 | **18%** | TERRIBLE - must exclude |

**Results**: ⚠️ **Marginally Positive** with correct whale selection

**Issues**:
- WebSocket provides 25ms latency but NO wallet addresses
- Must poll Trades API which adds ~1-2s latency
- Small sample size (20 markets, 19 whale trades)
- Not sure if +2.6% ROI is statistically significant

---

### Strategy 3: Trader Timing Analysis

**Hypothesis**: Top traders enter at specific times in the window. Copy their timing.

**Findings from charts**:
- **Account88888**: Heavy trading in first 3 minutes, large sizes ($500-2000)
- **gabagool22**: Distributed entries, smaller sizes, averages in
- **updateupdate**: Later entries (after 6 min), follows momentum

**Results**: 📊 **Informational only** - not actionable yet

---

## What's Not Working

1. **Latency arbitrage fails** - gaps are too small or close too fast
2. **Whale following works but barely** - +2.6% ROI with optimal settings
3. **Can't get wallet data from WebSocket** - forced to poll at 1-2s intervals
4. **Small sample size** - 20 markets isn't enough to validate strategies
5. **No pre-market data** - missing order flow before window opens

---

## Specific Questions I Need Help With

### Q1: Is there alpha in this market?
The top traders make consistent profits. Is it:
- Superior information (they know something we don't)?
- Better execution (they get better prices)?
- Just variance and they'll regress to the mean?

### Q2: What features should I be looking at?
Currently tracking:
- Binance price momentum
- Polymarket volume delta (buy vs sell)
- Order book imbalance
- Whale positions

What am I missing?

### Q3: How do I get faster whale detection?
WebSocket is fast (25ms) but doesn't include wallet addresses.
Trades API has wallets but is slow (1-2s polling).

Is there a way to get real-time wallet data? Or do I need to:
- Build a custom indexer?
- Use blockchain data directly?
- Accept the 1-2s latency?

### Q4: Position sizing and risk management?
Currently using fixed 5% position size. Should I:
- Use Kelly criterion based on estimated edge?
- Scale based on signal strength?
- Limit correlated positions (BTC + ETH moving together)?

### Q5: Is 20 markets enough to validate?
With 19 whale trades at 53% win rate, is that statistically significant?
What sample size do I need to be confident in the strategy?

---

## File Structure

All strategy materials are in: `backend/strategy_backtest/`

```
strategy_backtest/
├── STRATEGY_REVIEW.md          # This document
├── data/
│   ├── markets/                # 20 market JSON files with trades
│   ├── binance_prices.json     # Binance data for each market
│   └── gabagool_positions.json # Whale position data
├── exports/
│   ├── btc_trades_all.json     # Consolidated 80K trades
│   └── btc_binance_1s.csv      # 1-second Binance OHLCV
├── analysis/
│   └── timing_*.png            # Trader timing charts
├── results/
│   └── backtest_results.json   # Backtest output
├── collect_20_markets.py       # Data collection script
├── whale_following_backtest.py # Whale strategy backtest
├── trader_timing_analysis.py   # Timing chart generator
├── measure_ws_latency.py       # Latency measurement script
└── backfill_binance_data.py    # Binance data fetcher
```

---

## Summary of Results

| Strategy | Status | Best ROI | Notes |
|----------|--------|----------|-------|
| Latency Arbitrage | ❌ Failed | Negative | Gaps too small |
| Whale Following | ⚠️ Marginal | +2.6% | Needs more data |
| Timing Pattern | 📊 Research | N/A | Not implemented |

**Bottom line**: I have infrastructure and data but no consistently profitable strategy. Looking for guidance on what to try next or what I'm fundamentally missing.

---

*Updated: January 7, 2026*
