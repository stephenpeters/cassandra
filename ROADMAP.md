# Polymarket Whale Tracker - Product Roadmap

## V1 - Current Features (Production Ready)

### Core Trading System
- **15-minute crypto prediction markets** (BTC, ETH, SOL, XRP, DOGE)
- **Paper trading engine** with simulated execution
- **Live trading engine** with Polymarket CLOB integration
- **Dynamic paper/live mode toggle** via API (no restart required)
- **Kill switch** for emergency trading halt
- **Circuit breaker** for automated risk management

### Data Feeds
- **Binance WebSocket** - Real-time price, volume, order book (100ms updates)
- **Polymarket WebSocket** - Real-time market prices
- **Polymarket polling** - Trade discovery, market resolution

### Whale Tracking
- **gabagool22** wallet monitoring (0x6031...)
- **Theo cluster** detection
- **WebSocket-based detection** (5-15s latency)
- **Polling-based detection** (39s latency)

### Trading Signals
- **Checkpoint-based signals** at 3m, 6m, 7.5m, 9m, 12m
- **Latency arbitrage detection** - Binance vs Polymarket price gaps
- **Momentum indicators** - Volume delta, order book imbalance
- **Confidence scoring** with configurable thresholds

### User Interface
- **TradingView-style OHLCV charts** (lightweight-charts)
- **Real-time streaming price chart** (recharts)
- **Paper trading card** - Balance, P&L, positions, trades
- **Market analysis panel** - Momentum, confidence, order book
- **Whale sentiment display** - Recent whale trades
- **Market countdown timer** - Time to next window

### Alerts & Notifications
- **Telegram alerts** - Signals, trades, end-of-market summaries
- **Discord webhook support**
- **Account balance alerts**

### Persistence & Logging
- **SQLite trade ledger** - Full audit trail
- **JSON state files** - Paper trading state, historical markets
- **Daily log files** - Trade logs, live trading logs

### Deployment
- **Systemd service** with auto-restart
- **Nginx reverse proxy** with TLS 1.2+
- **Health check endpoint** with component status
- **AWS Lightsail / Vultr VPS** deployment guides

---

## V1.1 - Production Hardening (Pre-Launch)

| Item | Status | Description |
|------|--------|-------------|
| Automated S3 Backups | ✅ Done | Daily backup of trades.db, state files |
| API Key Security | ✅ Done | Require API_KEY in production, exit if not set |
| Graceful Shutdown | ✅ Done | Kill switch activation on SIGTERM, save state |
| WebSocket Reconnection | ✅ Done | Exponential backoff with jitter for Binance |
| CloudWatch Monitoring | Planned | Alarms for service health, disk, memory |

---

## V2 - Multi-Market & Enhanced UI (Roadmap)

### Market Expansion

**Target Markets (when available on Polymarket):**

| Timeframe | Slug Pattern | Duration | Markets/Day/Symbol |
|-----------|--------------|----------|-------------------|
| 15-min | `{sym}-updown-15m-{ts}` | 900s | 96 |
| 1-hour | `{sym}-updown-1h-{ts}` | 3600s | 24 |
| 4-hour | `{sym}-updown-4h-{ts}` | 14400s | 6 |
| Daily | `{sym}-updown-1d-{ts}` | 86400s | 1 |

**Supported Symbols:** BTC, ETH (primary), SOL, XRP, DOGE (secondary)

**Total Markets/Day (BTC + ETH):** 254 when all timeframes available

### Phase 1: Trading Card Redesign

- **Rename to "Trading"** with PAPER/LIVE mode indicator (RED banner)
- **Flip digit countdown timer** (react-flip-numbers, PM-style)
- **Remove recent trades**, keep open positions
- **Collapsible analysis panels** (confidence, market pressure, order book)
- **Vertical market pressure bar** (more compact)
- **Show UP/DOWN market prices** instead of non-functional controls
- **Tooltips on all controls and data widgets**
- **Kill switch button in UI** - Emergency halt trading from frontend
- **Telegram kill switch** - Reply `/kill` to any alert to halt trading

### Phase 2: Multi-Market Support

- **Market configuration system** - Enable/disable symbols and timeframes
- **Simple market selector in Trading Settings** - Intuitive Symbol x Timeframe grid
  - Easy toggle for each market combination
  - Visual indication of available vs unavailable markets
  - Settings persist across sessions
- **Multi-market trading card** - Tab bar to switch between markets
- **Aggregated P&L** across all active markets

### Phase 3: Rolling 24-Hour Data Collection

**Collection Intervals:**

| Market Type | Binance Interval | PM Trade Poll | Storage |
|-------------|------------------|---------------|---------|
| 15-min | 1s WebSocket | 1s | SQLite -> S3 |
| 1-hour | 1m klines | 10s | SQLite -> S3 |
| 4-hour | Aggregate 1h | N/A | Derived |
| Daily | Aggregate 1h | N/A | Derived |

**Features:**
- Rolling 24-hour retention in SQLite
- S3 archiving for data older than 24h
- Price snapshots table (Binance + PM prices)
- Market trades table (all trades per market)

### Phase 4: Backend API Extensions

**New Endpoints:**
- `GET /api/markets/config` - Get enabled symbols and market types
- `POST /api/markets/config` - Update market configuration
- `GET /api/markets/available` - Check which markets exist on Polymarket
- `GET /api/history/snapshots` - Rolling price snapshot data
- `GET /api/history/trades` - Trades for a specific market

---

## Implementation Timeline

1. **V1.1 Production Hardening** - On `main` branch before deployment
2. **V2 Development** - On `feature/v2-multi-market` branch
3. **V2 Release** - Merge to main after testing

---

## Technical Notes

- Hourly, 4-hour, and daily markets do not exist yet on Polymarket
- System designed to support them when launched
- 4-hour and daily data derived from hourly candles (no separate collection)
- Multi-market UI ready for future timeframes

---

## Strategy Ideas (Backlog)

### Market Maker / Liquidity Rebates Strategy

**Source:** Twitter observation - 44% profit in one day

**Concept:** Exploit Polymarket's "Maker Rebates Program"

**How It Works:**
1. 15-minute crypto markets charge a fee from takers (up to 1.5%)
2. The fee fully redistributes from takers to makers
3. To become a "maker" you place limit orders (provide liquidity)

**Strategy Details:**
- Place limit orders using a **delta-neutral strategy**
- Mirror positions on YES and NO (not dependent on market outcome)
- Example: If placing YES at 50%, also place NO at 49-50%
- Reuse same liquidity: bet → claim → bet again (every 15 min)

**Reported Results:**
- $60 initial capital
- $7 earned on spreads
- $19 earned on liquidity rewards
- Total: $86 (44% growth in one day)
- ~1.5% profit from taker fees every 15 minutes

**Implementation Notes:**
- Need to calculate optimal spread width
- Must handle order placement timing
- Consider slippage and fill rates
- May compete with other market makers
- Need to verify actual rebate rates and mechanics

**Risks:**
- Spread may not fill on both sides
- Rapid price moves could leave one side unfilled
- Other market makers may offer tighter spreads
- Capital locked during waiting for fills

---

### Volatility Farming / Spread Capture Strategy

**Source:** Analysis of @alliswell's Polymarket account ($227K+ profits)
**Account:** https://polymarket.com/@alliswell

**Concept:** Exploit micro-inefficiencies in real-time order flow

**Key Mechanics:**
- Algorithmic ladder orders: sell at 77c, buy back at 73-74c within minutes
- Position sizes scale: $0.37 to $3K depending on spread width and book depth
- Not about predicting outcomes - pure spread capture

**Stats:**
- Trades ~40x per hour
- Most positions under $50
- Biggest single win: $137K (covers 500+ micro-losses)
- Hit rate doesn't matter when winners are asymmetric

**Infrastructure Required:**
- WebSocket feeds monitoring order book changes in milliseconds
- Automated position sizing that adjusts to volatility
- Sub-second execution when spreads dislocate
- 24/7 uptime without breaking

**Implementation Notes:**
- Need real-time order book depth analysis
- Calculate optimal spread widths dynamically
- Scale position size based on liquidity available
- Handle rapid position cycling (buy-claim-bet again)
- Asymmetric risk: small losses, large wins

**Risks:**
- Infrastructure complexity (24/7 reliability)
- Execution risk in fast-moving markets
- Capital efficiency during position cycling
- API rate limits and websocket stability

---

## Cleanup & Refactoring (Backlog)

| Item | Priority | Description |
|------|----------|-------------|
| Rename "whale trades" to "copy trades" | Low | Update all references to whale trades terminology across frontend, backend, and docs. Affects ~16 files including: WhaleTradesTable.tsx, whale_following.py, useWebSocket.ts, FollowingModal.tsx, etc. |
