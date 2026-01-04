# Product Requirements Document (PRD)
## Prediction Market Arbitrage System

**Version:** 1.0
**Date:** December 2024
**Author:** [Product Team]
**Status:** Draft

---

## Executive Summary

This document outlines the requirements for an automated prediction market arbitrage system that operates 24/7 to identify and execute risk-free profit opportunities across multiple prediction market platforms. The system will scan for pricing inefficiencies where the combined cost of complementary positions is less than the guaranteed payout, executing trades automatically within defined parameters.

**Target Opportunity:** $40M+ in arbitrage profits were extracted from prediction markets between April 2024 - April 2025. The top performer generated $2.01M across 4,049 transactions.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Goals & Success Metrics](#2-goals--success-metrics)
3. [Target Users](#3-target-users)
4. [Scope](#4-scope)
5. [System Architecture](#5-system-architecture)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Arbitrage Strategies](#8-arbitrage-strategies)
9. [Platform Integrations](#9-platform-integrations)
10. [Risk Management](#10-risk-management)
11. [Trading Parameters](#11-trading-parameters)
12. [User Interface](#12-user-interface)
13. [Data & Analytics](#13-data--analytics)
14. [Security Requirements](#14-security-requirements)
15. [Compliance & Legal](#15-compliance--legal)
16. [Release Phases](#16-release-phases)
17. [Open Questions](#17-open-questions)

---

## 1. Problem Statement

### Current State
Prediction markets (Polymarket, Kalshi) frequently exhibit pricing inefficiencies where:
- YES + NO prices sum to less than $1.00 (guaranteed arbitrage)
- Multi-outcome markets have all options sum to less than $1.00
- Cross-platform prices diverge for identical events
- Related/correlated markets become temporarily mispriced

These opportunities exist for seconds to minutes before being captured by sophisticated traders.

### Challenges
- Manual trading cannot capture fleeting opportunities
- 24/7 markets require constant monitoring
- Multi-leg trades require simultaneous execution
- Fee structures and slippage can eliminate profits
- Competition from HFT bots compresses margins

### Opportunity
Build an automated system that:
- Monitors markets continuously (24/7/365)
- Identifies arbitrage opportunities in real-time
- Executes trades within milliseconds
- Manages risk through configurable parameters
- Scales across multiple platforms and strategies

---

## 2. Goals & Success Metrics

### Primary Goals

| Goal | Target | Timeframe |
|------|--------|-----------|
| Capture arbitrage opportunities | >80% of identified opportunities executed | Ongoing |
| Maintain positive PnL | >95% of trades profitable after fees | Monthly |
| System uptime | 99.9% availability | Ongoing |
| Execution speed | <500ms from detection to order | Per trade |

### Success Metrics

**Financial Metrics:**
- Total profit generated (target: $10K+ monthly in Phase 1)
- Average profit per trade (target: >1% net after fees)
- Win rate (target: >95%)
- Sharpe ratio of returns

**Operational Metrics:**
- Opportunities identified per day
- Opportunities executed vs. missed
- Average execution latency
- System uptime percentage

**Risk Metrics:**
- Maximum drawdown
- Failed trade rate
- Leg risk incidents (partial fills)
- Capital utilization efficiency

---

## 3. Target Users

### Primary User: System Operator
- Monitors system performance via dashboard
- Configures trading parameters
- Manages capital allocation
- Reviews and approves strategy changes

### Secondary User: Developer/Maintainer
- Deploys and maintains infrastructure
- Adds new strategies and platform integrations
- Optimizes execution performance
- Troubleshoots issues

---

## 4. Scope

### In Scope (Phase 1)

| Feature | Priority |
|---------|----------|
| Polymarket API integration | P0 |
| Kalshi API integration | P0 |
| InMarket arbitrage (YES+NO < $1) | P0 |
| Multi-outcome arbitrage (Σ YES < $1) | P0 |
| Real-time WebSocket monitoring | P0 |
| Automated trade execution | P0 |
| Risk management guardrails | P0 |
| Basic monitoring dashboard | P1 |
| Alerting system | P1 |

### In Scope (Phase 2)

| Feature | Priority |
|---------|----------|
| Cross-platform arbitrage (Polymarket ↔ Kalshi) | P1 |
| Negative risk strategy | P1 |
| Combinatorial arbitrage (related markets) | P2 |
| PredictIt price monitoring (read-only) | P2 |
| Advanced analytics dashboard | P2 |

### In Scope (Phase 3)

| Feature | Priority |
|---------|----------|
| News sniping / event-driven trading | P2 |
| ML-based opportunity prediction | P3 |
| Correlation arbitrage | P3 |
| Market making capabilities | P3 |

### Out of Scope
- Mobile application
- Multi-user support
- Social/copy trading features
- Manual trading interface
- Robinhood integration (no API)
- Crypto.com Sports integration (no API)

---

## 5. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MONITORING LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Polymarket │  │   Kalshi    │  │  PredictIt  │              │
│  │  WebSocket  │  │  WebSocket  │  │  REST Poll  │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │   Market Data Store   │                          │
│              │   (Order Books, Prices)│                          │
│              └───────────┬───────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  ARBITRAGE DETECTOR   │                          │
│              ├───────────────────────┤                          │
│              │ • InMarket Scanner    │                          │
│              │ • Multi-Outcome Scan  │                          │
│              │ • Cross-Platform Scan │                          │
│              │ • Negative Risk Scan  │                          │
│              │ • Combinatorial Scan  │                          │
│              └───────────┬───────────┘                          │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  OPPORTUNITY QUEUE    │                          │
│              └───────────┬───────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │   RISK VALIDATOR      │                          │
│              ├───────────────────────┤                          │
│              │ • Min spread check    │                          │
│              │ • Liquidity check     │                          │
│              │ • Position limit check│                          │
│              │ • Capital check       │                          │
│              │ • Fee calculation     │                          │
│              └───────────┬───────────┘                          │
│                          │                                       │
│                   Pass   │   Fail → Log & Skip                   │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  EXECUTION ENGINE     │                          │
│              ├───────────────────────┤                          │
│              │ • Order Builder       │                          │
│              │ • Atomic Executor     │                          │
│              │ • Timeout Handler     │                          │
│              │ • Partial Fill Mgmt   │                          │
│              └───────────┬───────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │   POSITION MANAGER    │                          │
│              ├───────────────────────┤                          │
│              │ • Track open positions│                          │
│              │ • Monitor resolution  │                          │
│              │ • Calculate PnL       │                          │
│              │ • Rebalance capital   │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                          ▼                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Dashboard  │  │   Alerts    │  │   Logging   │              │
│  │    (Web)    │  │  (Discord/  │  │ (TimeSeries │              │
│  │             │  │   Telegram) │  │     DB)     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                     OBSERVABILITY LAYER                          │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Runtime | Python 3.11+ | Official SDKs available, async support |
| WebSocket | `websockets` / `aiohttp` | Async real-time data |
| Database | PostgreSQL + TimescaleDB | Time-series data, reliability |
| Cache | Redis | Order book snapshots, rate limiting |
| Queue | Redis Streams / RabbitMQ | Opportunity processing |
| Dashboard | Streamlit / Grafana | Rapid development |
| Alerting | Discord/Telegram webhooks | Real-time notifications |
| Hosting | AWS / GCP (low-latency region) | Proximity to platform servers |
| Secrets | AWS Secrets Manager / Vault | Secure key storage |

---

## 6. Functional Requirements

### FR-1: Market Data Ingestion

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | Connect to Polymarket CLOB WebSocket for real-time order book updates | P0 |
| FR-1.2 | Connect to Kalshi WebSocket for real-time order book updates | P0 |
| FR-1.3 | Poll PredictIt REST API every 60 seconds for price data | P2 |
| FR-1.4 | Maintain local order book state for each monitored market | P0 |
| FR-1.5 | Handle WebSocket disconnections with automatic reconnection | P0 |
| FR-1.6 | Store historical price data for analysis | P1 |

### FR-2: Arbitrage Detection

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | Scan for YES + NO < $1.00 opportunities (binary markets) | P0 |
| FR-2.2 | Scan for Σ(all YES) < $1.00 opportunities (multi-outcome) | P0 |
| FR-2.3 | Scan for Σ(all NO) < $(n-1) opportunities (negative risk) | P1 |
| FR-2.4 | Scan for cross-platform price discrepancies | P1 |
| FR-2.5 | Identify related markets for combinatorial arbitrage | P2 |
| FR-2.6 | Calculate net profit after all fees for each opportunity | P0 |
| FR-2.7 | Filter opportunities below minimum profit threshold | P0 |

### FR-3: Trade Execution

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | Build properly formatted orders for Polymarket CLOB | P0 |
| FR-3.2 | Build properly formatted orders for Kalshi API | P0 |
| FR-3.3 | Execute multi-leg trades atomically (all-or-none when possible) | P0 |
| FR-3.4 | Implement timeout protection (cancel after 5 seconds) | P0 |
| FR-3.5 | Handle partial fills appropriately | P0 |
| FR-3.6 | Retry failed orders with backoff | P1 |
| FR-3.7 | Support limit orders only (no market orders) | P0 |

### FR-4: Position Management

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | Track all open positions across platforms | P0 |
| FR-4.2 | Calculate unrealized and realized PnL | P0 |
| FR-4.3 | Monitor position resolution and settlement | P0 |
| FR-4.4 | Maintain capital allocation per platform | P0 |
| FR-4.5 | Support manual position closure | P1 |

### FR-5: Risk Management

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | Enforce minimum spread threshold per trade | P0 |
| FR-5.2 | Enforce maximum position size per market | P0 |
| FR-5.3 | Enforce maximum capital per single opportunity | P0 |
| FR-5.4 | Check order book liquidity before execution | P0 |
| FR-5.5 | Implement daily loss limit (circuit breaker) | P0 |
| FR-5.6 | Block trading during high-volatility events (optional) | P2 |

### FR-6: Alerting & Notifications

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-6.1 | Send alert on successful trade execution | P1 |
| FR-6.2 | Send alert on trade failure | P0 |
| FR-6.3 | Send alert on system error or downtime | P0 |
| FR-6.4 | Send daily PnL summary | P1 |
| FR-6.5 | Send alert when position resolves | P1 |

---

## 7. Non-Functional Requirements

### NFR-1: Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1.1 | Order book update processing latency | <50ms |
| NFR-1.2 | Arbitrage detection latency | <100ms |
| NFR-1.3 | End-to-end execution (detect → order sent) | <500ms |
| NFR-1.4 | Concurrent market monitoring | 500+ markets |
| NFR-1.5 | WebSocket message throughput | 1000+ msg/sec |

### NFR-2: Reliability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.1 | System uptime | 99.9% |
| NFR-2.2 | Automatic recovery from crashes | <60 seconds |
| NFR-2.3 | Data persistence (no loss on restart) | 100% |
| NFR-2.4 | Graceful degradation on API errors | Required |

### NFR-3: Scalability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-3.1 | Add new platform integration | <1 week |
| NFR-3.2 | Add new arbitrage strategy | <3 days |
| NFR-3.3 | Handle 10x market volume | No degradation |

### NFR-4: Security

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-4.1 | API keys encrypted at rest | Required |
| NFR-4.2 | No secrets in code or logs | Required |
| NFR-4.3 | Audit logging of all trades | Required |
| NFR-4.4 | Rate limiting compliance | Required |

---

## 8. Arbitrage Strategies

### Strategy 1: Binary InMarket Arbitrage (P0)

**Condition:** `YES_ask + NO_ask < $1.00 - fees`

**Execution:**
1. Detect when YES + NO best asks sum to less than threshold
2. Calculate position size based on available liquidity
3. Place simultaneous limit orders for YES and NO
4. Monitor fills, cancel unfilled after timeout
5. If both fill → guaranteed profit at resolution

**Parameters:**
- `min_spread`: Minimum spread to execute (default: 2.5%)
- `max_position`: Maximum position size (default: $1,000)
- `timeout`: Order timeout in seconds (default: 5)

### Strategy 2: Multi-Outcome Arbitrage (P0)

**Condition:** `Σ(all YES outcomes) < $1.00 - fees`

**Execution:**
1. Calculate sum of best ask prices for all outcomes
2. If sum < threshold, calculate equal position sizes
3. Place limit orders for all outcomes simultaneously
4. Require all fills or cancel all

**Parameters:**
- `min_spread`: Minimum spread (default: 2.0%)
- `max_position`: Per outcome (default: $500)
- `max_outcomes`: Maximum outcomes to trade (default: 10)

### Strategy 3: Negative Risk Arbitrage (P1)

**Condition:** `Σ(all NO outcomes) < $(n-1) - fees`

**Execution:**
1. For n-outcome market, calculate sum of all NO prices
2. If sum < (n-1) - fees, execute
3. Place NO orders on all outcomes
4. Guaranteed (n-1) outcomes pay $1 each

**Parameters:**
- `min_spread`: Minimum spread (default: 2.0%)
- `max_outcomes`: Maximum outcomes (default: 6)

### Strategy 4: Cross-Platform Arbitrage (P1)

**Condition:** `YES(Platform A) + NO(Platform B) < $1.00 - fees`

**Execution:**
1. Match identical markets across platforms
2. Compare YES price on A with NO price on B
3. If combined < threshold, execute on both platforms
4. Higher risk due to different resolution mechanisms

**Parameters:**
- `min_spread`: Higher threshold (default: 5.0%)
- `max_position`: Lower limit (default: $500)
- `market_matching`: Require exact resolution criteria match

### Strategy 5: Combinatorial Arbitrage (P2)

**Condition:** Related markets with logical dependencies mispriced

**Execution:**
1. Identify logically related markets (e.g., "Harris wins" + "GOP margins")
2. Calculate combined probability exposure
3. If arbitrage exists, execute hedge positions
4. Requires careful market relationship mapping

**Parameters:**
- `relationship_map`: Predefined market relationships
- `min_spread`: Higher threshold (default: 3.5%)
- `manual_approval`: Optional human verification

### Strategy 6: Range Coverage / Statistical Arbitrage (P2)

**Condition:** Bucketed/range markets where historical statistics provide predictable distribution

**Example:** Elon Musk Tweet Count Market
```
Historical average: ~43 tweets/day
Period: Dec 23-30 (7 days)
Expected range: 301-343 tweets

Buy 1 share of adjacent ranges:
- 300 range @ $0.11
- 320 range @ $0.14
- 340 range @ $0.16
- 360 range @ $0.14
- 380 range @ $0.13
────────────────────
Total cost: $0.68
Payout if ANY hits: $1.00
Profit: $0.32 (47% return)
```

**Execution:**
1. Identify bucketed markets with predictable behavior (tweets, weather, sports stats)
2. Calculate historical mean and standard deviation
3. Cover ranges within 2-3 standard deviations
4. Ensure total cost < $1.00 with high probability of success
5. Place orders on all covered ranges

**Parameters:**
- `historical_lookback`: Days of data to analyze (default: 30)
- `coverage_std`: Standard deviations to cover (default: 2)
- `max_total_cost`: Maximum combined cost (default: $0.75)
- `min_expected_profit`: Minimum expected return (default: 25%)

**Best Markets:**
- Tweet/post counts (consistent posters)
- Weather ranges (temperature, rainfall)
- Economic indicators (known seasonality)
- Sports statistics (player averages)

**Risk:** Outlier events (account suspension, illness, unusual circumstances)

---

## 9. Platform Integrations

### Polymarket Integration

**Authentication:**
- Wallet-based authentication (private key)
- L2 API credentials for trading
- Signature type: EOA (type 0) for standard wallets, type 1 for Magic/email, type 2 for Gnosis Safe

**Endpoints Used:**
```
Base URL: https://clob.polymarket.com

GET  /book?token_id={id}     # Order book
GET  /markets                 # List markets
POST /order                   # Place order
DELETE /order/{id}           # Cancel order
GET  /orders                  # Get orders
WS   /ws                      # Real-time updates
```

**Rate Limits:**
- ~1,000 calls/hour (free tier)
- WebSocket: Unlimited for subscribed markets

**Fees:**
- US: 0.01% on trades
- International: 2% on net winnings

#### Official GitHub Repositories (Reusable Code)

| Repository | Language | Stars | Use Case |
|------------|----------|-------|----------|
| [py-clob-client](https://github.com/Polymarket/py-clob-client) | Python | 485 | Primary trading client |
| [clob-client](https://github.com/Polymarket/clob-client) | TypeScript | 325 | TS/Node.js integration |
| [rs-clob-client](https://github.com/Polymarket/rs-clob-client) | Rust | - | Infrastructure-grade, 24/7 stability |
| [agents](https://github.com/Polymarket/agents) | Python | 980 | AI agent framework with LLM |
| [ctf-exchange](https://github.com/Polymarket/ctf-exchange) | Solidity | 229 | Smart contract reference |
| [neg-risk-ctf-adapter](https://github.com/Polymarket/neg-risk-ctf-adapter) | Solidity | 49 | Multi-outcome conversions |

**Recommended:** Use `py-clob-client` for Python implementation, `rs-clob-client` for production 24/7 systems.

#### Smart Contracts (On-Chain Atomic Execution)

| Contract | Address (Polygon) | Purpose |
|----------|-------------------|---------|
| CTF Exchange | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` | Binary YES/NO atomic swaps |
| NegRisk CTFExchange | `0x5b049...2016a` | Multi-outcome market exchange |
| NegRiskAdapter | `0x39b3e...b83c2` | NO→YES position conversion |
| UMA Oracle | `0x5022...6e474` | Decentralized resolution |
| USDC | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` | Collateral token |

#### Atomic Execution (Eliminates Leg Risk)

The CTF Exchange supports **MINT** and **MERGE** operations that enable atomic execution:

**MINT Operation:** When buying YES and NO simultaneously:
```
Maker: Buy YES @ $0.50 (offers USDC)
Taker: Buy NO @ $0.50 (offers USDC)
→ Exchange mints YES+NO pair from combined collateral
→ Distributes to each party atomically
```

**MERGE Operation:** When selling both positions:
```
Hold: YES + NO tokens
→ Exchange merges into $1 USDC
→ Single atomic transaction
```

**Order Types:**
- `GTC` (Good-Till-Cancelled): Standard limit order
- `FOK` (Fill-or-Kill): Must fill entirely or cancel immediately - **use for arbitrage**

**FOK Precision Limits:**
- Maker amount: 2 decimal places
- Taker amount: 4 decimal places

#### py-clob-client Example

```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

# Initialize
client = ClobClient(
    "https://clob.polymarket.com",
    key=PRIVATE_KEY,
    chain_id=137,
    signature_type=0,  # EOA wallet
    funder=FUNDER_ADDRESS
)
client.set_api_creds(client.create_or_derive_api_creds())

# Get order book
book = client.get_order_book(token_id="<token_id>")

# Place FOK order (atomic - fills entirely or cancels)
mo = MarketOrderArgs(
    token_id="<token_id>",
    amount=100.0,
    side=BUY,
    order_type=OrderType.FOK
)
signed = client.create_market_order(mo)
resp = client.post_order(signed, OrderType.FOK)
```

---

### Kalshi Integration

**Authentication:**
- RSA-PSS signed requests
- API key + private key
- Token refresh every 30 minutes

**Headers:**
```
KALSHI-ACCESS-KEY: <key_id>
KALSHI-ACCESS-TIMESTAMP: <ms_timestamp>
KALSHI-ACCESS-SIGNATURE: <signature>
```

**Endpoints Used:**
```
Base URL: https://trading-api.kalshi.com/trade-api/v2

GET  /markets/{ticker}/orderbook  # Order book
GET  /markets                      # List markets
POST /orders                       # Place order
DELETE /orders/{id}               # Cancel order
GET  /portfolio/positions         # Positions
WS   /trade-api/ws/v2            # Real-time
```

**Rate Limits:**
- Basic: 20 req/sec
- Premier: 100 req/sec

**Fees:**
- ~0.7% on trades

---

### PredictIt Integration (Read-Only)

**Endpoints:**
```
GET https://www.predictit.org/api/marketdata/all/
GET https://www.predictit.org/api/marketdata/markets/{id}
```

**Limitations:**
- Update frequency: 60 seconds
- No trading API (TOS prohibits automation)
- Use for price comparison only

---

## 10. Risk Management

### Position Limits

| Parameter | Default | Range |
|-----------|---------|-------|
| Max position per market | $1,000 | $100 - $10,000 |
| Max capital per opportunity | 20% of portfolio | 5% - 50% |
| Max open positions | 50 | 10 - 200 |
| Max daily trades | 500 | 50 - 2,000 |

### Spread Thresholds

| Strategy | Min Spread | Rationale |
|----------|------------|-----------|
| Binary InMarket | 2.5% | Cover 2% winner fee + buffer |
| Multi-Outcome | 2.0% | Lower per-position fee impact |
| Cross-Platform | 5.0% | Higher risk, resolution uncertainty |
| Negative Risk | 2.0% | Multiple payouts offset fees |

### Circuit Breakers

| Trigger | Action |
|---------|--------|
| Daily loss > 5% of capital | Pause all trading for 24h |
| 3 consecutive failed trades | Pause for 1 hour |
| API errors > 10/minute | Pause for 5 minutes |
| WebSocket disconnect > 30s | Switch to REST polling |

### Liquidity Requirements

Before executing, verify:
- Order book depth supports position size
- Slippage estimate < 0.5%
- Both sides have sufficient liquidity

---

## 11. Trading Parameters

### Configurable Parameters

```yaml
# config.yaml

trading:
  enabled: true
  dry_run: false  # Paper trading mode

capital:
  total_usd: 10000
  per_platform:
    polymarket: 6000
    kalshi: 4000
  reserve_pct: 10  # Keep 10% as reserve

execution:
  order_type: "limit"
  timeout_seconds: 5
  max_retries: 2
  retry_delay_ms: 500

spreads:
  binary_min_pct: 2.5
  multi_outcome_min_pct: 2.0
  cross_platform_min_pct: 5.0
  negative_risk_min_pct: 2.0

positions:
  max_per_market_usd: 1000
  max_open_count: 50
  max_per_opportunity_pct: 20

risk:
  daily_loss_limit_pct: 5
  consecutive_fail_pause: 3
  min_liquidity_usd: 500

markets:
  blacklist: []  # Market IDs to ignore
  whitelist: []  # If set, only trade these
  min_volume_24h: 1000
  exclude_categories: ["adult", "explicit"]
```

---

## 12. User Interface

### Dashboard Requirements

**Overview Panel:**
- Current portfolio value
- Today's PnL ($ and %)
- Open positions count
- Active opportunities count
- System status (running/paused/error)

**Positions Table:**
| Column | Description |
|--------|-------------|
| Market | Market name/ID |
| Platform | Polymarket/Kalshi |
| Strategy | Arbitrage type used |
| Entry Cost | Total cost basis |
| Current Value | Mark-to-market |
| Unrealized PnL | Profit/loss |
| Resolution Date | When market resolves |

**Trade History:**
| Column | Description |
|--------|-------------|
| Timestamp | Trade execution time |
| Market | Market name |
| Strategy | Strategy used |
| Spread | Captured spread |
| Amount | Position size |
| Fees | Total fees paid |
| Net PnL | Realized profit |

**Opportunities Panel:**
- Live feed of detected opportunities
- Spread percentage
- Available liquidity
- Execution status (pending/executed/skipped)

**Controls:**
- Start/Stop trading button
- Emergency stop (close all positions)
- Parameter adjustment forms
- Manual trade override

---

## 13. Data & Analytics

### Data Storage

**Time-Series Data (TimescaleDB):**
- Order book snapshots (1-second resolution)
- Price history for all monitored markets
- Trade executions with full details
- System performance metrics

**Relational Data (PostgreSQL):**
- Market metadata
- Position records
- Configuration history
- User settings

### Analytics Requirements

**Real-Time:**
- Current arbitrage opportunities
- Position PnL tracking
- System health metrics

**Historical:**
- Trade performance by strategy
- Profit attribution analysis
- Market efficiency trends
- Opportunity capture rate

**Reports:**
- Daily PnL summary
- Weekly performance report
- Monthly strategy analysis
- Tax reporting data export

---

## 14. Security Requirements

### Credential Management

| Requirement | Implementation |
|-------------|----------------|
| API keys encrypted at rest | AWS Secrets Manager / HashiCorp Vault |
| Private keys never logged | Redaction in all log outputs |
| Separate keys per environment | Dev/Staging/Prod isolation |
| Key rotation support | Automated rotation every 90 days |

### Access Control

| Requirement | Implementation |
|-------------|----------------|
| Dashboard authentication | OAuth2 / API key |
| Parameter changes audited | Full audit log |
| Trading controls protected | Require 2FA for sensitive ops |

### Network Security

| Requirement | Implementation |
|-------------|----------------|
| All API calls over HTTPS | TLS 1.3 required |
| WebSocket connections encrypted | WSS only |
| IP allowlisting (optional) | Platform-specific if supported |

---

## 15. Compliance & Legal

### Platform Terms of Service

| Platform | Automated Trading | Notes |
|----------|-------------------|-------|
| Polymarket | ✅ Allowed | Official API provided |
| Kalshi | ✅ Allowed | Official API provided |
| PredictIt | ❌ Prohibited | Read-only use only |
| Robinhood | N/A | No API available |

### Regulatory Considerations

- **Kalshi:** CFTC-regulated, US residents only for certain markets
- **Polymarket:** Prior CFTC settlement, international focus
- Consult legal counsel for specific jurisdictional requirements
- Maintain records for tax reporting purposes

### Data Retention

- Trade records: 7 years (tax compliance)
- Performance logs: 1 year
- System logs: 90 days

---

## 16. Release Phases

### Phase 1: MVP (Weeks 1-4)

**Goals:**
- Basic arbitrage detection and execution
- Polymarket integration only
- Binary and multi-outcome strategies
- Command-line interface
- Basic logging and alerting

**Deliverables:**
- [ ] Polymarket WebSocket connection
- [ ] Order book state management
- [ ] Binary arbitrage scanner
- [ ] Multi-outcome arbitrage scanner
- [ ] Trade execution engine
- [ ] Position tracker
- [ ] Basic risk management
- [ ] Discord/Telegram alerts
- [ ] Configuration file support

**Success Criteria:**
- Execute 10+ profitable trades
- 95%+ trade success rate
- <500ms execution latency

---

### Phase 2: Multi-Platform (Weeks 5-8)

**Goals:**
- Add Kalshi integration
- Cross-platform arbitrage
- Negative risk strategy
- Web dashboard

**Deliverables:**
- [ ] Kalshi API integration
- [ ] Kalshi WebSocket connection
- [ ] Cross-platform price comparison
- [ ] Cross-platform execution
- [ ] Negative risk scanner
- [ ] Streamlit dashboard
- [ ] Historical data storage
- [ ] Performance analytics

**Success Criteria:**
- Cross-platform trades executing
- Dashboard showing live data
- 50+ trades/week automated

---

### Phase 3: Advanced Strategies (Weeks 9-12)

**Goals:**
- Combinatorial arbitrage
- Enhanced analytics
- PredictIt monitoring
- Production hardening

**Deliverables:**
- [ ] Market relationship mapping
- [ ] Combinatorial arbitrage logic
- [ ] PredictIt price feed
- [ ] Advanced dashboard features
- [ ] Comprehensive testing suite
- [ ] Documentation
- [ ] Production deployment

**Success Criteria:**
- All 5 strategy types operational
- 99.9% uptime achieved
- Positive monthly PnL

---

### Phase 4: Optimization (Ongoing)

**Goals:**
- Performance optimization
- ML-based predictions
- News integration
- Scale infrastructure

**Deliverables:**
- [ ] Latency optimization
- [ ] ML opportunity scoring
- [ ] News/social feed integration
- [ ] Auto-scaling infrastructure
- [ ] Advanced risk models

---

## 17. Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | What is the initial capital allocation? | Product | Open |
| 2 | Which cloud provider for hosting? | Engineering | Open |
| 3 | Tax reporting requirements by jurisdiction? | Legal | Open |
| 4 | Should we support multiple wallets/accounts? | Product | Open |
| 5 | What alerting channels are required? | Product | Open |
| 6 | Backup execution strategy if primary fails? | Engineering | Open |
| 7 | How to handle market resolution disputes? | Product | Open |
| 8 | Geographic restrictions for platform access? | Legal | Open |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Arbitrage** | Risk-free profit from price discrepancies |
| **CLOB** | Central Limit Order Book |
| **Leg Risk** | Risk that one part of multi-leg trade fails |
| **NegRisk** | Polymarket's negative risk market type |
| **Slippage** | Price movement during trade execution |
| **Spread** | Difference between best bid and ask |

---

## Appendix B: References

- [Research Document](./research/prediction-market-api-research.md)
- [Polymarket Documentation](https://docs.polymarket.com/)
- [Kalshi API Documentation](https://docs.kalshi.com/)
- [Academic Paper: Arbitrage in Prediction Markets](https://arxiv.org/abs/2508.03474)

---

*Document Version History:*
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 2024 | Product Team | Initial draft |
