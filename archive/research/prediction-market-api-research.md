# Prediction Market API & Arbitrage Strategy Research

## Table of Contents
1. [Platform API Overview](#platform-api-overview)
2. [Arbitrage Strategy Types](#arbitrage-strategy-types)
3. [Trading Strategies (Non-Arbitrage)](#trading-strategies-non-arbitrage)
4. [Bot Trading](#bot-trading)
5. [Notable Traders & Bots](#notable-traders--bots)
6. [Profitability Data](#profitability-data)
7. [Execution Requirements](#execution-requirements)
8. [Risk Analysis](#risk-analysis)
9. [Bot Configuration Reference](#bot-configuration-reference)
10. [Tools & Resources](#tools--resources)
11. [Sources](#sources)

---

## Platform API Overview

### 1. Polymarket ✅ Best for Arbitrage

| Aspect | Details |
|--------|---------|
| **CLOB Base URL** | `https://clob.polymarket.com` |
| **Gamma API URL** | `https://gamma-api.polymarket.com` |
| **Chain** | Polygon (Chain ID: 137) |
| **Settlement** | On-chain, USDC-based |
| **Authentication** | L1 (wallet signer), L2 (API credentials for trading) |
| **Rate Limits** | ~1,000 calls/hour (free tier) |
| **WebSocket** | Yes - real-time order book updates |

#### Key Endpoints

**Market Data (Public)**
- `GET /book?token_id={id}` - Order book with bids/asks
- `GET /price` - Market price for token and side
- `GET /prices` - Multiple market prices
- `POST /spreads` - Bid-ask spread data
- `GET /midpoint` - Midpoint price for token
- `GET /prices-history` - Historical price data
- `GET /markets` - List all markets

**Trading (L2 Auth Required)**
- `POST /order` - Place single order
- `POST /orders` - Batch orders
- `DELETE /order/{id}` - Cancel order
- `GET /orders` - Get user orders

#### Order Book Response Structure
```json
{
  "market": "0x1b6f76e5b8587ee896c35847e12d11e75290a8c3...",
  "asset_id": "string",
  "timestamp": "2024-01-01T00:00:00Z",
  "hash": "string",
  "bids": [{"price": "0.45", "size": "1000"}],
  "asks": [{"price": "0.47", "size": "500"}],
  "min_order_size": "1",
  "tick_size": "0.01",
  "neg_risk": false
}
```

#### Client Libraries
- Python: [py-clob-client](https://github.com/Polymarket/py-clob-client)
- TypeScript: [clob-client](https://github.com/Polymarket/clob-client)
- Rust: [rs-clob-client](https://github.com/Polymarket/rs-clob-client)
- Go: Available on GitHub

---

### Polymarket Smart Contracts (On-Chain Execution)

Critical for atomic execution and eliminating leg risk.

| Contract | Address (Polygon) | Purpose |
|----------|-------------------|---------|
| **CTF Exchange** | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` | Binary YES/NO atomic swaps |
| **NegRisk CTFExchange** | `0x5b049...2016a` | Multi-outcome market exchange |
| **NegRiskAdapter** | `0x39b3e...b83c2` | NO→YES position conversion |
| **UMA Oracle (V2.0)** | `0x5022...6e474` | Decentralized resolution |
| **USDC (Collateral)** | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` | Polygon USDC |

#### CTF Exchange - Atomic Swap Mechanics

The exchange enables **atomic swaps between ERC1155 outcome tokens and USDC collateral**:

**Scenario 1: NORMAL Exchange**
```
Maker: Buy 100 token A @ $0.50 (offers 50 USDC)
Taker: Sell 50 token A @ $0.50 (offers 50 token A)
→ Direct token-for-collateral swap
```

**Scenario 2: MINT Operation (Key for Arbitrage)**
```
Maker: Buy 100 A @ $0.50
Taker: Buy 50 A' (complement) @ $0.50
→ Exchange mints 50 A + 50 A' from combined collateral
→ Distributes to each party
```

**Scenario 3: MERGE Operation**
```
Maker: Sell 50 A @ $0.50
Taker: Sell 100 A' @ $0.50
→ Exchange merges A + A' back into USDC
→ Distributes collateral proceeds
```

**Key Insight:** The MINT operation allows crossing orders for complementary positions (YES + NO) atomically, eliminating leg risk.

#### On-Chain Events to Monitor

| Event | Contract | Meaning |
|-------|----------|---------|
| `OrderFilled` | CTF Exchange | Trade executed |
| `OrdersMatched` | CTF Exchange | Two orders crossed |
| `PositionSplit` | CTF/NegRisk | Collateral → YES+NO tokens |
| `PositionsMerge` | CTF/NegRisk | YES+NO → Collateral |
| `PositionsConverted` | NegRiskAdapter | Multi-outcome conversion |

#### NegRiskAdapter - Multi-Outcome Conversions

For markets with n outcomes where exactly 1 wins:
```
1 NO-A + 1 NO-B ≡ 1 USDC + 1 YES-C
```

This equivalence enables:
- Atomic conversion between equivalent positions
- No leg risk on negative risk strategies
- Gas-efficient batch operations

---

### Polymarket Official GitHub Repositories

| Repository | Language | Stars | Purpose |
|------------|----------|-------|---------|
| [py-clob-client](https://github.com/Polymarket/py-clob-client) | Python | 485 | Official Python CLOB client |
| [clob-client](https://github.com/Polymarket/clob-client) | TypeScript | 325 | Official TS CLOB client |
| [rs-clob-client](https://github.com/Polymarket/rs-clob-client) | Rust | - | Infrastructure-grade client |
| [agents](https://github.com/Polymarket/agents) | Python | 980 | AI agent framework |
| [ctf-exchange](https://github.com/Polymarket/ctf-exchange) | Solidity | 229 | Exchange smart contracts |
| [neg-risk-ctf-adapter](https://github.com/Polymarket/neg-risk-ctf-adapter) | Solidity | 49 | Multi-outcome adapter |
| [uma-ctf-adapter](https://github.com/Polymarket/uma-ctf-adapter) | Solidity | 76 | Oracle resolution |

#### py-clob-client Usage

```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderType

# Initialize client
client = ClobClient(
    "https://clob.polymarket.com",
    key=PRIVATE_KEY,
    chain_id=137,
    signature_type=1,  # 0=EOA, 1=Magic, 2=Gnosis
    funder=FUNDER_ADDRESS
)
client.set_api_creds(client.create_or_derive_api_creds())

# Get order book
book = client.get_order_book(token_id="<token_id>")

# Place FOK (Fill-or-Kill) order - atomic execution
mo = MarketOrderArgs(
    token_id="<token_id>",
    amount=25.0,
    side=BUY,
    order_type=OrderType.FOK
)
signed = client.create_market_order(mo)
resp = client.post_order(signed, OrderType.FOK)
```

**Order Types:**
- `GTC` (Good-Till-Cancelled): Standard limit order
- `FOK` (Fill-or-Kill): Must fill entirely or cancel immediately

**FOK Precision Limits:**
- Maker amount: 2 decimal places
- Taker amount: 4 decimal places
- size × price ≤ 2 decimal places

#### Polymarket Agents Framework

For building autonomous trading bots:

```python
# CLI Commands
python scripts/python/cli.py get-all-markets --limit 100
python agents/application/trade.py

# Architecture
├── Chroma.py      # Vector DB for news/context
├── Gamma.py       # Market metadata API
├── Polymarket.py  # Trading execution
└── Objects.py     # Pydantic data models
```

**Capabilities:**
- Market discovery and filtering
- News/search context integration
- LLM-based probability estimation
- Automated trade execution

---

### 2. Kalshi ✅ Best Regulated Option

| Aspect | Details |
|--------|---------|
| **Production URL** | `https://trading-api.kalshi.com/trade-api/v2` |
| **Demo URL** | `https://demo-api.kalshi.co` |
| **Elections URL** | `https://api.elections.kalshi.com/trade-api/v2` |
| **Authentication** | RSA-PSS signed requests |
| **Rate Limits** | 20 req/sec (basic), up to 100 req/sec (premier) |
| **WebSocket** | Yes - real-time orderbook, tickers, trades |
| **Max Orders** | 20 per batch, 200,000 open orders per user |

#### Authentication Headers
```
KALSHI-ACCESS-KEY: <your_key_id>
KALSHI-ACCESS-TIMESTAMP: <milliseconds>
KALSHI-ACCESS-SIGNATURE: <signed_hash>
```

**Signature Generation:**
```
message = timestamp + http_method + path (without query params)
signature = RSA-PSS sign(message, private_key)
```

**Important:** Tokens expire every 30 minutes - implement re-auth logic.

#### Key Endpoints

**Market Data (No Auth)**
- `GET /markets/{ticker}/orderbook` - Order book (bids for yes/no)
- `GET /markets` - List markets with filters
- `GET /series/{series}/markets/{ticker}/candlesticks` - Price history

**Trading (Auth Required)**
- `POST /orders` - Place orders
- `DELETE /orders/{order_id}` - Cancel order
- `PUT /orders/{order_id}` - Amend order
- `GET /portfolio/positions` - Current positions
- `GET /portfolio/fills` - Trade fills

#### Order Book Structure
Binary market mechanics: `bid for YES at price X = ask for NO at price (100-X)`

Returns only bids for both YES and NO sides.

#### Client Libraries
- Python: [kalshi-python](https://pypi.org/project/kalshi-python/)

---

### 3. PredictIt ⚠️ Read-Only API

| Aspect | Details |
|--------|---------|
| **Base URL** | `https://www.predictit.org/api/marketdata/all/` |
| **Single Market** | `https://www.predictit.org/api/marketdata/markets/{id}` |
| **Authentication** | None (public) |
| **Update Frequency** | Every 60 seconds |
| **Trading** | **NOT SUPPORTED** - TOS prohibits automated trading |
| **License** | Non-commercial use only |

#### Response Fields
```json
{
  "id": 12345,
  "name": "Market Name",
  "contracts": [{
    "id": 67890,
    "name": "Contract Name",
    "lastTradePrice": 0.65,
    "bestBuyYesCost": 0.66,
    "bestBuyNoCost": 0.35,
    "bestSellYesCost": 0.64,
    "bestSellNoCost": 0.34,
    "lastClosePrice": 0.63,
    "status": "Open"
  }]
}
```

**Use Case:** Price monitoring and comparison only.

---

### 4. Robinhood ❌ No Event Contracts API

| Aspect | Details |
|--------|---------|
| **Crypto API** | Official at `docs.robinhood.com` |
| **Event Contracts** | **NO PUBLIC API** - app-only |
| **Backend** | Uses Kalshi as underlying exchange |

**Workaround:** Access same markets directly through Kalshi API.

---

### 5. Crypto.com Sports ❌ No Public API

| Aspect | Details |
|--------|---------|
| **Exchange API** | `exchange-docs.crypto.com` (crypto only) |
| **Prediction Markets** | **NO PUBLIC API** |
| **Access** | App/web interface only |

---

## Arbitrage Strategy Types

### Strategy Classification by Delta Neutrality

Delta measures directional exposure to outcome uncertainty. **Delta-neutral** strategies profit regardless of which outcome wins. **Directional** strategies require predicting the correct outcome.

| Delta | Category | Risk Profile | Example Strategies |
|-------|----------|--------------|-------------------|
| **0** | Fully Hedged | Zero directional risk | InMarket Arb, Negative Risk, Cross-Platform Arb |
| **~0** | Near-Neutral | Minimal directional risk | Market Making, Asymmetric Scalping, Combinatorial Arb |
| **0.3-0.5** | Partially Hedged | Some outcome dependency | Range Coverage, 3-Way Sports, Correlation Arb |
| **0.7-0.9** | Mostly Directional | High outcome dependency | Resolution Timing, Endgame Sweep |
| **1.0** | Fully Directional | Binary win/lose | News Sniping, Esports Parsing, LLM Trading |

---

### Tier 1: Delta-Neutral (Δ = 0) — Guaranteed Profit

These strategies lock in profit at entry regardless of outcome. True arbitrage.

| Strategy | Mechanism | Typical Return | Competition |
|----------|-----------|----------------|-------------|
| InMarket Binary | YES + NO < $1 | 1-3% | Very High |
| InMarket Multi-Outcome | Σ YES < $1 | 1-3% | Very High |
| Negative Risk | Σ NO < (n-1) | 1-2% | High |
| Cross-Platform | YES₁ + NO₂ < $1 | 2-5% | Medium |

**Key Characteristics:**
- Profit locked at execution (no outcome risk)
- Requires atomic/simultaneous execution
- Heavily competed by bots
- Opportunities exist for milliseconds

---

### Tier 2: Near-Neutral (Δ ≈ 0) — Constructed Neutrality

These strategies build delta-neutral positions over time or through spread capture.

| Strategy | Mechanism | Typical Return | Time to Neutral |
|----------|-----------|----------------|-----------------|
| Asymmetric Scalping | Accumulate YES+NO on dips | 2-5% | Days/weeks |
| Market Making | Bid/ask spread capture | 10-30% APY | Continuous |
| Combinatorial Arb | Logically linked markets | 2-4% | Immediate |

**Key Characteristics:**
- Neutrality achieved through position management
- Some execution/timing risk during construction
- Less competed than pure arb
- Requires patience and capital

---

### Tier 3: Partially Hedged (Δ = 0.3-0.5) — Statistical Edge

These strategies have directional exposure but use statistics/coverage to reduce risk.

| Strategy | Mechanism | Win Rate | Expected Return |
|----------|-----------|----------|-----------------|
| Range Coverage | Cover probable outcomes | 70-90% | 20-50% |
| 3-Way Sports | Cover all outcomes if sum < $1 | 100% if arb exists | 2-4% |
| Correlation Arb | Front-run lagging markets | 60-70% | 5-15% |

**Key Characteristics:**
- Not guaranteed profit, but high probability
- Relies on statistical patterns or market inefficiency
- Outcome partially matters
- Risk of outlier events

---

### Tier 4: Mostly Directional (Δ = 0.7-0.9) — Near-Certain Outcomes

These strategies bet on outcomes that are nearly certain but not yet resolved.

| Strategy | Mechanism | Win Rate | Risk |
|----------|-----------|----------|------|
| Resolution Timing | Event occurred, market not updated | 95%+ | Black swan |
| Endgame Sweep | Buy $0.95+ positions | 90%+ | Late reversal |

**Key Characteristics:**
- High win rate but not guaranteed
- Tail risk from unexpected events
- Time-value trade (patience for certainty)
- Lower returns due to high prices

---

### Tier 5: Fully Directional (Δ = 1.0) — Information Edge

These strategies require predicting outcomes faster/better than the market.

| Strategy | Edge Source | Speed Required | Win Rate |
|----------|-------------|----------------|----------|
| News Sniping | Faster news detection | <1 second | 70-85% |
| Esports Parsing | API vs stream delay | <30 seconds | 75-90% |
| Scoreboard Front-Running | Live API vs TV | <10 seconds | 70-85% |
| LLM News Trading | AI analysis speed | <5 seconds | 60-75% |
| Sequential Entry (Crypto) | Price momentum | <5 minutes | 65-75% |

**Key Characteristics:**
- Requires information/speed advantage
- Binary outcomes (right or wrong)
- Infrastructure-intensive
- Edge degrades as competition increases

---

### Strategy Selection Matrix

| Your Situation | Recommended Tier | Why |
|----------------|------------------|-----|
| Risk-averse, want guaranteed returns | Tier 1 (Δ=0) | No outcome risk |
| Have capital, patient | Tier 2 (Δ≈0) | Build positions over time |
| Statistical background | Tier 3 (Δ=0.3-0.5) | Leverage analysis skills |
| Want passive income | Tier 4 (Δ=0.7-0.9) | High probability, low effort |
| Fast infrastructure, domain expertise | Tier 5 (Δ=1.0) | Maximize edge when you have it |

---

### Automatability by Delta

| Delta Tier | Automation Score | Notes |
|------------|------------------|-------|
| Tier 1 (Δ=0) | 9/10 | Fully automatable, well-defined rules |
| Tier 2 (Δ≈0) | 7/10 | Needs position tracking, some judgment |
| Tier 3 (Δ=0.3-0.5) | 6/10 | Statistical models needed |
| Tier 4 (Δ=0.7-0.9) | 5/10 | Requires event verification |
| Tier 5 (Δ=1.0) | 8/10 | High automation, but needs domain-specific models |

---

## Detailed Strategy Documentation

### Type 1: InMarket Arbitrage (Single Platform)

#### 1A: Binary Markets (YES + NO < $1)

**Formula:**
```
Profit = $1.00 - (YES_ask + NO_ask) - (winner_fee × $1.00)
```

**Example:**
- YES ask: $0.48
- NO ask: $0.49
- Total cost: $0.97
- Gross profit: $0.03 (3.09%)
- After 2% winner fee: ~$0.01 (1.03%)

**Break-Even Calculation:**
```
Required Spread > Winner Fee + Gas Costs + Desired Margin
For Polymarket: Spread > 2% + ~0.1% + margin
```

#### 1B: Multi-Outcome Markets (Σ YES < $1)

**Formula:**
```
Profit = $1.00 - Σ(all YES outcomes) - fees
```

**Example (5 candidates):**
| Candidate | Price |
|-----------|-------|
| A | $0.35 |
| B | $0.28 |
| C | $0.18 |
| D | $0.12 |
| E | $0.05 |
| **Total** | **$0.98** |

Buy 1 share of each → Guaranteed $1 payout → $0.02 profit

**Real Performance:** One wallet turned $10,000 → $100,000 in 6 months across 10,000+ markets.

---

### Type 2: Combinatorial Arbitrage (Related Markets)

**Formula:**
```
YES(Market1) + NO(Market2) < $1.00
Where: Market1 outcome TRUE implies Market2 outcome FALSE
```

**Example: Election Hedging (Jeremy Whittaker Strategy)**
- Market 1: "Kamala Harris wins presidency" - YES
- Market 2: "GOP Electoral College margins" - ALL positions
- These are logical opposites
- If combined cost < $1 → guaranteed profit
- **Actual yield achieved:** 3.5% over 41 days

**Example: Fed Rates**
- Market A: "Fed cuts rates Dec" at 72¢ YES
- Market B: "Fed holds rates Dec" at 26¢ YES
- If mutually exclusive and sum < $1 → arbitrage

**Research Finding:**
- Only 38% of LLM-detected dependent pairs generated profit
- Total combinatorial profit: $95,157 (0.24% of all arbitrage)
- High failure rate due to execution barriers

---

### Type 3: Cross-Platform Arbitrage

**Formula:**
```
YES(Platform1) + NO(Platform2) < $1.00
```

**Example: Kalshi ↔ Polymarket**
| Platform | Position | Price |
|----------|----------|-------|
| Kalshi | YES | $0.42 |
| Polymarket | NO | $0.56 |
| **Total** | | **$0.98** |

Guaranteed payout: $1.00
**Profit: $0.02 (2.04%)**

**Best Conditions:**
- During major news events when platforms react at different speeds
- 2024 election saw 5-10% discrepancies frequently
- Polymarket leads Kalshi in price discovery

**Critical Risks:**
- Resolution criteria may differ between platforms
- Oracle divergence (UMA governance vs CFTC regulation)
- **Recommendation:** Only execute if spread > 15¢

---

### Type 4: Negative Risk (Multi-Outcome NO Strategy)

**Formula:**
```
NO₁ + NO₂ + ... + NOₙ < (n - 1)
```

**Logic:** In an n-outcome market where exactly 1 wins:
- Exactly (n-1) NO positions pay out $1 each
- Total guaranteed payout = $(n-1)
- If you buy all n NOs for less than $(n-1) → profit

**Example (4 outcomes):**
| Outcome | NO Price |
|---------|----------|
| A | $0.70 |
| B | $0.72 |
| C | $0.78 |
| D | $0.75 |
| **Total** | **$2.95** |

- Guaranteed payout: $3.00 (3 NOs win)
- **Profit: $0.05 per set**

**Polymarket NegRisk Markets:**
- "Winner-take-all" events
- Betting YES on one = betting NO on all others
- Atomic swaps between positions supported

---

### Type 5: Endgame Sweep (Tail-End Trading)

**Strategy:** Buy outcomes priced $0.95-$0.99 and wait for resolution.

**Statistics:**
- 90% of large orders (>$10,000) execute above $0.95
- Logic: "Time in exchange for certainty"

**Risk:** Black swan events can reverse "sure things"
- Assassination scenario
- Last-minute legal challenges
- Scandal revelations

---

### Type 6: News Sniping / Event-Driven Arbitrage

**Execution Flow:**
```
1. Twitter API detects: "Trump nominates [X] for cabinet"
2. NLP model extracts: nomination event + candidate name
3. Bot identifies: relevant Polymarket market
4. Execute: BUY in 0.3 seconds
5. Price moves: 60% → 100%
6. Close: when market catches up
```

**Infrastructure Required:**
- Twitter/X API real-time stream
- NLP models for signal extraction
- Sub-second execution pipeline
- Polymarket WebSocket connections
- Server proximity to Polygon nodes

**Research Finding:** "Platforms like Polymarket often respond to probabilities faster than the media."

---

### Type 7: Correlation Arbitrage

**Strategy:** Exploit lag between correlated markets.

**Example:**
- Overall election probability drops 10%
- Individual swing state probabilities haven't adjusted
- Buy/sell the mispriced state markets

**Example:**
- Trump primary odds: 70%
- Trump general election odds: 40%
- If primary confirmed → general odds will spike
- Front-run the adjustment

---

### Type 8: Market Making

**Strategy:** Provide liquidity on both sides, capture spread.

**Execution:**
```
1. Place YES limit order at $0.48
2. Place NO limit order at $0.50
3. If both fill: cost $0.98, payout $1.00
4. Earn spread + Polymarket liquidity rewards
```

**Note:** Only 3-4 serious liquidity providers existed on Polymarket historically.

---

### Type 9: Range Coverage (Statistical Arbitrage)

**Strategy:** Cover multiple outcomes in bucketed markets where statistics favor a predictable range.

**Example: Elon Musk Tweet Count Market**
```
Elon averages ~43 tweets/day
For Dec 23-30 (7 days): Expected range = 301-343 tweets

Buy 1 share of each range:
- 300 range @ $0.11
- 320 range @ $0.14
- 340 range @ $0.16
- 360 range @ $0.14
- 380 range @ $0.13
─────────────────────
Total cost: $0.68

If ANY hit → Payout: $1.00
Profit: $0.32 (47% return)
```

**Why It Works:**
- Historical data provides statistical baseline
- Covering adjacent ranges = high probability of success
- Not prediction, just statistical coverage
- Works on any bucketed/range market with predictable behavior

**Best Markets:**
- Tweet counts (consistent posters)
- Weather ranges (temperature, rainfall)
- Economic indicators (known seasonality)
- Sports statistics (player averages)

**Key Calculation:**
```
Expected Value = P(any_range_hits) × $1.00 - total_cost
If P(any_range_hits) > total_cost → profitable
```

**Risk:** Outlier events (account suspension, illness, etc.)

---

### Type 10: 3-Way Sports Arbitrage

**Formula:**
```
YES(Team1) + YES(Team2) + YES(Draw) < $1.00
```

**Concept:** In sports markets with 3 possible outcomes (win/lose/draw), if all three YES prices sum to less than $1, you can guarantee profit.

**Example (Soccer Match):**
| Outcome | YES Price |
|---------|-----------|
| Team A Wins | $0.35 |
| Team B Wins | $0.32 |
| Draw | $0.30 |
| **Total** | **$0.97** |

- Buy 1 share of each outcome
- Exactly one will pay $1.00
- Cost: $0.97 → Profit: $0.03 (3.1%)

**Alternative Formula:**
```
YES(Team1) + NO(Team2) < DRAW price
```
When Team 1 winning implies Team 2 losing, but draw is still possible.

**Best Markets:**
- Soccer/Football (draws common)
- Hockey (regulation time)
- Any 3-way outcome market

**Key Consideration:** Market must resolve to exactly one of three outcomes.

---

### Type 11: Resolution Timing Arbitrage

**Strategy:** Exploit markets where resolution is imminent and outcome is near-certain.

**Example:**
- Market: "Will X happen by Dec 31?"
- Current date: Dec 30
- Event already occurred on Dec 28
- YES price: $0.97 (hasn't updated to $1.00)

Buy YES at $0.97, wait for resolution, collect $1.00 = 3% risk-free in 1-2 days.

**Best Conditions:**
- Events that have already occurred but market hasn't resolved
- Clear, verifiable outcomes
- Short time to resolution

---

## Trading Strategies (Non-Arbitrage)

Beyond pure arbitrage, these strategies can generate consistent returns on prediction markets.

### Strategy 1: Niche Trading

**Difficulty:** 8/10 | **Starting Capital:** $100-200

**Concept:** Trade in small, overlooked markets where you have informational edge.

**Why It Works:**
- Most volume concentrates on elections and crypto
- Thousands of smaller markets receive little attention
- Local/specialized knowledge creates edge

**Best Niches:**
- Local politics (city elections, state referendums)
- Obscure sports (minor leagues, esports tournaments)
- Specialized topics (scientific discoveries, industry-specific events)
- Foreign events (elections in countries you follow closely)

**Example:**
- User "Malte" made $46K in 3 months trading Ukrainian conflict markets
- Deep knowledge of the conflict gave informational advantage

**Key Success Factors:**
- Domain expertise is essential
- Patience (lower volume = slower execution)
- Monitor markets others ignore

---

### Strategy 2: Spreads Trading

**Difficulty:** 5/10 | **Starting Capital:** $50-100

**Concept:** Buy positions when spreads are wide, then sell as liquidity improves.

**How It Works:**
```
1. Find market with wide bid-ask spread (e.g., 45¢ / 55¢)
2. Buy YES at 46¢ (slightly above bid)
3. Wait for more liquidity to enter market
4. Sell YES at 53¢ (slightly below ask)
5. Profit: 7¢ per share (15% return)
```

**Best Conditions:**
- New markets with low liquidity
- Markets about to get media attention
- After major news when spreads widen temporarily

**Risk:** Spreads may never tighten, position becomes illiquid

---

### Strategy 3: Bonding (UMA Staking)

**Difficulty:** 5/10 | **Starting Capital:** $10,000-50,000

**Concept:** Stake UMA tokens to propose or dispute market resolutions, earning rewards.

**How It Works:**
1. When market ends, anyone can propose resolution by staking UMA
2. If proposal is correct and unchallenged → get stake back + reward
3. If disputed and you're right → win disputer's stake
4. If wrong → lose your stake

**Reward Structure:**
- Proposer reward: ~$5-50 per correct resolution (varies by market)
- Dispute reward: Opponent's stake if you win

**Strategy:**
```
1. Monitor markets approaching resolution
2. Wait for clear, verifiable outcomes
3. Propose resolution with UMA stake
4. Collect reward after challenge period
```

**Risks:**
- Ambiguous resolutions can be disputed
- Requires UMA tokens (price volatility)
- Gas costs for staking transactions

**Best For:** Risk-averse participants with UMA holdings

---

### Strategy 4: Daily Rewards Farming

**Difficulty:** 6/10 | **Starting Capital:** $1,000-5,000

**Platform:** Polymarket distributes $12.4M in daily USDC rewards to liquidity providers.

**How Rewards Work:**
- Proportional to order book liquidity provided
- Tighter spreads = higher reward multiplier
- Must keep orders open (not just place and cancel)

**Strategy:**
```
1. Identify high-volume markets
2. Place competitive limit orders on both sides
3. Maintain tight spreads to maximize reward share
4. Compound rewards into larger positions
```

**ROI Range:** 10-30% APY depending on competition

**Risks:**
- Adverse selection (smarter traders pick you off)
- Price movements against your positions
- Competition from professional market makers

---

### Strategy 5: Copy Trading

**Difficulty:** 5/10 | **Starting Capital:** $100-500

**Concept:** Follow profitable traders and replicate their positions.

**Tools:**
- [polymarket.com/leaderboard](https://polymarket.com/leaderboard) - Official leaderboard
- Third-party trackers for whale wallets

**Who to Follow:**
- Traders with 1000+ trades and positive PnL
- Consistent performers across multiple market types
- Avoid flash-in-the-pan winners (luck vs skill)

**Execution:**
```
1. Identify consistent winners
2. Monitor their new positions via wallet tracking
3. Enter same positions (smaller size)
4. Set stop-losses since you don't know their thesis
```

**Caution:**
- You don't know their exit strategy
- They may have hedged positions you can't see
- Slippage if you're late to follow

---

### Strategy 6: Dispute Trading

**Difficulty:** 4/10 | **Starting Capital:** $100-1,000

**Concept:** Trade during UMA oracle dispute periods when markets are mispriced.

**How UMA Resolution Works:**
1. Market ends → Initial resolution proposed
2. 2-hour dispute window opens
3. If disputed → escalates to UMA token holder vote
4. Final resolution after voting period

**Strategy:**
```
1. Monitor markets reaching resolution
2. If resolution is incorrect, buy the "real" outcome
3. Dispute if necessary (requires UMA stake)
4. Profit when correct resolution confirmed
```

**Example:**
- Market resolves to NO incorrectly
- YES price drops to $0.10
- Buy YES, dispute, correct resolution = $0.90 profit

**Risks:**
- Requires understanding UMA oracle process
- Gas costs for disputes
- May be wrong about "correct" resolution

---

### Strategy 7: Mention Markets (Social Media Arbitrage)

**Difficulty:** 6/10 | **Starting Capital:** $100-500

**Concept:** Trade on "Will X mention Y?" markets using insider knowledge or analysis.

**Examples:**
- "Will Trump tweet about [topic]?"
- "Will [influencer] mention [product]?"
- "Will [politician] reference [event] in speech?"

**Edge Sources:**
- Pattern analysis (posting habits, time zones)
- Event calendars (speeches, interviews scheduled)
- Social graph analysis (who interacts with whom)

**Automation Potential:**
- Monitor social media for trigger events
- NLP analysis of content themes
- Predictive models based on historical patterns

---

### Strategy 8: News Trading

**Difficulty:** 7/10 | **Starting Capital:** $500-2,000

**Concept:** React to news faster than the market.

**Infrastructure Required:**
- News API feeds (Reuters, AP, Bloomberg Terminal)
- Social media monitoring (Twitter/X API)
- NLP for sentiment/entity extraction
- Low-latency execution pipeline

**Strategy Tiers:**

**Tier 1: Manual (Slowest)**
- Watch news, manually trade
- 30-60 second reaction time
- Catches only major mispricings

**Tier 2: Alert-Based**
- Automated alerts for keywords
- Manual execution
- 5-15 second reaction time

**Tier 3: Fully Automated**
- News parsing → signal extraction → order execution
- Sub-second reaction time
- Requires significant infrastructure

**Best Events:**
- Economic data releases (jobs report, CPI)
- Political announcements (endorsements, nominations)
- Sports injuries/lineup changes

---

## Bot Trading

### Bot Type 1: Arbitrage Bots

**Difficulty:** 9/10 | **Capital:** $10,000-50,000

**Function:** Continuously scan for pricing inefficiencies and execute atomic trades.

**Architecture:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  WebSocket  │───►│   Scanner   │───►│  Executor   │
│   Feeds     │    │   Engine    │    │   (FOK)     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
   Order Books      Opportunity         Atomic
   (Poly/Kalshi)     Detection          Trades
```

**Key Components:**
- Real-time order book aggregation
- Multi-market correlation detection
- Atomic execution (MINT/MERGE operations)
- Position tracking and risk limits

---

### Bot Type 2: Market Making Bots

**Difficulty:** 7/10 | **Capital:** $5,000-20,000

**Function:** Provide liquidity on both sides, capture spread and rewards.

**Strategy:**
```python
# Simplified market making logic
fair_value = calculate_fair_value(market_data)
spread = 0.02  # 2 cents

bid_price = fair_value - spread/2
ask_price = fair_value + spread/2

place_order(side="BUY", price=bid_price, size=100)
place_order(side="SELL", price=ask_price, size=100)
```

**Key Challenges:**
- Inventory management (don't accumulate one-sided risk)
- Adverse selection (informed traders pick you off)
- Competition from professional MMs

---

### Bot Type 3: Sniper Bots

**Difficulty:** 8/10 | **Capital:** $1,000-10,000

**Function:** React to news/events faster than manual traders.

**Architecture:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Data Feed  │───►│    NLP      │───►│   Trade     │
│  (Twitter)  │    │  Processor  │    │  Executor   │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Speed Requirements:**
- News detection: <100ms
- Signal extraction: <200ms
- Order execution: <500ms
- Total latency: <1 second

**Edge Sources:**
- Twitter API streaming
- RSS feed aggregation
- Telegram channel monitoring
- Official announcement APIs

---

### Bot Type 4: Copy Trading Bots

**Difficulty:** 5/10 | **Capital:** $500-5,000

**Function:** Automatically replicate trades from successful wallets.

**Implementation:**
```python
# Monitor target wallets
target_wallets = ["0x...", "0x...", "0x..."]

for wallet in target_wallets:
    new_positions = monitor_wallet(wallet)
    for position in new_positions:
        if position.size > MIN_SIZE:
            replicate_trade(position, scale=0.1)  # 10% of their size
```

**Challenges:**
- Detecting trades in real-time (on-chain monitoring)
- Slippage from delayed execution
- Unknown exit strategies

---

### Bot Type 5: Esports Parsing Bots

**Difficulty:** 9/10 | **Capital:** $1,000-10,000

**Function:** Front-run esports markets using direct game API data that arrives 30-40 seconds before Twitch/YouTube stream delay.

**Real Performance:**
- One trader turned $111 → $208,000+ profit on League of Legends/Dota 2 markets
- Edge: Direct Riot/Valve API feeds vs viewers watching delayed streams

**Architecture:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Game API Feed  │───►│  State Parser   │───►│  Trade Signal   │
│  (Riot/Valve)   │    │  (Kill/Obj)     │    │  Generator      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
    Real-time data         Event detection        FOK execution
    (30-40s ahead)         (first blood,          on Polymarket
                           tower, baron)
```

**Key Data Sources:**
| Game | API | Latency Edge |
|------|-----|--------------|
| League of Legends | Riot Games API | 30-40s vs Twitch |
| Dota 2 | Valve Steam API | 30-40s vs Twitch |
| CS2 | Steam Game State | 20-30s vs stream |

**Why It Works:**
- Esports markets on Polymarket are retail-dominated
- Most bettors watch streams (delayed)
- API data is real-time (no delay)
- First blood, tower kills, baron = massive probability swings

**Execution Flow:**
```
1. Subscribe to game API WebSocket
2. Detect game-winning event (e.g., baron kill)
3. Calculate new win probability
4. Place FOK order before stream shows event
5. Market moves 10-30% in your favor as stream catches up
```

**Infrastructure:**
- Low-latency connection to game APIs
- Event detection models trained on historical data
- Pre-signed orders ready for instant execution
- Polymarket WebSocket for orderbook monitoring

**Risks:**
- API access may be restricted
- Low liquidity on esports markets
- Game-specific knowledge required
- Riot/Valve TOS considerations

---

### Bot Type 6: Scoreboard API Front-Running Bots

**Difficulty:** 8/10 | **Capital:** $1,000-5,000

**Function:** Trade traditional sports markets using raw scoreboard APIs (teletext, official feeds) that update before TV broadcasts.

**Concept:** TV broadcasts have 5-15 second delays. Raw score APIs update in real-time.

**Data Sources:**
| Source | Latency | Sports |
|--------|---------|--------|
| Teletext feeds | Near real-time | Soccer, Football |
| Official league APIs | 1-5s | NBA, NFL, MLB |
| Betting exchange feeds | 2-3s | All |
| TV broadcast | 5-15s delayed | All |

**Example Flow:**
```
1. Monitor NBA game via official stats API
2. Detect buzzer-beater basket in final seconds
3. Calculate game outcome probability
4. Execute trade on Polymarket/Kalshi
5. TV viewers see play 10 seconds later
6. Market reprices in your favor
```

**Key Opportunities:**
- Final seconds/minutes of close games
- Overtime triggers
- Game-winning plays
- Red zone scoring (NFL)

**Risks:**
- API access costs
- Low liquidity on live game markets
- Execution speed competition
- Variable API reliability

---

### Bot Type 7: Asymmetric Scalping Bots (Gabagool Strategy)

**Difficulty:** 7/10 | **Capital:** $5,000-50,000

**Function:** Accumulate YES and NO positions opportunistically on volatile markets until average basis < $1.

**Named After:** gabagool22, successful Polymarket trader

**Strategy Logic:**
```
Instead of simultaneous YES+NO purchase:
1. Buy YES when price dips (panic selling)
2. Buy NO when price dips (opposite panic)
3. Track: Avg_YES_Cost + Avg_NO_Cost
4. Target: Combined average < $0.95-0.98
5. Guaranteed profit at resolution regardless of outcome
```

**Example Execution:**
| Day | Action | Price | Running Avg |
|-----|--------|-------|-------------|
| 1 | Buy 100 YES | $0.52 | YES: $0.52 |
| 2 | Buy 100 NO | $0.46 | NO: $0.46 |
| 3 | Buy 50 YES | $0.48 | YES: $0.507 |
| 4 | Buy 50 NO | $0.44 | NO: $0.453 |
| **Final** | | | **$0.96 combined** |

**Result:** Regardless of outcome, receive $1.00 per share, profit $0.04/share.

**Best Markets:**
- High-volatility binary markets (elections, crypto)
- Markets with frequent sentiment swings
- News-driven markets with overreaction patterns

**Key Metrics to Track:**
```python
avg_yes = sum(yes_costs * yes_sizes) / sum(yes_sizes)
avg_no = sum(no_costs * no_sizes) / sum(no_sizes)
combined_basis = avg_yes + avg_no
profit_per_share = 1.0 - combined_basis
```

**Advantages over Pure Arbitrage:**
- Doesn't require simultaneous opportunities
- Works in markets without instant arb
- Captures volatility premium over time
- Less competition than pure arb bots

**Risks:**
- Capital tied up until resolution
- Market may not swing enough
- Requires patience and position tracking
- Resolution timing uncertainty

---

### Bot Type 8: LLM News Trading Bots

**Difficulty:** 7/10 | **Capital:** $2,000-10,000

**Function:** Use Claude/GPT for real-time sentiment analysis and trading signals.

**Architecture:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  News Feed  │───►│  LLM Parser │───►│  Trade      │
│  (Twitter,  │    │  (Claude/   │    │  Executor   │
│   RSS, AP)  │    │   GPT API)  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

**LLM Prompt Strategy:**
```python
prompt = """
Given this news headline: "{headline}"

Relevant Polymarket markets:
{market_list}

For each market, respond with:
1. Probability impact (positive/negative/neutral)
2. Magnitude (0-100% change expected)
3. Confidence (0-1)
4. Suggested action (BUY_YES, BUY_NO, HOLD)

JSON format only.
"""
```

**Use Cases:**
- Entity extraction (who is mentioned)
- Sentiment classification (positive/negative for outcome)
- Probability estimation (how much should price move)
- Market mapping (which markets affected)

**Infrastructure:**
- Streaming news API (Twitter, AP, Reuters)
- LLM API with low latency (Claude Instant, GPT-4-turbo)
- Market database with entity mappings
- Execution pipeline with position limits

**Latency Considerations:**
| Component | Target Latency |
|-----------|---------------|
| News detection | <100ms |
| LLM inference | 200-500ms |
| Signal generation | <50ms |
| Order execution | <200ms |
| **Total** | **<1 second** |

**Cost Optimization:**
- Use smaller/faster models for initial screening
- Reserve large models for high-confidence signals
- Cache entity-market mappings
- Batch low-urgency analysis

---

### Bot Type 9: Automated Market Making (AMM) Bots

**Difficulty:** 8/10 | **Capital:** $20,000-100,000

**Function:** Continuously provide liquidity on both sides of markets, capturing spread and liquidity rewards.

**Strategy:**
```python
class AMMBot:
    def calculate_quotes(self, fair_value, inventory):
        # Skew quotes based on inventory
        inventory_skew = inventory * INVENTORY_FACTOR

        bid = fair_value - BASE_SPREAD/2 - inventory_skew
        ask = fair_value + BASE_SPREAD/2 - inventory_skew

        return bid, ask

    def manage_inventory(self):
        # Reduce position when too large
        if abs(self.inventory) > MAX_INVENTORY:
            self.hedge_position()
```

**Key Components:**
1. **Fair value estimation** - Model true probability
2. **Spread calculation** - Balance profit vs fill rate
3. **Inventory management** - Don't accumulate directional risk
4. **Quote adjustment** - Widen spreads in volatile periods

**Revenue Sources:**
- Bid-ask spread capture
- Polymarket liquidity rewards ($12.4M daily pool)
- Information from order flow

**Risks:**
- Adverse selection (informed traders pick you off)
- Inventory accumulation
- Model mispricing
- Competition from professional MMs

---

### Bot Type 10: Whale Spotting/Copy Bots

**Difficulty:** 5/10 | **Capital:** $1,000-10,000

**Function:** Monitor large wallets and replicate their trades with slight delay.

**Implementation:**
```python
class WhaleTracker:
    def __init__(self, whale_addresses):
        self.whales = whale_addresses
        self.last_positions = {}

    async def monitor_whales(self):
        for whale in self.whales:
            current = await get_positions(whale)
            changes = self.detect_changes(whale, current)

            for change in changes:
                if change.size > MIN_SIGNAL_SIZE:
                    await self.queue_copy_trade(change)
```

**Signal Filtering:**
- Minimum position size threshold
- Whale track record check
- Market liquidity verification
- Risk limits per signal

**Data Sources:**
- On-chain position monitoring
- Polymarket leaderboard API
- Third-party whale tracking services

---

## Notable Traders & Bots

### Top Performing Bots

| Name | Strategy | Notable Trades | ROI |
|------|----------|----------------|-----|
| **gabagool22** | News sniping + Asymmetric scalping | Trump cabinet picks <1s, BTC markets | Unknown |
| **distinct-baguette** | Arbitrage | $50K+ in arb profits | High |
| **ilovecircle** | Market making | Consistent spreads | Steady |
| **RN1** | Multi-strategy | Diversified approach | Moderate |
| **Account88888** | Sequential entry (crypto) | $324K in 25 days on 15-min markets | 3240%+ |

### Top Traders to Follow

| Trader | Style | Best Markets | PnL |
|--------|-------|--------------|-----|
| **Malte** | Niche | Ukraine/conflict | $46K (3 mo) |
| **Theo** | Arbitrage | Cross-platform | High volume |
| **Domer** | News | Politics | Consistent |
| **Polymarket Whales** | Various | High-volume | Track on leaderboard |
| **Esports Parser** | API front-running | LoL/Dota 2 | $111 → $208K |

### Case Study: Account88888 (JaneStreetIndia Analysis)

**Strategy:** Sequential Entry Arbitrage on 15-Minute Crypto Markets

**Performance:** $324,000 profit in 25 days

**Market Focus:** BTC/ETH/SOL/XRP Up/Down 15-minute markets on Polymarket

**How It Works:**

These markets ask: "Will [COIN] go UP or DOWN in the next 15 minutes?"
- Binary outcome: UP (price higher) or DOWN (price lower/equal)
- Markets open every 15 minutes, 24/7
- Retail-dominated orderbooks with predictable inefficiencies

**The Strategy:**
```
1. Monitor BTC/ETH price movement in first 5-7 minutes of 15-min window
2. If significant move detected (e.g., BTC drops 0.3%):
   - DOWN becomes more likely
   - But YES_DOWN hasn't fully repriced yet
3. Enter position aligned with observed momentum
4. Market reprices as more participants notice
5. Exit with profit OR hold to resolution
```

**Key Insight - NOT Simultaneous Arbitrage:**

This is NOT buying YES+NO < $1 simultaneously. It's:
- **Sequential entry** based on real-time price observation
- **Volatility compression** - early moves predict final outcome
- **Microstructure edge** - retail orderbooks reprice slowly

**Why It Works (For Now):**
- New market type (limited historical data)
- Retail participants don't monitor crypto prices in real-time
- No professional MMs yet (will come)
- 15-min resolution = fast capital turnover

**Statistical Edge:**
- If BTC moves 0.5% in first 5 minutes, continuation probability ~65-70%
- Combined with slow orderbook repricing = profitable edge

**Sustainability Window:** 6-18 months before competition eliminates edge

**Replication Requirements:**
- Real-time crypto price feeds (Binance, Coinbase WebSocket)
- Polymarket WebSocket for orderbook monitoring
- Sub-second execution capability
- Statistical model for continuation probability

---

### Trend Detection Methods for 15-Minute Crypto Markets

For the Account88888/Sequential Entry strategy, detecting BTC/ETH direction early in the 15-minute window is critical. Here's a comprehensive framework ranked by effectiveness.

#### Signal Hierarchy by Effectiveness

| Category | Latency | Predictive Power | Implementation | Priority |
|----------|---------|------------------|----------------|----------|
| **Order Flow Analysis** | Milliseconds | 80% | Medium-High | CRITICAL |
| **Market Microstructure** | Microseconds | 75% | High | ESSENTIAL |
| **Derivatives Signals** | Seconds | 78% | Medium | HIGH |
| **Cross-Exchange Signals** | 1-3 seconds | 72% | Medium-High | HIGH |
| **Technical Indicators** | Sub-second | 65% | Low | CONTEXT |
| **Sentiment Signals** | Minutes | 60% | High | LOW |
| **On-Chain Signals** | 10-60 minutes | 55% | Medium | IGNORE |

---

#### Tier 1: Primary Signals (80% Weight)

##### 1. Order Flow - Volume Delta

The most powerful signal for short-term prediction. Measures aggressive buyers vs sellers.

```python
def calc_volume_delta(trades, window_seconds=300):
    """
    Calculate cumulative volume delta over rolling window.
    Positive = more aggressive buying, Negative = more aggressive selling.
    """
    recent = [t for t in trades if t.timestamp > time.time() - window_seconds]

    buys = sum(t.size for t in recent if t.aggressor == 'BUY')
    sells = sum(t.size for t in recent if t.aggressor == 'SELL')

    return buys - sells  # Positive = bullish pressure

# Signal interpretation
# > $5M imbalance in 5-min window = 78% directional accuracy
```

**Key Insight - CVD Divergence:**
- Price makes new high BUT Cumulative Volume Delta (CVD) does NOT make new high
- This divergence predicts reversal with ~75% accuracy in next 5-15 minutes

**Data Sources:**
- Binance WebSocket: `wss://stream.binance.com:9443/ws/btcusdt@trade`
- Coinbase WebSocket: `wss://ws-feed.exchange.coinbase.com`

##### 2. Liquidation Cascade Detection

**The highest-conviction signal.** When cascading liquidations begin, direction is nearly certain for next 5-15 minutes.

```python
class LiquidationMonitor:
    def __init__(self):
        self.liq_buffer = []
        self.CASCADE_THRESHOLD = 1_000_000  # $1M in 500ms
        self.CASCADE_WINDOW_MS = 500

    def on_liquidation(self, liq_event):
        self.liq_buffer.append(liq_event)
        self.cleanup_old()

        recent_volume = sum(l.size for l in self.liq_buffer)

        if recent_volume > self.CASCADE_THRESHOLD:
            direction = "DOWN" if liq_event.side == "LONG" else "UP"
            return {"signal": direction, "confidence": 0.85}

        return None
```

**Data Sources:**
- Bybit: `wss://stream.bybit.com/v5/public/linear` (liquidation channel)
- Coinglass API: Real-time liquidation aggregation
- Binance Futures: Force order stream

##### 3. Cross-Exchange Price Leadership

When Binance moves first, others follow within 30-120 seconds.

```python
class PriceLeadershipDetector:
    def __init__(self):
        self.prices = {}  # {exchange: [(timestamp, price), ...]}
        self.LEAD_THRESHOLD = 0.0005  # 0.05% move

    def detect_leader(self):
        """
        Returns the exchange currently leading price discovery.
        """
        binance_move = self.get_recent_move('binance')
        coinbase_move = self.get_recent_move('coinbase')

        if abs(binance_move) > self.LEAD_THRESHOLD:
            if abs(coinbase_move) < self.LEAD_THRESHOLD:
                # Binance moved, Coinbase hasn't caught up
                return {
                    "leader": "binance",
                    "direction": "UP" if binance_move > 0 else "DOWN",
                    "confidence": 0.72
                }

        return None
```

**Key Pattern:**
- Binance perpetual futures often leads by 0.5-2 seconds
- Trade in direction of leader before laggards catch up

---

#### Tier 2: Confirmation Signals (20% Weight)

##### 4. VWAP Deviation + Volume

```python
def vwap_signal(candles, current_price):
    """
    Price > 2 std from VWAP with rising volume = strong signal.
    """
    vwap = calculate_vwap(candles)
    std = calculate_vwap_std(candles)

    deviation = (current_price - vwap) / std

    if abs(deviation) > 2:
        # Extended from VWAP - mean reversion likely
        return -1 if deviation > 0 else 1  # Fade the extension

    return 0
```

##### 5. Open Interest Shocks

```python
def oi_signal(oi_data, price_data):
    """
    OI spike >15% + price move = trend confirmation.
    OI drop during rally = exhaustion warning.
    """
    oi_change = (oi_data[-1] - oi_data[-5]) / oi_data[-5]
    price_change = (price_data[-1] - price_data[-5]) / price_data[-5]

    if oi_change > 0.15 and price_change > 0:
        return 1  # Bullish confirmation
    elif oi_change > 0.15 and price_change < 0:
        return -1  # Bearish confirmation
    elif oi_change < -0.10 and abs(price_change) > 0.02:
        return 0  # Exhaustion - trend weakening

    return 0
```

##### 6. EMA Ribbon (Trend Context)

```python
def ema_ribbon_signal(candles):
    """
    5/10/20 EMA crossover on 1-min chart.
    Only use in trending markets (ADX > 25).
    """
    ema5 = calculate_ema(candles, 5)
    ema10 = calculate_ema(candles, 10)
    ema20 = calculate_ema(candles, 20)
    adx = calculate_adx(candles, 14)

    if adx < 25:
        return 0  # Ranging market, ignore

    if ema5 > ema10 > ema20:
        return 1  # Bullish alignment
    elif ema5 < ema10 < ema20:
        return -1  # Bearish alignment

    return 0
```

---

#### Complete Ensemble Decision Logic

```python
def get_15min_direction():
    """
    Weighted ensemble of all signals.
    Returns: "UP", "DOWN", or "NEUTRAL"
    """
    # Primary signals (80% weight)
    primary_score = (
        0.35 * volume_delta_signal() +
        0.30 * liquidation_cascade_signal() +
        0.15 * price_leadership_signal()
    )

    # High conviction on primary signals alone
    if abs(primary_score) >= 0.6:
        return "UP" if primary_score > 0 else "DOWN"

    # Check confirmations if primary unclear
    secondary_score = (
        0.10 * vwap_deviation_signal() +
        0.05 * oi_change_signal() +
        0.05 * ema_ribbon_signal()
    )

    total = primary_score + secondary_score

    if total > 0.15:
        return "UP"
    elif total < -0.15:
        return "DOWN"
    else:
        return "NEUTRAL"  # No trade
```

---

#### Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXCHANGE WEBSOCKETS                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Binance Spot   │  Binance Futures│  Coinbase/Bybit             │
│  @trade stream  │  @trade + @liq  │  Trade + Book feeds         │
└────────┬────────┴────────┬────────┴─────────────┬───────────────┘
         │                 │                      │
         ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-TIME ENGINE                             │
│  • Volume Delta Calculator (rolling 5-min window)               │
│  • Liquidation Cascade Detector                                 │
│  • Cross-Exchange Correlation Tracker                           │
│  • Order Book Imbalance Monitor                                 │
│  Throughput: 5,000+ msg/sec | Latency: <50ms                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL AGGREGATOR                            │
│  • Weighted ensemble scoring                                    │
│  • Confidence threshold gating                                  │
│  • Regime detection (trending vs ranging)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION OUTPUT                            │
│  Direction: UP/DOWN/NEUTRAL                                     │
│  Confidence: 0.0 - 1.0                                          │
│  Timestamp: Entry window (first 5-7 min of 15-min period)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POLYMARKET EXECUTION                         │
│  • FOK order on YES_UP or YES_DOWN                              │
│  • Position sizing based on confidence                          │
│  • Risk limits per 15-min window                                │
└─────────────────────────────────────────────────────────────────┘
```

---

#### Performance Expectations

| Market Regime | Expected Accuracy | Notes |
|---------------|-------------------|-------|
| High volatility (>1% daily) | 70-75% | Best conditions |
| Normal volatility | 65-70% | Primary target |
| Low volatility/ranging | 60-65% | Reduce position size |
| Black swan events | Variable | Circuit breakers essential |

---

#### Critical Implementation Notes

1. **Latency Budget:**
   - WebSocket ingestion: <10ms
   - Signal calculation: <50ms
   - Decision logic: <10ms
   - Order execution: <200ms
   - **Total: <500ms end-to-end**

2. **Avoid These Pitfalls:**
   - Overfitting to historical data (recalibrate weights weekly)
   - Chasing sentiment/social media noise (too slow, too noisy)
   - Ignoring exchange outages (implement redundant feeds)
   - Trading during low liquidity (>$50k depth required)

3. **Position Sizing:**
   - Risk ≤1% of portfolio per trade
   - Scale position with confidence score
   - Hard stop if daily loss exceeds 5%

---

#### Signals NOT Suitable for 15-Minute Predictions

| Signal Type | Why Not | Better Use |
|-------------|---------|------------|
| On-Chain (whale movements) | 10-60 min latency | Daily/weekly bias |
| Funding Rates | Updates every 8 hours | 4-24 hour predictions |
| Social Sentiment | Too noisy, lags price | Extreme fear/greed filter |
| News NLP | Processing delay | Major event detection only |

### Whale Watching

Monitor large position changes for signals:

- Leaderboard: `polymarket.com/leaderboard`
- On-chain: Track CTF Exchange events
- Third-party: Various whale tracking services

---

### Whale Copy Trading Strategy (Deep Dive)

Instead of generating algorithmic alpha, follow proven winners. This is a **Tier 2 (Δ≈0)** strategy with lower complexity than pure arbitrage.

#### Known Whale Wallets

| Trader | Wallet Address | Strategy | Track Record |
|--------|---------------|----------|--------------|
| **gabagool22** | `0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d` | News sniping, asymmetric scalping | Top performer |
| **Theo** | Track via leaderboard | Cross-platform arb | High volume |
| **Malte** | Track via leaderboard | Niche/conflict markets | $46K (3 mo) |

**Resources:**
- [gabagool22 Profile](https://polymarket.com/profile/0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d)
- [Polymarket Analytics](https://polymarketanalytics.com/traders/0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d)

---

#### Data Sources for Whale Monitoring

**Option 1: Polymarket Data API (REST Polling)**

```
GET https://data-api.polymarket.com/trades?user={wallet_address}
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `user` | `0x6031b6...` | Whale wallet address |
| `limit` | 100 | Max 10,000 per request |
| `offset` | 0 | For pagination |
| `side` | BUY/SELL | Filter by direction |

**Latency:** 200-500ms per request
**Rate Limit:** ~1,000 calls/hour (free tier)
**Polling Frequency:** Every 3-5 seconds is safe

```python
import requests
import time

WHALE_WALLETS = [
    "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d",  # gabagool22
    # Add more whales...
]

def poll_whale_trades(wallet, last_timestamp=None):
    url = f"https://data-api.polymarket.com/trades"
    params = {
        "user": wallet,
        "limit": 50,
        "takerOnly": "true"
    }

    resp = requests.get(url, params=params)
    trades = resp.json()

    # Filter new trades since last check
    if last_timestamp:
        trades = [t for t in trades if t['timestamp'] > last_timestamp]

    return trades

# Main loop
while True:
    for wallet in WHALE_WALLETS:
        new_trades = poll_whale_trades(wallet)
        for trade in new_trades:
            print(f"WHALE ALERT: {trade['side']} {trade['size']} @ {trade['price']}")
            print(f"  Market: {trade['title']}")
            # Execute copy trade here

    time.sleep(5)  # Poll every 5 seconds
```

---

**Option 2: Polymarket WebSocket (Real-Time)**

Lower latency but requires authentication for user-specific feeds.

```typescript
import { RealTimeDataClient } from '@polymarket/real-time-data-client';

const client = new RealTimeDataClient({
    onMessage: (message) => {
        if (message.type === 'trades') {
            // Check if trade is from watched wallet
            if (WHALE_WALLETS.includes(message.proxyWallet)) {
                console.log('WHALE TRADE:', message);
                // Execute copy trade
            }
        }
    },
    onConnect: () => console.log('Connected')
});

// Subscribe to all trades on specific markets
client.subscribe({
    subscriptions: [{
        topic: "activity",
        type: "trades",
        filters: {} // All trades, filter client-side by wallet
    }]
});

client.connect();
```

**Latency:** ~50-100ms from trade execution
**Limitation:** Can't filter by wallet server-side; must filter all trades client-side

---

**Option 3: On-Chain Monitoring (Fastest)**

Monitor Polygon blockchain directly for `OrderFilled` events.

```python
from web3 import Web3

# Polygon RPC (use Alchemy/Infura for reliability)
w3 = Web3(Web3.HTTPProvider('https://polygon-rpc.com'))

# CTF Exchange contract
CTF_EXCHANGE = '0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E'

# OrderFilled event signature
ORDER_FILLED_TOPIC = w3.keccak(text='OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)')

WHALE_WALLETS = {
    '0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d',  # gabagool22
}

def handle_event(event):
    # Decode event to get maker/taker
    maker = event['topics'][1]  # Extract from indexed params
    taker = event['topics'][2]

    if maker.lower() in WHALE_WALLETS or taker.lower() in WHALE_WALLETS:
        print(f"WHALE TRADE DETECTED: {event['transactionHash'].hex()}")
        # Parse amounts from data field
        # Execute copy trade immediately

# Subscribe to new blocks
def watch_blocks():
    block_filter = w3.eth.filter('latest')

    while True:
        for block_hash in block_filter.get_new_entries():
            block = w3.eth.get_block(block_hash, full_transactions=True)

            for tx in block.transactions:
                if tx['to'] and tx['to'].lower() == CTF_EXCHANGE.lower():
                    receipt = w3.eth.get_transaction_receipt(tx['hash'])

                    for log in receipt['logs']:
                        if log['topics'][0] == ORDER_FILLED_TOPIC:
                            handle_event(log)

        time.sleep(2)  # Polygon block time ~2s
```

**Latency:** 2-4 seconds (1-2 block confirmations)
**Cost:** RPC calls (free tier or $50-100/mo for dedicated node)
**Reliability:** Highest - direct blockchain data

---

**Option 4: Third-Party Services**

| Service | URL | Features | Latency | Cost |
|---------|-----|----------|---------|------|
| [PolyTrack](https://www.polytrackhq.app/) | polytrackhq.app | Whale alerts, Telegram, 20 wallets | ~1 min | $19/mo Pro |
| [Polywhaler](https://www.polywhaler.com/) | polywhaler.com | $10k+ trades, sentiment | ~1-5 min | Free |
| [PolyWallet](https://polymark.et/product/polywallet) | polymark.et | 20 wallet dashboard, Telegram | ~1 min | Varies |

---

#### Latency Comparison

| Method | Detection Latency | Implementation Effort | Best For |
|--------|-------------------|----------------------|----------|
| REST Polling (5s) | 5-10 seconds | Low | Simple copy trading |
| WebSocket | 50-100ms | Medium | Active monitoring |
| On-Chain Direct | 2-4 seconds | High | Maximum reliability |
| Third-Party | 1-5 minutes | None | Casual tracking |

---

#### Copy Trading Bot Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHALE MONITORING LAYER                       │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  REST Poller    │  WebSocket      │  On-Chain Watcher           │
│  (backup)       │  (primary)      │  (confirmation)             │
└────────┬────────┴────────┬────────┴─────────────┬───────────────┘
         │                 │                      │
         ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL DEDUPLICATION                         │
│  • Merge signals from all sources                               │
│  • Dedupe by transaction hash                                   │
│  • Track last known position per whale                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    COPY DECISION ENGINE                         │
│  • Whale reputation score (win rate, ROI)                       │
│  • Position size scaling (% of whale size)                      │
│  • Market liquidity check                                       │
│  • Existing position check (avoid doubling)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                              │
│  • Scale position: YOUR_SIZE = WHALE_SIZE * SCALE_FACTOR        │
│  • Check slippage vs whale's fill price                         │
│  • FOK order with tight slippage tolerance                      │
│  • Log and track for performance analysis                       │
└─────────────────────────────────────────────────────────────────┘
```

---

#### Position Sizing for Copy Trades

```python
def calculate_copy_size(whale_trade, your_portfolio, whale_data):
    """
    Scale copy trade appropriately.
    """
    # Don't blindly match whale size
    whale_size_usd = whale_trade['size'] * whale_trade['price']

    # Option 1: Fixed percentage of your portfolio
    max_position = your_portfolio * 0.05  # 5% max per trade

    # Option 2: Proportional to whale (if you know their portfolio)
    if whale_data.get('estimated_portfolio'):
        whale_pct = whale_size_usd / whale_data['estimated_portfolio']
        proportional_size = your_portfolio * whale_pct
    else:
        proportional_size = max_position

    # Option 3: Scale by whale win rate
    win_rate = whale_data.get('win_rate', 0.5)
    confidence_multiplier = (win_rate - 0.5) * 2 + 1  # 0.6 win rate = 1.2x

    final_size = min(
        proportional_size * confidence_multiplier,
        max_position,
        whale_size_usd * 0.1  # Never more than 10% of whale's trade
    )

    return final_size
```

---

#### Key Considerations

**Advantages of Whale Following:**
- No need to predict outcomes yourself
- Leverages someone else's research/edge
- Lower infrastructure complexity than pure arbitrage
- Works across all market types (not just crypto)

**Risks:**
- **Slippage**: By the time you detect + execute, price may have moved
- **Unknown exits**: You don't know when the whale will exit
- **Whale could be wrong**: Past performance ≠ future results
- **Front-running risk**: If many copy the same whale, edge degrades
- **Capital lock-up**: Whale may hold to resolution, tying up your capital

**Mitigation Strategies:**
1. **Set your own stop-loss** even if whale doesn't exit
2. **Track multiple whales** to diversify signal source
3. **Smaller position sizes** since you don't know their thesis
4. **Exit early** if position moves against you (you're not married to it)
5. **Track whale performance** - only follow consistent winners

---

#### Whale Selection Criteria

| Metric | Threshold | Why |
|--------|-----------|-----|
| Total trades | >500 | Enough sample size |
| Win rate | >55% | Consistently profitable |
| ROI | >20% | Meaningful edge |
| Avg position size | >$1,000 | Serious trader |
| Active in last 30 days | Yes | Currently engaged |
| Market diversity | >3 categories | Not one-trick pony |

```python
def score_whale(wallet_stats):
    """
    Score a whale for copy-worthiness.
    """
    score = 0

    # Win rate (0-30 points)
    if wallet_stats['win_rate'] > 0.6:
        score += 30
    elif wallet_stats['win_rate'] > 0.55:
        score += 20
    elif wallet_stats['win_rate'] > 0.5:
        score += 10

    # Volume (0-20 points)
    if wallet_stats['total_volume'] > 100_000:
        score += 20
    elif wallet_stats['total_volume'] > 50_000:
        score += 10

    # Consistency (0-20 points)
    if wallet_stats['trades_last_30d'] > 50:
        score += 20
    elif wallet_stats['trades_last_30d'] > 20:
        score += 10

    # ROI (0-30 points)
    if wallet_stats['roi'] > 0.5:
        score += 30
    elif wallet_stats['roi'] > 0.2:
        score += 20
    elif wallet_stats['roi'] > 0.1:
        score += 10

    return score  # Max 100

# Only copy whales with score > 60
```

---

## Profitability Data

### Overall Extraction (April 2024 - April 2025)

| Strategy Type | Profit | % of Total |
|--------------|--------|------------|
| Single-condition arbitrage | $10.58M | 27% |
| Multi-condition rebalancing | $28.99M | 73% |
| Cross-market combinatorial | $95K | 0.24% |
| **TOTAL** | **$40M+** | 100% |

### Top Performers

| Rank | Transactions | Total Profit | Avg per Trade |
|------|-------------|--------------|---------------|
| 1 | 4,049 | $2.01M | $496 |
| Top 3 | - | $4.2M | - |
| Top 10 | - | 21% of total | - |

### Distribution
- Only 0.5% of users made >$1,000 profit
- Only 1.7% had trading volume >$50,000
- Power law distribution favors sophisticated infrastructure

---

## Execution Requirements

### Minimum Viable Spreads (After Fees)

| Position Size | Min Spread Required |
|--------------|---------------------|
| $100 | 3.0%+ |
| $500 | 2.5%+ |
| $1,000+ | 2.2%+ |

### Fee Structure

| Platform | Fee Type | Amount |
|----------|----------|--------|
| Polymarket (US) | Trade fee | 0.01% |
| Polymarket (Intl) | Winner fee | 2% on net winnings |
| Kalshi | Trade fee | ~0.7% |
| Polygon | Gas | Variable (~$0.01-0.10) |

### Technical Infrastructure

**API Performance:**
- Kalshi: 20-100 req/sec depending on tier
- Polymarket: ~1,000 calls/hour
- REST latency: 50-200ms typical

**WebSocket Requirements:**
- Monitor 100+ active conditions
- Real-time order book updates
- Sub-second reaction time

**Execution Protection:**
- Sub-5-second timeout for leg risk
- Atomic order placement (simultaneous YES/NO)
- Automatic cancellation of partial fills

**Capital Allocation:**
- $10K-$50K per condition based on liquidity
- 20% max portfolio per single opportunity

---

## Risk Analysis

### Slippage Risk
- Markets move between identification and execution
- Multi-leg trades compound this risk
- **Mitigation:** Size positions based on order book depth

### Leg Risk
- One position fills, hedge fails
- **Mitigation:** Atomic execution, cancel unfilled orders within 5 seconds

### Resolution Risk
- Platform-specific resolution criteria
- Ambiguous or subjective outcomes
- **Mitigation:** Only trade markets with identical, clear resolution criteria

### Oracle Risk
- UMA governance (Polymarket) vs CFTC regulation (Kalshi)
- Potential for manipulation
- **Mitigation:** Avoid cross-platform unless spread > 15¢

### Black Swan Risk
- "Sure things" can reverse (assassination, scandal, legal challenge)
- **Mitigation:** Never assume 100% certainty, maintain position limits

### Competition Risk
- HFT bots capture opportunities in milliseconds
- Server proximity matters
- **Mitigation:** Code efficiency, infrastructure investment

### Regulatory Risk
- Massachusetts sued Kalshi (Sept 2025)
- Polymarket operates under CFTC settlement
- **Mitigation:** Monitor regulatory developments

---

## Bot Configuration Reference

### Environment Variables (.env Example)

Based on working Polymarket trading bots, here's a reference configuration:

```bash
# ===========================================
# POLYMARKET BOT CONFIGURATION
# ===========================================

# --- Wallet & Authentication ---
PRIVATE_KEY=0x...                          # Polygon wallet private key
FUNDER_ADDRESS=0x...                       # Address that funds trades
PROXY_ADDRESS=0x...                        # Proxy wallet (if using)

# --- API Endpoints ---
POLYMARKET_API_URL=https://clob.polymarket.com
POLYMARKET_WS_URL=wss://ws-subscriptions-clob.polymarket.com/ws/market
GAMMA_API_URL=https://gamma-api.polymarket.com

# --- Trading Parameters ---
TARGET_PAIR_COST=0.991                     # Max combined YES+NO cost (99.1¢)
MIN_SPREAD_THRESHOLD=0.02                  # Minimum spread to trade (2%)
MAX_POSITION_SIZE=1000                     # Max USDC per position
MIN_LIQUIDITY=500                          # Min orderbook depth required

# --- Execution Settings ---
USE_WSS=false                              # Use WebSocket (true) or REST polling (false)
ORDER_TYPE=FOK                             # FOK (Fill-or-Kill) or GTC (Good-Till-Cancelled)
SLIPPAGE_TOLERANCE=0.005                   # Max slippage (0.5%)
EXECUTION_TIMEOUT_MS=5000                  # Cancel unfilled orders after 5s

# --- Risk Management ---
MAX_DAILY_LOSS=500                         # Stop trading if daily loss exceeds
MAX_OPEN_POSITIONS=10                      # Maximum concurrent positions
POSITION_SIZE_PCT=0.05                     # 5% of portfolio per trade

# --- Monitoring ---
LOG_LEVEL=INFO                             # DEBUG, INFO, WARN, ERROR
ALERT_WEBHOOK=https://hooks.slack.com/...  # Slack/Discord webhook for alerts
HEARTBEAT_INTERVAL_MS=30000                # Health check interval

# --- Crypto Price Feeds (for 15-min markets) ---
BINANCE_WS_URL=wss://stream.binance.com:9443/ws
COINBASE_WS_URL=wss://ws-feed.exchange.coinbase.com
PRICE_UPDATE_INTERVAL_MS=100               # Price poll frequency

# --- Market Filters ---
MARKET_TYPES=crypto,politics,sports        # Which market types to trade
EXCLUDED_MARKETS=                          # Comma-separated market IDs to skip
MIN_TIME_TO_RESOLUTION_HOURS=1             # Don't trade markets resolving soon
```

### Key Configuration Decisions

| Parameter | Conservative | Aggressive |
|-----------|-------------|------------|
| TARGET_PAIR_COST | 0.985 | 0.995 |
| MIN_SPREAD_THRESHOLD | 0.03 | 0.015 |
| MAX_POSITION_SIZE | 500 | 5000 |
| SLIPPAGE_TOLERANCE | 0.003 | 0.01 |
| MAX_OPEN_POSITIONS | 5 | 20 |

### WebSocket vs REST Polling

**USE_WSS=true (WebSocket)**
- Real-time orderbook updates
- Lower latency (~50ms)
- Higher infrastructure complexity
- Better for high-frequency strategies

**USE_WSS=false (REST Polling)**
- Simpler implementation
- Higher latency (~200-500ms)
- More reliable (no connection drops)
- Sufficient for most arbitrage

---

## Tools & Resources

### Arbitrage Scanners

| Tool | URL | Function |
|------|-----|----------|
| **EventArb** | [eventarb.com](https://www.eventarb.com/) | Cross-platform arbitrage calculator |
| **BetMoar** | [betmoar.fun](https://betmoar.fun) | Multi-outcome arbitrage finder |
| **PizzintWatch** | Community tool | Whale/position tracking |

### Analytics & Tracking

| Tool | Function |
|------|----------|
| **Polymarket Leaderboard** | Top trader tracking |
| **PolyTrack** | Market analytics and alerts |
| **Dune Analytics** | On-chain data queries |

### Development Resources

| Resource | Purpose |
|----------|---------|
| [py-clob-client](https://github.com/Polymarket/py-clob-client) | Python trading client |
| [Polymarket Agents](https://github.com/Polymarket/agents) | AI bot framework |
| [ctf-exchange](https://github.com/Polymarket/ctf-exchange) | Smart contract reference |

---

## Sources

### Official Documentation
- [Polymarket Documentation](https://docs.polymarket.com/)
- [Polymarket CLOB Introduction](https://docs.polymarket.com/developers/CLOB/introduction)
- [Polymarket Gamma API](https://docs.polymarket.com/developers/gamma-markets-api/overview)
- [Polymarket NegRisk Overview](https://docs.polymarket.com/developers/neg-risk/overview)
- [Kalshi API Documentation](https://docs.kalshi.com/welcome)
- [Kalshi API Keys](https://docs.kalshi.com/getting_started/api_keys)
- [PredictIt API Info](https://predictit.freshdesk.com/support/solutions/articles/12000001878-does-predictit-make-market-data-available-via-an-api-)

### Client Libraries
- [Polymarket py-clob-client](https://github.com/Polymarket/py-clob-client)
- [Kalshi Python SDK](https://pypi.org/project/kalshi-python/)
- [Polymarket Agents Framework](https://github.com/Polymarket/agents)

### Research & Analysis
- [Academic Paper: Unravelling the Probabilistic Forest](https://arxiv.org/abs/2508.03474)
- [Jeremy Whittaker: Arbitrage in Polymarket](https://jeremywhittaker.com/index.php/2024/09/24/arbitrage-in-polymarket-com/)
- [Building a Prediction Market Arbitrage Bot](https://navnoorbawa.substack.com/p/building-a-prediction-market-arbitrage)
- [Combinatorial Arbitrage Analysis](https://medium.com/@navnoorbawa/combinatorial-arbitrage-in-prediction-markets-why-62-of-llm-detected-dependencies-fail-to-26f614804e8d)

### Tools & Bots
- [Polymarket-Kalshi Arbitrage Bot](https://github.com/terauss/Polymarket-Kalshi-Arbitrage-bot)
- [BTC Arbitrage Bot](https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot)
- [EventArb Calculator](https://www.eventarb.com/)
- [Polymarket Spike Bot](https://github.com/Trust412/Polymarket-spike-bot-v1)

### Market Analysis
- [Polymarket Arbitrage Guide 2025](https://www.polytrackhq.app/blog/polymarket-arbitrage-guide)
- [Polymarket Arbitrage Bot Guide](https://www.polytrackhq.app/blog/polymarket-arbitrage-bot-guide)
- [11 Arbitrage Strategies Explained](https://www.odaily.news/en/post/5205900)
- [People Making Fortune on Polymarket](https://www.chaincatcher.com/en/article/2212288)
- [Arbitrage Traders Turn Polymarket Into Profit Engine](https://beincrypto.com/polymarket-arbitrage-risk-free-profit/)
- [@0xMovez: Top 10 Strategies to Make Money on Polymarket](https://x.com/0xMovez/status/2004570871294239187) - Comprehensive strategy guide covering niche trading, spreads, rewards farming, bonding, arbitrage (5 types), bot trading, copy-trading, dispute trading, mention markets, and news trading

### Platform Comparisons
- [Polymarket vs Kalshi Analysis](https://phemex.com/blogs/polymarket-vs-kalshi-prediction-markets-analysis)
- [Prediction Market API Guide 2025](https://predictorsbest.com/prediction-market-api-guide/)
