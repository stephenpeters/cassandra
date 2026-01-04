# Cassandra - Polymarket Latency Arbitrage Trading System

Real-time trading system for Polymarket 15-minute crypto prediction markets. Exploits the 30-60 second latency gap between Binance price movements and Polymarket price adjustments.

## Strategy Overview

Polymarket offers 15-minute binary options on crypto prices (BTC, ETH, etc.) - "Will BTC go UP or DOWN in the next 15 minutes?"

**The Edge:** Polymarket prices lag Binance by 30-60 seconds. When Binance moves sharply, we can buy before Polymarket adjusts.

```
Binance price jumps +0.3%
  -> Implied UP probability: ~65%
  -> Polymarket UP price: still 52 cents
  -> Edge: 13%
  -> Execute trade
```

## Features

- **Latency Arbitrage Engine** - Detects price gaps between Binance and Polymarket
- **Paper Trading** - Test strategies with simulated money
- **Live Trading** - Real execution via Polymarket CLOB API (requires wallet)
- **Real-time Dashboard** - TradingView charts, order books, momentum signals
- **Whale Monitoring** - Track large traders (gabagool22, Theo cluster, etc.)
- **Risk Management** - Circuit breakers, kill switch, position limits
- **Alerts** - Telegram/Discord notifications for trades and signals

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                    │
│  - TradingView charts    - Paper/Live trading cards         │
│  - Order book display    - Whale trades table               │
│  - Momentum signals      - Theme toggle                      │
└─────────────────────────────────────────────────────────────┘
                              │ WebSocket
┌─────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Binance Feed │  │  Polymarket  │  │ Whale Tracker│       │
│  │  (WebSocket) │  │    Feed      │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│          │                │                  │               │
│          └────────────────┴──────────────────┘               │
│                           │                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Latency Arbitrage Engine                 │   │
│  │  - Record window open prices                         │   │
│  │  - Calculate implied probability from price move      │   │
│  │  - Compare to Polymarket price                        │   │
│  │  - Execute when edge > threshold                      │   │
│  └──────────────────────────────────────────────────────┘   │
│          │                                                   │
│  ┌───────┴───────┐                                          │
│  │               │                                          │
│  ▼               ▼                                          │
│ Paper          Live                                         │
│ Trading        Trading                                      │
│ Engine         Engine                                       │
│                  │                                          │
│                  ▼                                          │
│            Polymarket                                       │
│            CLOB API                                         │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone and start
git clone https://github.com/stephenpeters/cassandra.git
cd cassandra
chmod +x start.sh
./start.sh
```

**Access:**
- Dashboard: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Trading Modes

| Mode     | Description                                    |
|----------|------------------------------------------------|
| `paper`  | Simulated trades only - no real money          |
| `shadow` | Live signals, paper execution - for validation |
| `live`   | Real money trades via Polymarket CLOB          |

## Configuration

Copy and configure the environment file:

```bash
cp backend/.env.example backend/.env
```

Key settings:

```env
TRADING_MODE=paper                    # paper/shadow/live
POLYMARKET_PRIVATE_KEY=your_key       # Required for live trading
TELEGRAM_BOT_TOKEN=your_token         # For alerts
TELEGRAM_CHAT_ID=your_chat_id
API_KEY=your_secret_key               # Protects live trading endpoints
```

## API Endpoints

### Public

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Server status + trading engine health |
| `GET /api/candles/{symbol}` | Historical OHLCV data |
| `GET /api/momentum` | Current momentum signals |
| `GET /api/markets-15m` | Active 15-minute markets |
| `GET /api/paper-trading/account` | Paper trading status |

### Protected (require `X-API-Key` header)

| Endpoint | Description |
|----------|-------------|
| `GET /api/live-trading/status` | Live trading status |
| `POST /api/live-trading/mode` | Switch trading mode |
| `POST /api/live-trading/kill-switch` | Emergency stop |
| `GET /api/live-trading/allowances` | Check token allowances |
| `POST /api/live-trading/allowances` | Set token allowances |

### WebSocket (`ws://localhost:8000/ws`)

Real-time streams: `candle`, `trade`, `orderbook`, `momentum`, `whale_trade`, `paper_signal`, `paper_account`, `live_order`, `live_alert`

## Production Deployment

See [backend/production/aws-lightsail-setup.md](backend/production/aws-lightsail-setup.md) for deployment guide.

**Important:** Deploy to Ireland (eu-west-1) - not geoblocked and low latency to Polymarket.

## Project Structure

```text
cassandra/
├── backend/
│   ├── server.py           # FastAPI WebSocket server
│   ├── paper_trading.py    # Paper trading engine
│   ├── live_trading.py     # Live trading with CLOB
│   ├── data_feeds.py       # Binance + Polymarket feeds
│   ├── config.py           # Whale wallets, settings
│   └── production/         # Deployment configs
├── frontend/
│   ├── src/
│   │   ├── app/            # Next.js pages
│   │   ├── components/     # React components
│   │   ├── hooks/          # useWebSocket
│   │   └── types/          # TypeScript definitions
│   └── package.json
├── archive/                # Research & backtesting (historical)
├── start.sh                # Startup script
└── README.md
```

## Risk Disclaimers

- **This is experimental software** - use at your own risk
- Paper trade extensively before enabling live mode
- Start with small position sizes
- Never trade with money you can't afford to lose
- Check local regulations on prediction market trading

## License

MIT
