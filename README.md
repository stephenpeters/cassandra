# Polymarket Whale Tracker

Real-time dashboard for tracking whale trades on Polymarket prediction markets with live crypto price feeds, order book visualization, and momentum signals.

## Features

- **Whale Trade Monitoring** - Track large traders (gabagool22, Theo cluster, Account88888, etc.) across Polymarket markets
- **Live Price Charts** - TradingView-style candlestick charts with Binance WebSocket feeds
- **Order Book Visualization** - Real-time bid/ask depth display
- **Momentum Signals** - Technical analysis with volume delta, price momentum, and order book imbalance
- **Market Filtering** - Crypto 15-minute markets, politics, sports, or all markets
- **Multi-Symbol Support** - BTC, ETH, SOL, XRP, DOGE

## Tech Stack

**Frontend**
- Next.js 16 / React 19
- TypeScript
- Tailwind CSS + Radix UI
- lightweight-charts (TradingView)

**Backend**
- Python 3.11+ / FastAPI
- WebSocket (real-time updates)
- Binance & Polymarket API integration

## Quick Start

```bash
# Clone and start both services
chmod +x start.sh
./start.sh
```

Or run manually:

```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

**Access:**
- Dashboard: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Project Structure

```
predmkt/
├── backend/
│   ├── server.py          # FastAPI WebSocket & REST server
│   ├── config.py          # Whale wallets, API endpoints
│   └── data_feeds.py      # Binance feeds, whale tracking
├── frontend/
│   ├── src/
│   │   ├── app/           # Next.js pages
│   │   ├── components/    # React components
│   │   │   ├── charts/    # PriceChart, OrderBook, Momentum
│   │   │   └── WhaleTradesTable.tsx
│   │   ├── hooks/         # useWebSocket
│   │   └── types/         # TypeScript definitions
│   └── package.json
├── whale_tracker.py       # Standalone CLI tool
├── start.sh               # Startup script
└── PRD.md                 # Product requirements
```

## API

### REST Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Server status |
| `GET /api/whales` | List tracked whales |
| `GET /api/candles/{symbol}` | Historical candles |
| `GET /api/whale-trades` | Recent whale trades |

### WebSocket (`ws://localhost:8000/ws`)

Message types: `init`, `candle`, `trade`, `orderbook`, `momentum`, `whale_trade`

## Configuration

Environment variables in `frontend/.env.local`:

```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Whale wallets and API settings in `backend/config.py`.

## Tracked Whales

| Whale | Strategy | Focus |
|-------|----------|-------|
| gabagool22 | News sniping, asymmetric scalping | Crypto, Politics |
| Fredi9999 (Theo cluster) | Political high-stakes | Politics |
| Account88888 | Sequential entry arbitrage | Crypto 15-min |

## CLI Tool

Standalone whale tracker with strategy filtering:

```bash
python whale_tracker.py --strategy crypto   # Default: 15-min crypto markets
python whale_tracker.py --strategy politics # Political markets
python whale_tracker.py --strategy sports   # Sports markets
python whale_tracker.py --strategy all      # All markets
```

## License

MIT
