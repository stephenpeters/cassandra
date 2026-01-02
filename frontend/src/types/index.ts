// Type definitions for the whale tracker

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface TradeData {
  time: number;
  price: number;
  size: number;
  side: "BUY" | "SELL";
}

export interface OrderBookData {
  mid: number;
  spread: number;
  imbalance: number;
  bids: [number, number][];
  asks: [number, number][];
}

export interface MomentumSignal {
  direction: "UP" | "DOWN" | "NEUTRAL";
  confidence: number;
  volume_delta: number;
  price_change_pct: number;
  orderbook_imbalance: number;
}

export interface WhaleTrade {
  whale: string;
  wallet: string;
  market: string;
  slug: string;
  outcome: string;
  side: "BUY" | "SELL";
  size: number;
  price: number;
  usd_value: number;
  timestamp: number;
  tx_hash: string;
  icon: string;
}

export interface WhaleInfo {
  name: string;
  address: string;
  strategy?: string;
  focus?: string[];
}

export interface WebSocketMessage {
  type:
    | "init"
    | "trade"
    | "candle"
    | "orderbook"
    | "momentum"
    | "whale_trade"
    | "candles_snapshot"
    | "ping"
    | "pong";
  symbol?: string;
  data?: unknown;
  whales?: WhaleInfo[];
  symbols?: string[];
  candles?: CandleData[];
}
