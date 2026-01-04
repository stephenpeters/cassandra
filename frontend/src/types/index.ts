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

// 15-minute market types
export interface Market15Min {
  condition_id: string;
  token_id: string;
  question: string;
  symbol: string;
  outcome: string;
  start_time: number;
  end_time: number;
  price: number;
  volume: number;
  is_active: boolean;
}

export interface MarketTrade {
  condition_id: string;
  symbol: string;
  outcome: string;
  side: "BUY" | "SELL";
  size: number;
  price: number;
  usd_value: number;
  timestamp: number;
  maker: string;
  taker: string;
}

export interface MarketTiming {
  start: number;
  end: number;
  time_until_start: number;
  is_open: boolean;
}

export interface Markets15MinData {
  active: Record<string, Market15Min>;
  timing: MarketTiming;
  trades: MarketTrade[];
}

// Paper Trading types
export type PaperSignalType = "HOLD" | "BUY_UP" | "BUY_MORE_UP" | "BUY_DOWN" | "BUY_MORE_DOWN";

export interface PaperTradingConfig {
  enabled: boolean;
  starting_balance: number;
  slippage_pct: number;
  commission_pct: number;
  max_position_pct: number;
  max_position_usd: number;  // Hard cap at $5K (liquidity constraint)
  daily_loss_limit_pct: number;
  enabled_assets: string[];  // ["BTC", "ETH"] - hot-reloadable
}

export interface PaperPosition {
  id: string;
  symbol: string;
  side: "UP" | "DOWN";
  entry_price: number;
  size: number;
  cost_basis: number;
  entry_time: number;
  market_start: number;
  market_end: number;
  checkpoint: string;
}

export interface PaperTrade {
  id: string;
  symbol: string;
  side: "UP" | "DOWN";
  entry_price: number;
  exit_price: number;
  size: number;
  cost_basis: number;
  settlement_value: number;
  pnl: number;
  pnl_pct: number;
  entry_time: number;
  exit_time: number;
  market_start: number;
  market_end: number;
  resolution: "UP" | "DOWN";
  binance_open: number;
  binance_close: number;
  checkpoint: string;
  signal_confidence: number;
}

export interface PaperSignal {
  symbol: string;
  checkpoint: string;
  timestamp: number;
  signal: PaperSignalType;
  fair_value: number;
  market_price: number;
  edge: number;
  confidence: number;
  momentum: {
    direction: string;
    confidence: number;
    volume_delta: number;
    price_change_pct: number;
    orderbook_imbalance: number;
  };
}

export interface PaperAccount {
  balance: number;
  starting_balance: number;
  total_pnl: number;
  today_pnl: number;
  total_return_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  positions: PaperPosition[];
  recent_trades: PaperTrade[];
  trading_halted: boolean;
  halt_reason: string;
}

// Live Trading types
export type TradingMode = "paper" | "live";

export interface LiveTradingConfig {
  mode: TradingMode;
  max_position_usd: number;
  max_daily_volume_usd: number;
  max_open_positions: number;
  max_consecutive_losses: number;
  daily_loss_limit_usd: number;
  max_drawdown_pct: number;
  order_type: "GTC" | "FOK" | "GTD";
  max_slippage_pct: number;
  require_manual_confirm: boolean;
  min_signal_confidence: number;
  enabled_assets: string[];
}

export interface CircuitBreaker {
  triggered: boolean;
  reason: string;
  triggered_at: number | null;
  consecutive_losses: number;
  daily_loss_usd: number;
  daily_volume_usd: number;
  peak_balance_usd: number;
  current_balance_usd: number;
  last_reset_date: string;
}

export interface WalletBalance {
  usdc_balance: number;
  allowance: number;
  collateral_locked: number;
  total_value: number;
  available_for_trading: number;
  error?: string;
}

export interface LiveOrder {
  id: string;
  symbol: string;
  side: "UP" | "DOWN";
  direction: "BUY" | "SELL";
  token_id: string;
  size_usd: number;
  price: number;
  order_type: string;
  status: "pending" | "pending_confirmation" | "submitted" | "filled" | "partial" | "cancelled" | "failed" | "rejected" | "paper";
  created_at: number;
  filled_at: number | null;
  filled_size: number;
  filled_price: number;
  error: string | null;
  polymarket_order_id: string | null;
  tx_hash: string | null;
}

export interface LivePosition {
  symbol: string;
  side: "UP" | "DOWN";
  token_id: string;
  size: number;
  avg_entry_price: number;
  cost_basis_usd: number;
  market_start: number;
  market_end: number;
  entry_orders: string[];
}

export interface LiveTradingStatus {
  mode: TradingMode;
  kill_switch_active: boolean;
  circuit_breaker: CircuitBreaker;
  open_positions: number;
  enabled_assets: string[];
  clob_connected: boolean;
  wallet?: WalletBalance;
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
    | "markets_15m"
    | "market_update"
    | "market_trade"
    | "paper_account"
    | "paper_signal"
    | "paper_trade"
    | "paper_position"
    | "live_order"
    | "live_fill"
    | "live_alert"
    | "ping"
    | "pong";
  symbol?: string;
  data?: unknown;
  whales?: WhaleInfo[];
  symbols?: string[];
  candles?: CandleData[];
  paper_trading?: PaperAccount;
  live_trading?: LiveTradingStatus;
}
