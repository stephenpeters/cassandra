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
  // Technical indicators
  vwap?: number;
  vwap_signal?: "UP" | "DOWN" | "NEUTRAL";
  rsi?: number;
  adx?: number;
  supertrend_direction?: "UP" | "DOWN" | "NEUTRAL";
  supertrend_value?: number;
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
  down_price?: number;  // RT DOWN price (updated every 500ms)
  binance_price?: number | null;  // RT Binance price (updated every 500ms)
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

export interface MarketWindowDataPoint {
  time: number;  // Unix timestamp in seconds
  binancePrice: number;
  upPrice: number;
  downPrice: number;
}

export interface MarketWindowChartData {
  symbol: string;
  start_price: number;  // Binance price at market open (snake_case from backend)
  data: MarketWindowDataPoint[];
}

export interface Markets15MinData {
  active: Record<string, Market15Min>;
  timing: MarketTiming;
  trades: MarketTrade[];
  chart_data?: Record<string, MarketWindowChartData>;  // Chart data per symbol
}

// Trading types
export type SignalType = "HOLD" | "BUY_UP" | "BUY_MORE_UP" | "BUY_DOWN" | "BUY_MORE_DOWN";

export interface TradingConfig {
  enabled: boolean;
  starting_balance: number;
  slippage_pct: number;
  commission_pct: number;
  max_position_pct: number;
  max_position_usd: number;  // Hard cap at $5K (liquidity constraint)
  daily_loss_limit_pct: number;
  enabled_assets: string[];  // ["BTC", "ETH"] - hot-reloadable

  // Latency arbitrage settings
  min_edge_pct: number;  // Minimum edge to trigger (default 5%)
  min_time_remaining_sec: number;  // Don't trade in last N seconds (default 120)
  cooldown_sec: number;  // Seconds between trades per symbol (default 30)

  // Checkpoint configuration (seconds into 15-min window)
  // Default: 180 (3m), 300 (5m), 450 (7m30s), 540 (9m), 720 (12m)
  signal_checkpoints: number[];  // Checkpoints to generate signals at
  active_checkpoint: number;  // Checkpoint to execute trades at (default 450 = 7m30s)

  // Legacy entry timing (kept for backwards compatibility)
  entry_time_up_sec: number;  // When to consider UP entries (default 450 = 7m30s)
  entry_time_down_sec: number;  // When to consider DOWN entries (default 450 = 7m30s)

  // Confirmation requirements (legacy - kept for backwards compatibility)
  require_volume_confirmation: boolean;
  require_orderbook_confirmation: boolean;
  min_volume_delta_usd: number;  // Minimum volume delta (default 10000)
  min_orderbook_imbalance: number;  // Minimum imbalance (default 0.1)

  // TIERED CONFIRMATION SYSTEM
  min_confirmations: number;  // Minimum confirmations to trade (default 2)
  partial_size_pct: number;  // Position size % for partial confirmation (default 50)
  edge_mandatory: boolean;  // Whether edge is required for all trades

  // Indicator toggles
  use_edge: boolean;
  use_volume_delta: boolean;
  use_orderbook: boolean;
  use_vwap: boolean;
  use_rsi: boolean;
  use_adx: boolean;
  use_supertrend: boolean;

  // Indicator thresholds
  rsi_oversold: number;  // RSI threshold for oversold (default 30)
  rsi_overbought: number;  // RSI threshold for overbought (default 70)
  adx_trend_threshold: number;  // ADX threshold for strong trend (default 25)
  supertrend_multiplier: number;  // Supertrend ATR multiplier (default 3.0)
}

export interface Position {
  id: string;
  symbol: string;
  slug: string;  // e.g., "btc-updown-15m-1736200800"
  side: "UP" | "DOWN";
  entry_price: number;
  size: number;
  cost_basis: number;
  entry_time: number;
  market_start: number;
  market_end: number;
  checkpoint: string;
}

export interface Trade {
  id: string;
  symbol: string;
  slug: string;  // e.g., "btc-updown-15m-1736200800"
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

export interface TradingSignal {
  symbol: string;
  slug: string;  // e.g., "btc-updown-15m-1736200800"
  checkpoint: string;
  timestamp: number;
  signal: SignalType;
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

export interface TradingAccount {
  balance: number;
  starting_balance: number;
  total_pnl: number;
  today_pnl: number;
  total_return_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  positions: Position[];
  recent_trades: Trade[];
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

// Sniper Strategy types
export type SniperStatusType = "disabled" | "skip" | "waiting" | "position_taken" | "ready" | "no_signal";

export interface SniperEvaluation {
  side: "UP" | "DOWN";
  price: number;
  ev_pct: number;
  in_range: boolean;
  ev_ok: boolean;
}

export interface SniperStatus {
  status: SniperStatusType;
  reason: string;
  symbol: string;
  market_start?: number;
  market_end?: number;
  timestamp?: number;
  elapsed_sec?: number;
  min_elapsed_sec?: number;
  time_remaining?: number;
  signal?: "UP" | "DOWN";
  entry_price?: number;
  ev_pct?: number;
  evaluations?: SniperEvaluation[];
}

export interface SniperSignal {
  symbol: string;
  signal: "UP" | "DOWN";
  entry_price: number;
  elapsed_sec: number;
  position_size: number;
  market_start: number;
  market_end: number;
  ev_pct?: number;
}

// Real-time price update from Polymarket WebSocket
export interface PMPriceUpdate {
  asset_id: string;
  symbol: string;        // BTC, ETH, etc
  outcome: "UP" | "DOWN";
  price: number;
  best_bid: number;
  best_ask: number;
  timestamp: number;
  update_type: "price_change" | "last_trade_price" | "book";
}

// Mode status from backend mode_controller
export type TradingModeValue = "live" | "paper" | "off";

export interface ModeStatus {
  mode: TradingModeValue;
  is_trading_enabled: boolean;
  is_live: boolean;
  changed_at: number;
  changed_by: string;
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
    | "chart_update"
    | "paper_account"
    | "paper_signal"
    | "paper_trade"
    | "paper_position"
    | "live_order"
    | "live_fill"
    | "live_alert"
    | "live_status"
    | "sniper_status"
    | "sniper_signal"
    | "pm_price_update"
    | "mode_update"
    | "ping"
    | "pong";
  symbol?: string;
  data?: unknown;
  whales?: WhaleInfo[];
  symbols?: string[];
  candles?: CandleData[];
  paper_trading?: TradingAccount;
  live_trading?: LiveTradingStatus;
}
