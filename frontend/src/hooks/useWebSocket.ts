"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import type {
  WebSocketMessage,
  CandleData,
  TradeData,
  OrderBookData,
  MomentumSignal,
  WhaleTrade,
  WhaleInfo,
  Markets15MinData,
  MarketTrade,
  MarketWindowChartData,
  TradingAccount,
  TradingSignal,
  TradingConfig,
  LiveTradingStatus,
  SniperStatus,
  SniperSignal,
  PMPriceUpdate,
  ModeStatus,
  TradingModeValue,
} from "@/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Re-export for convenience
export type { LiveTradingStatus };

export interface UseWebSocketReturn {
  isConnected: boolean;
  candles: Record<string, CandleData[]>;
  trades: Record<string, TradeData[]>;
  orderbooks: Record<string, OrderBookData>;
  momentum: Record<string, MomentumSignal>;
  whaleTrades: WhaleTrade[];
  whales: WhaleInfo[];
  symbols: string[];
  markets15m: Markets15MinData | null;
  marketTrades: MarketTrade[];
  chartData: Record<string, MarketWindowChartData>;
  paperAccount: TradingAccount | null;
  paperSignals: TradingSignal[];
  paperConfig: TradingConfig | null;
  liveStatus: LiveTradingStatus | null;
  currentMode: TradingModeValue;  // 3-way mode: off/paper/live
  sniperStatus: Record<string, SniperStatus>;  // Status per symbol
  sniperSignals: SniperSignal[];  // Recent signals
  requestCandles: (symbol: string) => void;
  togglePaperTrading: () => Promise<void>;
  resetTradingAccount: () => Promise<void>;
  factoryReset: () => Promise<void>;
  updatePaperConfig: (config: Partial<TradingConfig>) => Promise<void>;
  setTradingMode: (mode: TradingModeValue, apiKey: string) => Promise<{ success: boolean; error?: string }>;
  placeTestOrder: (symbol: string, side: "UP" | "DOWN", amount: number, apiKey: string) => Promise<{ success: boolean; error?: string; order?: unknown }>;
  setAllowances: (apiKey: string) => Promise<{ success: boolean; error?: string }>;
}

export function useWebSocket(): UseWebSocketReturn {
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [candles, setCandles] = useState<Record<string, CandleData[]>>({});
  const [trades, setTrades] = useState<Record<string, TradeData[]>>({});
  const [orderbooks, setOrderbooks] = useState<Record<string, OrderBookData>>(
    {}
  );
  const [momentum, setMomentum] = useState<Record<string, MomentumSignal>>({});
  const [whaleTrades, setWhaleTrades] = useState<WhaleTrade[]>([]);
  const [whales, setWhales] = useState<WhaleInfo[]>([]);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [markets15m, setMarkets15m] = useState<Markets15MinData | null>(null);
  const [marketTrades, setMarketTrades] = useState<MarketTrade[]>([]);
  const [chartData, setChartData] = useState<Record<string, MarketWindowChartData>>({});
  const [paperAccount, setTradingAccount] = useState<TradingAccount | null>(null);
  const [paperSignals, setTradingSignals] = useState<TradingSignal[]>([]);
  const [paperConfig, setPaperConfig] = useState<TradingConfig | null>(null);
  const [liveStatus, setLiveStatus] = useState<LiveTradingStatus | null>(null);
  const [currentMode, setCurrentMode] = useState<TradingModeValue>("paper");
  const [sniperStatus, setSniperStatus] = useState<Record<string, SniperStatus>>({});
  const [sniperSignals, setSniperSignals] = useState<SniperSignal[]>([]);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;

    try {
      ws.current = new WebSocket(WS_URL);

      ws.current.onopen = () => {
        console.log("[WS] Connected");
        setIsConnected(true);
      };

      ws.current.onclose = () => {
        console.log("[WS] Disconnected");
        setIsConnected(false);

        // Reconnect after 3 seconds
        reconnectTimeout.current = setTimeout(connect, 3000);
      };

      ws.current.onerror = (error) => {
        console.error("[WS] Error:", error);
      };

      ws.current.onmessage = (event) => {
        try {
          const msg: WebSocketMessage = JSON.parse(event.data);
          handleMessage(msg);
        } catch (e) {
          console.error("[WS] Parse error:", e);
        }
      };
    } catch (e) {
      console.error("[WS] Connection error:", e);
      reconnectTimeout.current = setTimeout(connect, 3000);
    }
  }, []);

  const handleMessage = useCallback((msg: WebSocketMessage) => {
    switch (msg.type) {
      case "init":
        if (msg.whales) setWhales(msg.whales);
        if (msg.symbols) setSymbols(msg.symbols);
        if (msg.paper_trading) setTradingAccount(msg.paper_trading);
        if (msg.live_trading) setLiveStatus(msg.live_trading as LiveTradingStatus);
        break;

      case "candle":
        if (msg.symbol && msg.data) {
          const candle = msg.data as CandleData;
          setCandles((prev) => {
            const current = prev[msg.symbol!] || [];
            // Update last candle if same timestamp, otherwise append
            if (
              current.length > 0 &&
              current[current.length - 1].time === candle.time
            ) {
              return {
                ...prev,
                [msg.symbol!]: [...current.slice(0, -1), candle],
              };
            }
            // Limit to 200 candles (reduced from 500 for memory)
            return {
              ...prev,
              [msg.symbol!]: [...current.slice(-199), candle],
            };
          });
        }
        break;

      case "candles_snapshot":
        if (msg.symbol && msg.candles) {
          setCandles((prev) => ({
            ...prev,
            [msg.symbol!]: msg.candles!,
          }));
        }
        break;

      case "trade":
        if (msg.symbol && msg.data) {
          const trade = msg.data as TradeData;
          setTrades((prev) => {
            const current = prev[msg.symbol!] || [];
            // Limit to 50 trades per symbol (reduced from 100 for memory)
            return {
              ...prev,
              [msg.symbol!]: [...current.slice(-49), trade],
            };
          });
        }
        break;

      case "orderbook":
        if (msg.symbol && msg.data) {
          setOrderbooks((prev) => ({
            ...prev,
            [msg.symbol!]: msg.data as OrderBookData,
          }));
        }
        break;

      case "momentum":
        if (msg.data) {
          setMomentum(msg.data as Record<string, MomentumSignal>);
        }
        break;

      case "whale_trade":
        if (msg.data) {
          const trade = msg.data as WhaleTrade;
          // Limit to 50 whale trades (reduced from 100 for memory)
          setWhaleTrades((prev) => [trade, ...prev.slice(0, 49)]);
        }
        break;

      case "markets_15m":
        if (msg.data) {
          const data = msg.data as Markets15MinData;
          setMarkets15m(data);
          if (data.trades) {
            setMarketTrades(data.trades);
          }
          if (data.chart_data) {
            setChartData(data.chart_data as Record<string, MarketWindowChartData>);
          }
        }
        break;

      case "market_trade":
        if (msg.data) {
          const trade = msg.data as MarketTrade;
          // Limit to 50 market trades (reduced from 100 for memory)
          setMarketTrades((prev) => [trade, ...prev.slice(0, 49)]);
        }
        break;

      case "chart_update":
        // Fast chart updates (every 1s) for smoother rendering
        if (msg.data) {
          setChartData(msg.data as Record<string, MarketWindowChartData>);
        }
        break;

      case "paper_account":
        if (msg.data) {
          setTradingAccount(msg.data as TradingAccount);
        }
        break;

      case "paper_signal":
        if (msg.data) {
          const signal = msg.data as TradingSignal;
          // Limit to 20 paper signals (reduced from 50 for memory)
          setTradingSignals((prev) => [signal, ...prev.slice(0, 19)]);
        }
        break;

      case "paper_trade":
        // Trade completed - account will be updated via paper_account message
        break;

      case "paper_position":
        // Position opened - account will be updated via paper_account message
        break;

      case "live_status":
        // Updated live trading status (sent when mode changes)
        if (msg.data) {
          setLiveStatus(msg.data as LiveTradingStatus);
        }
        break;

      case "mode_update":
        // 3-way mode update from backend (off/paper/live)
        if (msg.data) {
          const modeStatus = msg.data as ModeStatus;
          setCurrentMode(modeStatus.mode);
        }
        break;

      case "sniper_status":
        // Sniper strategy evaluation status per symbol
        if (msg.data) {
          const status = msg.data as SniperStatus;
          setSniperStatus((prev) => ({
            ...prev,
            [status.symbol]: status,
          }));
        }
        break;

      case "sniper_signal":
        // Sniper strategy triggered a signal
        if (msg.data) {
          const signal = msg.data as SniperSignal;
          setSniperSignals((prev) => [signal, ...prev.slice(0, 19)]);
        }
        break;

      case "pm_price_update":
        // Real-time price update from Polymarket WebSocket
        if (msg.data) {
          const update = msg.data as PMPriceUpdate;

          // Update market price badges
          setMarkets15m((prev) => {
            if (!prev?.active?.[update.symbol]) return prev;
            const market = prev.active[update.symbol];
            const newPrice = update.outcome === "UP" ? update.price : market.price;
            return {
              ...prev,
              active: {
                ...prev.active,
                [update.symbol]: { ...market, price: newPrice },
              },
            };
          });

          // Update chart data in real-time (update last point)
          setChartData((prev) => {
            const symbolData = prev[update.symbol];
            if (!symbolData?.data?.length) return prev;

            const data = [...symbolData.data];
            const lastPoint = { ...data[data.length - 1] };

            // Update the appropriate price
            if (update.outcome === "UP") {
              lastPoint.upPrice = update.price;
            } else {
              lastPoint.downPrice = update.price;
            }

            data[data.length - 1] = lastPoint;
            return {
              ...prev,
              [update.symbol]: { ...symbolData, data },
            };
          });
        }
        break;

      case "ping":
        ws.current?.send(JSON.stringify({ type: "pong" }));
        break;
    }
  }, []);

  const requestCandles = useCallback((symbol: string) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(
        JSON.stringify({
          type: "get_candles",
          symbol: symbol,
        })
      );
    }
  }, []);

  // Paper trading API functions
  const togglePaperTrading = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/paper-trading/toggle`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        setPaperConfig((prev) =>
          prev ? { ...prev, enabled: data.enabled } : null
        );
      }
    } catch (e) {
      console.error("Failed to toggle paper trading:", e);
    }
  }, []);

  const resetTradingAccount = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/paper-trading/reset`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        setTradingAccount(data.account);
        setTradingSignals([]);
      }
    } catch (e) {
      console.error("Failed to reset paper account:", e);
    }
  }, []);

  // Factory reset - resets both account AND config to defaults
  const factoryReset = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/paper-trading/factory-reset`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        setTradingAccount(data.account);
        setPaperConfig(data.config);
        setTradingSignals([]);
      }
    } catch (e) {
      console.error("Failed to factory reset paper trading:", e);
    }
  }, []);

  const updatePaperConfig = useCallback(
    async (config: Partial<TradingConfig>) => {
      try {
        const res = await fetch(`${API_URL}/api/paper-trading/config`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config),
        });
        if (res.ok) {
          const data = await res.json();
          setPaperConfig(data.config);
        }
      } catch (e) {
        console.error("Failed to update paper config:", e);
      }
    },
    []
  );

  // Set trading mode (3-way: off/paper/live, requires API key for paper/live)
  const setTradingMode = useCallback(
    async (mode: TradingModeValue, apiKey: string): Promise<{ success: boolean; error?: string }> => {
      try {
        const res = await fetch(`${API_URL}/api/mode`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": apiKey,
          },
          body: JSON.stringify({ mode }),
        });

        const data = await res.json();

        if (res.ok) {
          // Update local state
          setCurrentMode(mode);
          if (mode !== "off") {
            setLiveStatus((prev) => prev ? { ...prev, mode } : null);
          }
          return { success: true };
        } else {
          return { success: false, error: data.error || "Failed to set mode" };
        }
      } catch (e) {
        console.error("Failed to set trading mode:", e);
        return { success: false, error: "Network error" };
      }
    },
    []
  );

  // Set token allowances for Polymarket (one-time setup)
  const setAllowances = useCallback(
    async (apiKey: string): Promise<{ success: boolean; error?: string }> => {
      try {
        const res = await fetch(`${API_URL}/api/live-trading/allowances`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": apiKey,
          },
        });

        const data = await res.json();

        if (res.ok) {
          return { success: true };
        } else {
          return { success: false, error: data.error || "Failed to set allowances" };
        }
      } catch (e) {
        console.error("Failed to set allowances:", e);
        return { success: false, error: "Network error" };
      }
    },
    []
  );

  // Place a manual test order (for testing live trading)
  const placeTestOrder = useCallback(
    async (
      symbol: string,
      side: "UP" | "DOWN",
      amount: number,
      apiKey: string
    ): Promise<{ success: boolean; error?: string; order?: unknown }> => {
      try {
        const res = await fetch(`${API_URL}/api/live-trading/test-order`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": apiKey,
          },
          body: JSON.stringify({
            symbol,
            side,
            amount_usd: amount,
          }),
        });

        const data = await res.json();

        if (res.ok) {
          return { success: true, order: data.order };
        } else {
          return { success: false, error: data.error || "Failed to place order" };
        }
      } catch (e) {
        console.error("Failed to place test order:", e);
        return { success: false, error: "Network error" };
      }
    },
    []
  );

  // Initial connection
  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      ws.current?.close();
    };
  }, [connect]);

  // Fetch initial candles via REST API for each symbol
  useEffect(() => {
    const fetchInitialCandles = async () => {
      for (const symbol of symbols) {
        try {
          const res = await fetch(`${API_URL}/api/candles/${symbol}`);
          if (res.ok) {
            const data = await res.json();
            setCandles((prev) => ({
              ...prev,
              [symbol]: data.candles,
            }));
          }
        } catch (e) {
          console.error(`Failed to fetch candles for ${symbol}:`, e);
        }
      }
    };

    if (symbols.length > 0) {
      fetchInitialCandles();
    }
  }, [symbols]);

  // Fetch initial whale trades
  useEffect(() => {
    const fetchWhaleTrades = async () => {
      try {
        const res = await fetch(`${API_URL}/api/whale-trades`);
        if (res.ok) {
          const data = await res.json();
          setWhaleTrades(data.trades);
        }
      } catch (e) {
        console.error("Failed to fetch whale trades:", e);
      }
    };

    fetchWhaleTrades();
  }, []);

  // Fetch paper trading config
  useEffect(() => {
    const fetchPaperConfig = async () => {
      try {
        const res = await fetch(`${API_URL}/api/paper-trading/config`);
        if (res.ok) {
          const data = await res.json();
          setPaperConfig(data);
        }
      } catch (e) {
        console.error("Failed to fetch paper config:", e);
      }
    };

    fetchPaperConfig();
  }, []);

  return {
    isConnected,
    candles,
    trades,
    orderbooks,
    momentum,
    whaleTrades,
    whales,
    symbols,
    markets15m,
    marketTrades,
    chartData,
    paperAccount,
    paperSignals,
    paperConfig,
    liveStatus,
    currentMode,
    sniperStatus,
    sniperSignals,
    requestCandles,
    togglePaperTrading,
    resetTradingAccount,
    factoryReset,
    updatePaperConfig,
    setTradingMode,
    placeTestOrder,
    setAllowances,
  };
}
