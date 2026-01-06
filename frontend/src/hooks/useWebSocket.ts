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
  PaperAccount,
  PaperSignal,
  PaperTradingConfig,
} from "@/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
  paperAccount: PaperAccount | null;
  paperSignals: PaperSignal[];
  paperConfig: PaperTradingConfig | null;
  requestCandles: (symbol: string) => void;
  togglePaperTrading: () => Promise<void>;
  resetPaperAccount: () => Promise<void>;
  updatePaperConfig: (config: Partial<PaperTradingConfig>) => Promise<void>;
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
  const [paperAccount, setPaperAccount] = useState<PaperAccount | null>(null);
  const [paperSignals, setPaperSignals] = useState<PaperSignal[]>([]);
  const [paperConfig, setPaperConfig] = useState<PaperTradingConfig | null>(null);

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
        if (msg.paper_trading) setPaperAccount(msg.paper_trading);
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
          setPaperAccount(msg.data as PaperAccount);
        }
        break;

      case "paper_signal":
        if (msg.data) {
          const signal = msg.data as PaperSignal;
          // Limit to 20 paper signals (reduced from 50 for memory)
          setPaperSignals((prev) => [signal, ...prev.slice(0, 19)]);
        }
        break;

      case "paper_trade":
        // Trade completed - account will be updated via paper_account message
        break;

      case "paper_position":
        // Position opened - account will be updated via paper_account message
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

  const resetPaperAccount = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/paper-trading/reset`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        setPaperAccount(data.account);
        setPaperSignals([]);
      }
    } catch (e) {
      console.error("Failed to reset paper account:", e);
    }
  }, []);

  const updatePaperConfig = useCallback(
    async (config: Partial<PaperTradingConfig>) => {
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
    requestCandles,
    togglePaperTrading,
    resetPaperAccount,
    updatePaperConfig,
  };
}
