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
  requestCandles: (symbol: string) => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout>();

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
            return {
              ...prev,
              [msg.symbol!]: [...current.slice(-499), candle],
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
            return {
              ...prev,
              [msg.symbol!]: [...current.slice(-99), trade],
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
          setWhaleTrades((prev) => [trade, ...prev.slice(0, 99)]);
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

  return {
    isConnected,
    candles,
    trades,
    orderbooks,
    momentum,
    whaleTrades,
    whales,
    symbols,
    requestCandles,
  };
}
