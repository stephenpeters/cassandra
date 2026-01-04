"use client";

import { memo, useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import type { WhaleTrade, MomentumSignal } from "@/types";

interface PolymarketCardProps {
  symbol: string;
  whaleTrades: WhaleTrade[];
  momentum: MomentumSignal | undefined;
}

function getNextMarketTime(): { start: Date; end: Date; timeUntilStart: number } {
  const now = new Date();
  const minutes = now.getMinutes();

  // 15-min markets start at :00, :15, :30, :45
  const nextSlot = Math.ceil((minutes + 1) / 15) * 15;
  const start = new Date(now);
  start.setMinutes(nextSlot % 60);
  start.setSeconds(0);
  start.setMilliseconds(0);

  if (nextSlot >= 60) {
    start.setHours(start.getHours() + 1);
  }

  const end = new Date(start);
  end.setMinutes(end.getMinutes() + 15);

  const timeUntilStart = start.getTime() - now.getTime();

  return { start, end, timeUntilStart };
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function formatTimeAgo(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp * 1000;

  if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  return `${Math.floor(diff / 3600000)}h ago`;
}

function PolymarketCardComponent({ symbol, whaleTrades, momentum }: PolymarketCardProps) {
  const { start, end, timeUntilStart } = useMemo(() => getNextMarketTime(), []);

  // Filter trades for this symbol's markets
  const symbolTrades = useMemo(() => {
    const keywords = {
      BTC: ["bitcoin", "btc"],
      ETH: ["ethereum", "eth"],
      SOL: ["solana", "sol"],
      XRP: ["xrp", "ripple"],
      DOGE: ["doge", "dogecoin"],
    }[symbol] || [];

    return whaleTrades.filter((t) =>
      keywords.some((kw) => t.market.toLowerCase().includes(kw))
    );
  }, [symbol, whaleTrades]);

  // Get recent trades (last 15 minutes)
  const recentTrades = useMemo(() => {
    const cutoff = Date.now() / 1000 - 900; // 15 minutes
    return symbolTrades.filter((t) => t.timestamp > cutoff);
  }, [symbolTrades]);

  // Calculate sentiment from recent whale trades
  const sentiment = useMemo(() => {
    if (recentTrades.length === 0) return null;

    let bullish = 0;
    let bearish = 0;

    recentTrades.forEach((t) => {
      const isBullish = t.outcome.toLowerCase().includes("up") ||
                        t.outcome.toLowerCase().includes("yes");
      if (t.side === "BUY") {
        if (isBullish) bullish += t.usd_value;
        else bearish += t.usd_value;
      } else {
        if (isBullish) bearish += t.usd_value;
        else bullish += t.usd_value;
      }
    });

    const total = bullish + bearish;
    if (total === 0) return null;

    return {
      bullish,
      bearish,
      ratio: bullish / total,
      direction: bullish > bearish ? "BULLISH" : bearish > bullish ? "BEARISH" : "NEUTRAL",
    };
  }, [recentTrades]);

  const timeUntilMinutes = Math.floor(timeUntilStart / 60000);
  const timeUntilSeconds = Math.floor((timeUntilStart % 60000) / 1000);

  return (
    <div className="space-y-4">
      {/* Next market timing */}
      <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-zinc-500">Next 15-Min Market</span>
          <Badge className="bg-blue-500/20 text-blue-400 text-xs">
            {symbol} Up/Down
          </Badge>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-mono text-zinc-700 dark:text-zinc-300">
            {formatTime(start)} - {formatTime(end)}
          </span>
          <span className={`text-sm font-mono ${
            timeUntilStart < 60000 ? "text-yellow-500 dark:text-yellow-400" : "text-zinc-600 dark:text-zinc-400"
          }`}>
            {timeUntilStart > 0 ? `${timeUntilMinutes}m ${timeUntilSeconds}s` : "OPEN"}
          </span>
        </div>
      </div>

      {/* Momentum alignment */}
      {momentum && (
        <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
          <div className="text-xs text-zinc-500 mb-2">Binance Momentum</div>
          <div className="flex items-center justify-between">
            <Badge className={`text-xs ${
              momentum.direction === "UP"
                ? "bg-green-500/20 text-green-400"
                : momentum.direction === "DOWN"
                ? "bg-red-500/20 text-red-400"
                : "bg-zinc-600/20 text-zinc-400"
            }`}>
              {momentum.direction}
            </Badge>
            <span className="text-xs text-zinc-600 dark:text-zinc-400">
              {(momentum.confidence * 100).toFixed(0)}% confidence
            </span>
          </div>
        </div>
      )}

      {/* Whale sentiment */}
      <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
        <div className="text-xs text-zinc-500 mb-2">
          Whale Activity (15m)
        </div>
        {sentiment ? (
          <>
            <div className="flex items-center justify-between mb-2">
              <Badge className={`text-xs ${
                sentiment.direction === "BULLISH"
                  ? "bg-green-500/20 text-green-400"
                  : sentiment.direction === "BEARISH"
                  ? "bg-red-500/20 text-red-400"
                  : "bg-zinc-600/20 text-zinc-400"
              }`}>
                {sentiment.direction}
              </Badge>
              <span className="text-xs text-zinc-600 dark:text-zinc-400">
                {recentTrades.length} trades
              </span>
            </div>
            <div className="h-2 bg-zinc-300 dark:bg-zinc-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-green-500 to-green-400"
                style={{ width: `${sentiment.ratio * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs mt-1">
              <span className="text-green-400">${(sentiment.bullish / 1000).toFixed(1)}K</span>
              <span className="text-red-400">${(sentiment.bearish / 1000).toFixed(1)}K</span>
            </div>
          </>
        ) : (
          <div className="text-xs text-zinc-500">No recent whale trades</div>
        )}
      </div>

      {/* Recent whale trades for this symbol */}
      <div>
        <div className="text-xs text-zinc-500 mb-2">
          Recent {symbol} Trades
        </div>
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {symbolTrades.slice(0, 5).map((trade, i) => (
            <div
              key={`${trade.tx_hash}-${i}`}
              className="flex items-center justify-between text-xs p-2 bg-zinc-100 dark:bg-zinc-800/30 rounded"
            >
              <div className="flex items-center gap-2">
                <span className="text-cyan-600 dark:text-cyan-400 font-medium">{trade.whale}</span>
                <Badge className={`text-[10px] ${
                  trade.side === "BUY" ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                }`}>
                  {trade.side}
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-zinc-600 dark:text-zinc-400">
                  ${trade.usd_value.toFixed(0)}
                </span>
                <span className="text-zinc-500">
                  {formatTimeAgo(trade.timestamp)}
                </span>
              </div>
            </div>
          ))}
          {symbolTrades.length === 0 && (
            <div className="text-xs text-zinc-500 p-2">
              No {symbol} trades yet
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export const PolymarketCard = memo(PolymarketCardComponent);
