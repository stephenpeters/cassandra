"use client";

import { memo, useMemo, useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Markets15MinData, MomentumSignal, MarketTrade } from "@/types";

interface Markets15MinStatusProps {
  markets15m: Markets15MinData | null;
  marketTrades: MarketTrade[];
  momentum: Record<string, MomentumSignal>;
  selectedSymbol: string;
  onSymbolSelect: (symbol: string) => void;
}

// Get user's timezone
function getUserTimezone(): string {
  return Intl.DateTimeFormat().resolvedOptions().timeZone;
}

// Format time with timezone
function formatTimeWithTZ(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZoneName: "short",
  });
}

// Format time range
function formatTimeRange(start: number, end: number): string {
  const startDate = new Date(start * 1000);
  const endDate = new Date(end * 1000);
  const format = (d: Date) =>
    d.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  const tz = startDate.toLocaleTimeString("en-US", { timeZoneName: "short" }).split(" ").pop();
  return `${format(startDate)} - ${format(endDate)} ${tz}`;
}

// Format countdown
function formatCountdown(seconds: number): string {
  if (seconds <= 0) return "OPEN";
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs}s`;
}

// Simulated resolved markets (last 4 windows)
interface ResolvedMarket {
  symbol: string;
  start: number;
  end: number;
  result: "UP" | "DOWN";
  open_price: number;
  close_price: number;
  high: number;
  low: number;
  volume: number;
}

function Markets15MinStatusComponent({
  markets15m,
  marketTrades,
  momentum,
  selectedSymbol,
  onSymbolSelect,
}: Markets15MinStatusProps) {
  const [now, setNow] = useState(Date.now());
  const timezone = useMemo(() => getUserTimezone(), []);

  // Update time every second
  useEffect(() => {
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, []);

  const timing = markets15m?.timing;
  const activeMarkets = markets15m?.active || {};

  // Calculate countdown
  const countdown = useMemo(() => {
    if (!timing) return 0;
    return Math.max(0, timing.time_until_start);
  }, [timing, now]);

  // Get current market for selected symbol
  const currentMarket = activeMarkets[selectedSymbol];

  // Calculate market progress (time elapsed in current window)
  const marketProgress = useMemo(() => {
    if (!currentMarket) return 0;
    const now_sec = Math.floor(Date.now() / 1000);
    const elapsed = now_sec - currentMarket.start_time;
    const duration = currentMarket.end_time - currentMarket.start_time;
    return Math.min(100, Math.max(0, (elapsed / duration) * 100));
  }, [currentMarket, now]);

  // Get recent trades for selected symbol
  const symbolTrades = useMemo(() => {
    return marketTrades.filter((t) => t.symbol === selectedSymbol).slice(0, 10);
  }, [marketTrades, selectedSymbol]);

  // Get momentum for selected symbol
  const symbolMomentum = momentum[`${selectedSymbol}USDT`];

  const symbols = ["BTC", "ETH", "SOL", "XRP", "DOGE"];

  return (
    <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
            15-Minute Markets
          </CardTitle>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">{timezone}</span>
            <Badge
              className={`text-xs ${
                countdown <= 60
                  ? "bg-yellow-500/20 text-yellow-400"
                  : countdown === 0
                  ? "bg-green-500/20 text-green-400"
                  : "bg-blue-500/20 text-blue-400"
              }`}
            >
              {countdown > 0 ? `Next: ${formatCountdown(countdown)}` : "TRADING"}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Symbol selector row */}
        <div className="grid grid-cols-5 gap-2">
          {symbols.map((sym) => {
            const market = activeMarkets[sym];
            const momo = momentum[`${sym}USDT`];
            const isSelected = selectedSymbol === sym;

            return (
              <button
                key={sym}
                onClick={() => onSymbolSelect(sym)}
                className={`p-3 rounded-lg transition-all ${
                  isSelected
                    ? "ring-2 ring-blue-500 bg-zinc-200 dark:bg-zinc-800"
                    : "bg-zinc-100 dark:bg-zinc-800/50 hover:bg-zinc-200 dark:hover:bg-zinc-800"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-sm">{sym}</span>
                  {momo && (
                    <Badge
                      className={`text-[10px] ${
                        momo.direction === "UP"
                          ? "bg-green-500/20 text-green-400"
                          : momo.direction === "DOWN"
                          ? "bg-red-500/20 text-red-400"
                          : "bg-zinc-600/20 text-zinc-400"
                      }`}
                    >
                      {momo.direction}
                    </Badge>
                  )}
                </div>
                {market && (
                  <div className="text-xs text-zinc-500">
                    {(market.price * 100).toFixed(0)}% Up
                  </div>
                )}
                {momo && (
                  <div
                    className={`text-xs font-mono ${
                      momo.price_change_pct > 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {momo.price_change_pct > 0 ? "+" : ""}
                    {momo.price_change_pct.toFixed(3)}%
                  </div>
                )}
              </button>
            );
          })}
        </div>

        {/* Current market details */}
        {currentMarket && (
          <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Current Window: {selectedSymbol}
              </span>
              <span className="text-xs text-zinc-400">
                {formatTimeRange(currentMarket.start_time, currentMarket.end_time)}
              </span>
            </div>

            {/* Progress bar */}
            <div className="h-2 bg-zinc-300 dark:bg-zinc-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-1000"
                style={{ width: `${marketProgress}%` }}
              />
            </div>

            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-500">
                {Math.floor(marketProgress)}% elapsed
              </span>
              <div className="flex items-center gap-4">
                <span className="text-zinc-400">
                  Up: <span className="text-green-400">{(currentMarket.price * 100).toFixed(1)}%</span>
                </span>
                <span className="text-zinc-400">
                  Down: <span className="text-red-400">{((1 - currentMarket.price) * 100).toFixed(1)}%</span>
                </span>
                {symbolMomentum && (
                  <span className="text-zinc-400">
                    Conf: <span className="text-blue-400">{(symbolMomentum.confidence * 100).toFixed(0)}%</span>
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Recent market trades */}
        {symbolTrades.length > 0 && (
          <div>
            <div className="text-xs text-zinc-500 mb-2">Recent {selectedSymbol} Market Trades</div>
            <div className="space-y-1 max-h-24 overflow-y-auto">
              {symbolTrades.map((trade, i) => (
                <div
                  key={`${trade.timestamp}-${i}`}
                  className="flex items-center justify-between text-xs p-2 bg-zinc-100 dark:bg-zinc-800/30 rounded"
                >
                  <div className="flex items-center gap-2">
                    <Badge
                      className={`text-[10px] ${
                        trade.side === "BUY"
                          ? "bg-green-500/20 text-green-400"
                          : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {trade.side}
                    </Badge>
                    <Badge
                      className={`text-[10px] ${
                        trade.outcome === "Up"
                          ? "bg-green-500/10 text-green-300"
                          : "bg-red-500/10 text-red-300"
                      }`}
                    >
                      {trade.outcome}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-zinc-400">${trade.usd_value.toFixed(0)}</span>
                    <span className="text-zinc-500 font-mono">
                      {formatTimeWithTZ(trade.timestamp).split(" ")[0]}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Timezone footer */}
        <div className="text-xs text-zinc-600 text-center pt-2 border-t border-zinc-300 dark:border-zinc-800">
          All times shown in {timezone} | Markets open at :00, :15, :30, :45
        </div>
      </CardContent>
    </Card>
  );
}

export const Markets15MinStatus = memo(Markets15MinStatusComponent);
