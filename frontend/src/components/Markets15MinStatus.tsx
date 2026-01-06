"use client";

import { memo, useMemo, useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MarketWindowChart } from "@/components/charts/MarketWindowChart";
import type { Markets15MinData, MomentumSignal, MarketTrade, MarketWindowChartData } from "@/types";

interface Markets15MinStatusProps {
  markets15m: Markets15MinData | null;
  marketTrades: MarketTrade[];
  momentum: Record<string, MomentumSignal>;
  chartData: Record<string, MarketWindowChartData>;
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

// Format countdown with large display
function formatCountdownLarge(seconds: number): { mins: string; secs: string; label: string } {
  if (seconds <= 0) return { mins: "00", secs: "00", label: "TRADING NOW" };
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return {
    mins: mins.toString().padStart(2, "0"),
    secs: secs.toString().padStart(2, "0"),
    label: seconds <= 60 ? "OPENING SOON" : "NEXT MARKET IN",
  };
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
  chartData,
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

  // Calculate time remaining in current window
  const timeRemainingInWindow = useMemo(() => {
    if (!timing?.is_open || !currentMarket) return 0;
    const now_sec = Math.floor(Date.now() / 1000);
    return Math.max(0, currentMarket.end_time - now_sec);
  }, [timing, currentMarket, now]);

  const countdownDisplay = useMemo(() => formatCountdownLarge(countdown), [countdown, now]);
  const windowCountdown = useMemo(() => formatCountdownLarge(timeRemainingInWindow), [timeRemainingInWindow, now]);

  return (
    <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
            15-Minute Markets
          </CardTitle>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">{timezone}</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Prominent Countdown Display */}
        <div className="grid grid-cols-2 gap-4">
          {/* Next Market Countdown */}
          <div className={`p-4 rounded-lg text-center ${
            countdown <= 0
              ? "bg-green-500/10 border border-green-500/30"
              : countdown <= 60
                ? "bg-yellow-500/10 border border-yellow-500/30 animate-pulse"
                : "bg-zinc-100 dark:bg-zinc-800/50 border border-zinc-300 dark:border-zinc-700"
          }`}>
            <div className="text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1">
              {countdownDisplay.label}
            </div>
            {countdown > 0 ? (
              <div className="flex items-center justify-center gap-1">
                <span className={`text-3xl font-mono font-bold ${
                  countdown <= 60 ? "text-yellow-500" : "text-zinc-700 dark:text-zinc-200"
                }`}>
                  {countdownDisplay.mins}
                </span>
                <span className="text-xl text-zinc-400">:</span>
                <span className={`text-3xl font-mono font-bold ${
                  countdown <= 60 ? "text-yellow-500" : "text-zinc-700 dark:text-zinc-200"
                }`}>
                  {countdownDisplay.secs}
                </span>
              </div>
            ) : (
              <div className="text-2xl font-bold text-green-500">LIVE</div>
            )}
            {timing && (
              <div className="text-xs text-zinc-500 mt-1">
                Opens at {new Date(timing.start * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false })}
              </div>
            )}
          </div>

          {/* Window Closes In */}
          <div className={`p-4 rounded-lg text-center ${
            timing?.is_open
              ? timeRemainingInWindow <= 120
                ? "bg-red-500/10 border border-red-500/30 animate-pulse"
                : "bg-blue-500/10 border border-blue-500/30"
              : "bg-zinc-100 dark:bg-zinc-800/50 border border-zinc-300 dark:border-zinc-700"
          }`}>
            <div className="text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1">
              {timing?.is_open ? "WINDOW CLOSES IN" : "WAITING FOR MARKET"}
            </div>
            {timing?.is_open && timeRemainingInWindow > 0 ? (
              <div className="flex items-center justify-center gap-1">
                <span className={`text-3xl font-mono font-bold ${
                  timeRemainingInWindow <= 120 ? "text-red-500" : "text-blue-500"
                }`}>
                  {windowCountdown.mins}
                </span>
                <span className="text-xl text-zinc-400">:</span>
                <span className={`text-3xl font-mono font-bold ${
                  timeRemainingInWindow <= 120 ? "text-red-500" : "text-blue-500"
                }`}>
                  {windowCountdown.secs}
                </span>
              </div>
            ) : (
              <div className="text-2xl font-bold text-zinc-400">--:--</div>
            )}
            {currentMarket && timing?.is_open && (
              <div className="text-xs text-zinc-500 mt-1">
                Closes at {new Date(currentMarket.end_time * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false })}
              </div>
            )}
          </div>
        </div>

        {/* Market Window Chart - shows BTC price vs UP/DOWN odds */}
        <div className="border border-zinc-300 dark:border-zinc-700 rounded-lg p-2 bg-zinc-50 dark:bg-zinc-800/30">
          {timing?.is_open && chartData[selectedSymbol] && chartData[selectedSymbol].data.length > 0 ? (
            <MarketWindowChart
              symbol={selectedSymbol}
              data={chartData[selectedSymbol].data}
              startPrice={chartData[selectedSymbol].start_price}
              priceRange={500}
              height={120}
            />
          ) : (
            <div className="h-[120px] flex flex-col items-center justify-center text-sm text-zinc-500">
              <div className="font-medium">Price vs Odds Chart</div>
              <div className="text-xs text-zinc-400 mt-1">
                {timing?.is_open
                  ? "Waiting for data..."
                  : "Chart appears when market is open"}
              </div>
            </div>
          )}
        </div>

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
                {/* Polymarket odds */}
                {market && (
                  <div className="text-xs text-zinc-500 flex items-center gap-1">
                    <span className="text-purple-400 font-medium">PM:</span>
                    <span>{(market.price * 100).toFixed(0)}% Up</span>
                  </div>
                )}
                {/* Binance price change */}
                {momo && (
                  <div className="text-xs flex items-center gap-1">
                    <span className="text-amber-400 font-medium">B:</span>
                    <span
                      className={`font-mono ${
                        momo.price_change_pct > 0 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {momo.price_change_pct > 0 ? "+" : ""}
                      {momo.price_change_pct.toFixed(3)}%
                    </span>
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
        <div className="text-xs text-zinc-600 text-center pt-2 border-t border-zinc-300 dark:border-zinc-800 space-y-1">
          <div className="flex items-center justify-center gap-4">
            <span className="flex items-center gap-1">
              <span className="text-purple-400 font-medium">PM</span>
              <span>= Polymarket odds</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="text-amber-400 font-medium">B</span>
              <span>= Binance price</span>
            </span>
          </div>
          <div>All times in {timezone} | Markets open at :00, :15, :30, :45</div>
        </div>
      </CardContent>
    </Card>
  );
}

export const Markets15MinStatus = memo(Markets15MinStatusComponent);
