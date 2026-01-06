"use client";

import { useState, useMemo, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SimpleStreamingChart } from "@/components/charts/SimpleStreamingChart";
import { MarketAnalysis } from "@/components/charts/MarketAnalysis";
import type {
  Markets15MinData,
  MomentumSignal,
  MarketTrade,
  MarketWindowChartData,
  OrderBookData,
  PaperPosition,
  PaperSignal,
} from "@/types";
import { TrendingUp, TrendingDown, Clock, ChevronDown, ChevronUp, ExternalLink } from "lucide-react";

interface PolymarketDashboardProps {
  markets15m: Markets15MinData | null;
  marketTrades: MarketTrade[];
  momentum: Record<string, MomentumSignal>;
  chartData: Record<string, MarketWindowChartData>;
  orderbooks: Record<string, OrderBookData>;
  positions: PaperPosition[];
  selectedSymbol: string;
  onSymbolSelect: (symbol: string) => void;
  signals?: PaperSignal[];  // Trading signals to display on chart
}

// Only BTC, ETH, SOL have volume
const ACTIVE_SYMBOLS = ["BTC", "ETH", "SOL"];

function formatCountdown(seconds: number): string {
  if (seconds <= 0) return "LIVE";
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function formatElapsed(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m${secs.toString().padStart(2, "0")}s`;
}

function formatTimeWindow(startTime: number, endTime: number): string {
  const start = new Date(startTime * 1000);
  const end = new Date(endTime * 1000);
  const formatTime = (d: Date) => d.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  return `${formatTime(start)}-${formatTime(end)}`;
}

function formatMarketTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export function PolymarketDashboard({
  markets15m,
  marketTrades,
  momentum,
  chartData,
  orderbooks,
  positions,
  selectedSymbol,
  onSymbolSelect,
  signals = [],
}: PolymarketDashboardProps) {
  const [now, setNow] = useState(Date.now());
  const [showAllTrades, setShowAllTrades] = useState(false);

  // Update time every second
  useEffect(() => {
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, []);

  const timing = markets15m?.timing;
  const activeMarkets = markets15m?.active || {};

  // Calculate countdown to next market
  const countdown = useMemo(() => {
    if (!timing) return 0;
    return Math.max(0, timing.time_until_start);
  }, [timing]);

  // Get current market for selected symbol
  const currentMarket = activeMarkets[selectedSymbol];

  // Time remaining in current window
  const timeRemainingInWindow = useMemo(() => {
    if (!timing?.is_open || !currentMarket) return 0;
    const now_sec = Math.floor(Date.now() / 1000);
    return Math.max(0, currentMarket.end_time - now_sec);
  }, [timing, currentMarket, now]);

  // Time elapsed in current window
  const timeElapsedInWindow = useMemo(() => {
    if (!timing?.is_open || !currentMarket) return 0;
    const now_sec = Math.floor(Date.now() / 1000);
    return Math.max(0, now_sec - currentMarket.start_time);
  }, [timing, currentMarket, now]);

  // Get position for selected symbol
  const symbolPosition = positions.find(
    (p) => p.symbol === selectedSymbol && currentMarket && p.market_start === currentMarket.start_time
  );

  // Get trades for selected symbol (recent 10)
  const symbolTrades = useMemo(() => {
    return marketTrades.filter((t) => t.symbol === selectedSymbol).slice(0, 10);
  }, [marketTrades, selectedSymbol]);

  // Get momentum and orderbook for selected symbol
  const selectedMomentum = momentum[`${selectedSymbol}USDT`];
  const selectedOrderbook = orderbooks[`${selectedSymbol}USDT`];

  return (
    <div className="space-y-4">
      {/* Header with timer */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">
            Polymarket
          </h2>
          {timing?.is_open && currentMarket && (
            <span className="text-xs text-zinc-500 font-mono">
              {formatTimeWindow(currentMarket.start_time, currentMarket.end_time)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <Clock className="w-4 h-4 text-zinc-400" />
            {timing?.is_open ? (
              <div className="flex items-center gap-2">
                <span className="font-mono text-purple-500">
                  @ {formatElapsed(timeElapsedInWindow)}
                </span>
                <span className="text-zinc-400">|</span>
                <span className="font-mono text-blue-500">
                  {formatCountdown(timeRemainingInWindow)} left
                </span>
              </div>
            ) : (
              <span className="font-mono text-zinc-500">
                Next in {formatCountdown(countdown)}
              </span>
            )}
          </div>
          {timing?.is_open && (
            <Badge className="bg-green-500/20 text-green-400 text-xs">LIVE</Badge>
          )}
        </div>
      </div>

      {/* Summary cards - 3 columns */}
      <div className="grid grid-cols-3 gap-3">
        {ACTIVE_SYMBOLS.map((sym) => {
          const market = activeMarkets[sym];
          const momo = momentum[`${sym}USDT`];
          const isSelected = selectedSymbol === sym;
          const hasPosition = positions.some(
            (p) => p.symbol === sym && market && p.market_start === market.start_time
          );
          const symChartData = chartData[sym];

          return (
            <button
              key={sym}
              onClick={() => onSymbolSelect(sym)}
              className={`p-3 rounded-lg transition-all text-left ${
                isSelected
                  ? "ring-2 ring-purple-500 bg-purple-50 dark:bg-purple-900/20"
                  : "bg-zinc-100 dark:bg-zinc-800/50 hover:bg-zinc-200 dark:hover:bg-zinc-800"
              }`}
            >
              {/* Header row */}
              <div className="flex items-center justify-between mb-1">
                <div>
                  <span className="font-semibold text-sm">{sym}</span>
                  {market && (
                    <span className="text-[10px] text-zinc-400 ml-1 font-mono">
                      {formatMarketTime(market.start_time)}
                    </span>
                  )}
                </div>
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
                    {momo.direction === "UP" ? (
                      <TrendingUp className="w-3 h-3" />
                    ) : momo.direction === "DOWN" ? (
                      <TrendingDown className="w-3 h-3" />
                    ) : null}
                  </Badge>
                )}
              </div>

              {/* Mini chart or placeholder */}
              <div className="h-12 mb-2 bg-zinc-200 dark:bg-zinc-700/50 rounded overflow-hidden">
                {timing?.is_open && symChartData && symChartData.data.length > 0 ? (
                  <SimpleStreamingChart
                    symbol={sym}
                    data={symChartData.data}
                    startPrice={symChartData.start_price}
                    height={48}
                    marketStart={market?.start_time}
                    marketEnd={market?.end_time}
                    showPriceToBeat={false}
                  />
                ) : (
                  <div className="h-full flex items-center justify-center text-[10px] text-zinc-400">
                    {timing?.is_open ? "Loading..." : "—"}
                  </div>
                )}
              </div>

              {/* Odds row */}
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-zinc-500">Odds:</span>
                {market ? (
                  <div className="flex items-center gap-2">
                    <span className="text-green-500 font-mono">
                      ↑{(market.price * 100).toFixed(0)}%
                    </span>
                    <span className="text-red-500 font-mono">
                      ↓{((1 - market.price) * 100).toFixed(0)}%
                    </span>
                  </div>
                ) : (
                  <span className="text-zinc-400">—</span>
                )}
              </div>

              {/* Position indicator */}
              <div className="text-[10px]">
                {hasPosition ? (
                  <span className="text-purple-400 font-medium">
                    Open Position
                  </span>
                ) : (
                  <span className="text-zinc-400">No Position</span>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {/* Detail view for selected symbol */}
      <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
        <CardContent className="p-4 space-y-4">
          {/* Symbol header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-lg">{selectedSymbol}</span>
              {selectedMomentum && (
                <Badge
                  className={`${
                    selectedMomentum.direction === "UP"
                      ? "bg-green-500/20 text-green-400"
                      : selectedMomentum.direction === "DOWN"
                      ? "bg-red-500/20 text-red-400"
                      : "bg-zinc-600/20 text-zinc-400"
                  }`}
                >
                  {selectedMomentum.direction}
                </Badge>
              )}
            </div>
            {currentMarket && (
              <div className="flex items-center gap-3 text-sm">
                <span className="text-green-500 font-mono">
                  UP: {(currentMarket.price * 100).toFixed(1)}%
                </span>
                <span className="text-red-500 font-mono">
                  DOWN: {((1 - currentMarket.price) * 100).toFixed(1)}%
                </span>
                <a
                  href={`https://polymarket.com/event/${selectedSymbol.toLowerCase()}-updown-15m-${currentMarket.start_time}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 transition-colors"
                  title="Open on Polymarket"
                >
                  <ExternalLink className="w-3 h-3" />
                  PM
                </a>
              </div>
            )}
          </div>

          {/* Chart */}
          <div className="border border-zinc-300 dark:border-zinc-700 rounded-lg p-2 bg-zinc-50 dark:bg-zinc-800/30">
            {timing?.is_open && chartData[selectedSymbol] && chartData[selectedSymbol].data.length > 0 ? (
              <SimpleStreamingChart
                symbol={selectedSymbol}
                data={chartData[selectedSymbol].data}
                startPrice={chartData[selectedSymbol].start_price}
                height={150}
                marketStart={currentMarket?.start_time}
                marketEnd={currentMarket?.end_time}
                showPriceToBeat={true}
                showCheckpoints={true}
                signals={signals}
              />
            ) : (
              <div className="h-[150px] flex flex-col items-center justify-center text-sm text-zinc-500">
                <div className="font-medium">Price vs Odds Chart</div>
                <div className="text-xs text-zinc-400 mt-1">
                  {timing?.is_open ? "Collecting data..." : "Opens when market is live"}
                </div>
              </div>
            )}
          </div>

          {/* Market Analysis */}
          <MarketAnalysis
            symbol={selectedSymbol}
            momentum={selectedMomentum}
            orderbook={selectedOrderbook}
          />

          {/* Open Position */}
          <div>
            <div className="text-xs font-medium text-zinc-500 mb-2">Open Position</div>
            {symbolPosition ? (
              <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge className={symbolPosition.side === "UP" ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}>
                      {symbolPosition.side}
                    </Badge>
                    <span className="text-sm font-mono">
                      ${symbolPosition.cost_basis.toFixed(2)}
                    </span>
                  </div>
                  <span className="text-xs text-zinc-400">
                    Entry: {symbolPosition.entry_price.toFixed(3)}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-sm text-zinc-400 italic">No open positions</div>
            )}
          </div>

          {/* Recent Trades */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="text-xs font-medium text-zinc-500">Recent Trades</div>
              <button
                onClick={() => setShowAllTrades(!showAllTrades)}
                className="text-xs text-zinc-400 hover:text-zinc-300 flex items-center gap-1"
              >
                {showAllTrades ? "Show less" : "Show more"}
                {showAllTrades ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>
            </div>
            {symbolTrades.length > 0 ? (
              <div className={`space-y-1 ${showAllTrades ? "" : "max-h-24"} overflow-y-auto`}>
                {symbolTrades.slice(0, showAllTrades ? 10 : 4).map((trade, i) => (
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
                        {formatMarketTime(trade.timestamp)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-zinc-400 italic">No recent trades</div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Footer legend */}
      <div className="text-xs text-zinc-500 text-center">
        Markets open at :00, :15, :30, :45 each hour
      </div>
    </div>
  );
}
