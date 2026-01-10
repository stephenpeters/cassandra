"use client";

import { useState, useMemo, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CountdownTimerBadgeCompact } from "@/components/ui/countdown-timer-badge";
import { FlipCountdown } from "@/components/ui/flip-countdown";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { SimpleStreamingChart } from "@/components/charts/SimpleStreamingChart";
import { MarketAnalysis } from "@/components/charts/MarketAnalysis";
import { IndicatorGauges } from "@/components/charts/IndicatorGauges";
import { CompactIndicatorPanel } from "@/components/charts/CompactIndicatorPanel";
import type {
  Markets15MinData,
  MomentumSignal,
  MarketTrade,
  MarketWindowChartData,
  OrderBookData,
  Position,
  TradingSignal,
} from "@/types";
import { TrendingUp, TrendingDown, ChevronDown, ChevronUp, ExternalLink } from "lucide-react";

interface PolymarketDashboardProps {
  markets15m: Markets15MinData | null;
  marketTrades: MarketTrade[];
  momentum: Record<string, MomentumSignal>;
  chartData: Record<string, MarketWindowChartData>;
  orderbooks: Record<string, OrderBookData>;
  positions: Position[];
  selectedSymbol: string;
  onSymbolSelect: (symbol: string) => void;
  signals?: TradingSignal[];  // Trading signals to display on chart
}

// Only BTC and ETH have sufficient volume (>$20K)
const ACTIVE_SYMBOLS = ["BTC", "ETH"];

function formatMarketSlugET(symbol: string, startTime: number): string {
  // Format: "BTC @ 7:15 AM ET"
  const date = new Date(startTime * 1000);
  const timeStr = date.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZone: "America/New_York",
  });
  return `${symbol} @ ${timeStr} ET`;
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
  const [compactIndicators, setCompactIndicators] = useState(true);

  // Update time every second
  useEffect(() => {
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, []);

  const timing = markets15m?.timing;
  const activeMarkets = markets15m?.active || {};

  // Get current market for selected symbol
  const currentMarket = activeMarkets[selectedSymbol];

  // Get position for selected symbol (only show positions for current market)
  const symbolPosition = positions.find(
    (p) => p.symbol === selectedSymbol && currentMarket && p.market_start === currentMarket.start_time
  );

  // Get momentum and orderbook for selected symbol
  const selectedMomentum = momentum[`${selectedSymbol}USDT`];
  const selectedOrderbook = orderbooks[`${selectedSymbol}USDT`];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">
          Markets
        </h2>
        {timing?.is_open && (
          <Badge className="bg-green-500/20 text-green-400 text-xs">LIVE</Badge>
        )}
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
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sm">{sym}</span>
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
              </div>

              {/* Odds row */}
              <div className="flex items-center justify-between text-xs mb-1">
                <SimpleTooltip content="Current market odds for price going UP or DOWN">
                  <span className="text-zinc-500 cursor-help">Odds:</span>
                </SimpleTooltip>
                {market ? (
                  <div className="flex items-center gap-2">
                    <SimpleTooltip content={`Buy UP token at $${(market.price).toFixed(2)} - pays $1 if price goes up`}>
                      <span className="text-green-500 font-mono cursor-help">
                        ↑${(market.price).toFixed(2)}
                      </span>
                    </SimpleTooltip>
                    <SimpleTooltip content={`Buy DOWN token at $${(1 - market.price).toFixed(2)} - pays $1 if price goes down`}>
                      <span className="text-red-500 font-mono cursor-help">
                        ↓${(1 - market.price).toFixed(2)}
                      </span>
                    </SimpleTooltip>
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
          {/* Market header with slug, countdown (centered), and prices */}
          <div className="flex items-center justify-between">
            {/* Left: Market title */}
            <span className="font-semibold text-lg">
              {currentMarket
                ? `${selectedSymbol.toLowerCase()}-updown-15m @ ${new Date(currentMarket.start_time * 1000).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true, timeZone: "America/New_York" })} ET`
                : selectedSymbol}
            </span>
            {/* Center: Flip countdown timer */}
            {timing?.is_open && currentMarket && (
              <SimpleTooltip content="Time remaining until market closes">
                <div className="cursor-help">
                  <FlipCountdown
                    seconds={Math.max(0, currentMarket.end_time - Math.floor(Date.now() / 1000))}
                  />
                </div>
              </SimpleTooltip>
            )}
            {currentMarket && (
              <div className="flex items-center gap-3 text-sm">
                <SimpleTooltip content={`UP token price: $${currentMarket.price.toFixed(3)} - implied ${(currentMarket.price * 100).toFixed(1)}% chance of price increase`}>
                  <span className="text-green-500 font-mono font-bold cursor-help">
                    UP: ${currentMarket.price.toFixed(2)}
                  </span>
                </SimpleTooltip>
                <SimpleTooltip content={`DOWN token price: $${(1 - currentMarket.price).toFixed(3)} - implied ${((1 - currentMarket.price) * 100).toFixed(1)}% chance of price decrease`}>
                  <span className="text-red-500 font-mono font-bold cursor-help">
                    DOWN: ${(1 - currentMarket.price).toFixed(2)}
                  </span>
                </SimpleTooltip>
                <SimpleTooltip content="Open this market on Polymarket">
                  <a
                    href={`https://polymarket.com/event/${selectedSymbol.toLowerCase()}-updown-15m-${currentMarket.start_time}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 transition-colors"
                  >
                    <ExternalLink className="w-3 h-3" />
                    PM
                  </a>
                </SimpleTooltip>
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
                showCheckpoints={false}  // Only needed for latency_gap strategy
                signals={signals}
                momentum={selectedMomentum}
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

          {/* Market Indicators - Compact or Full */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="text-xs font-medium text-zinc-500">Market Indicators</div>
              <SimpleTooltip content={compactIndicators ? "Show full indicator details" : "Show compact single-row view"}>
                <button
                  onClick={() => setCompactIndicators(!compactIndicators)}
                  className="text-[10px] text-blue-500 hover:text-blue-400 flex items-center gap-1"
                >
                  {compactIndicators ? (
                    <>
                      <ChevronDown className="w-3 h-3" /> Expand
                    </>
                  ) : (
                    <>
                      <ChevronUp className="w-3 h-3" /> Compact
                    </>
                  )}
                </button>
              </SimpleTooltip>
            </div>

            {compactIndicators ? (
              /* Compact single-row view */
              <CompactIndicatorPanel
                symbol={selectedSymbol}
                momentum={selectedMomentum}
                orderbook={selectedOrderbook}
              />
            ) : (
              /* Full expanded view */
              <div className="space-y-3">
                <MarketAnalysis
                  symbol={selectedSymbol}
                  momentum={selectedMomentum}
                  orderbook={selectedOrderbook}
                />
                <div>
                  <div className="text-xs font-medium text-zinc-500 mb-2">Indicator Gauges</div>
                  <div className="bg-zinc-100 dark:bg-zinc-800/50 rounded-lg p-3">
                    <IndicatorGauges momentum={selectedMomentum} />
                  </div>
                </div>
              </div>
            )}
          </div>

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

        </CardContent>
      </Card>
    </div>
  );
}
