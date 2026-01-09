"use client";

import { memo, useMemo, useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { useTheme } from "@/components/ThemeProvider";
import { Clock } from "lucide-react";
// Note: ChevronUp, ChevronDown, ChevronRight replaced with CSS triangles
import type { TradingSignal, MomentumSignal } from "@/types";

// Filled triangle components
function TriangleUp({ className }: { className?: string }) {
  return (
    <div
      className={className}
      style={{
        width: 0,
        height: 0,
        borderLeft: "6px solid transparent",
        borderRight: "6px solid transparent",
        borderBottom: "10px solid currentColor",
      }}
    />
  );
}

function TriangleDown({ className }: { className?: string }) {
  return (
    <div
      className={className}
      style={{
        width: 0,
        height: 0,
        borderLeft: "6px solid transparent",
        borderRight: "6px solid transparent",
        borderTop: "10px solid currentColor",
      }}
    />
  );
}

function TriangleRight({ className }: { className?: string }) {
  return (
    <div
      className={className}
      style={{
        width: 0,
        height: 0,
        borderTop: "5px solid transparent",
        borderBottom: "5px solid transparent",
        borderLeft: "8px solid currentColor",
      }}
    />
  );
}

export interface ChartDataPoint {
  time: number;
  binancePrice: number;
  upPrice: number;
  downPrice: number;
}

// Checkpoint definitions
const CHECKPOINTS = [
  { label: "3m", seconds: 180 },
  { label: "6m", seconds: 360 },
  { label: "7:30", seconds: 450 },
  { label: "9m", seconds: 540 },
  { label: "12m", seconds: 720 },
];

interface SimpleStreamingChartProps {
  symbol: string;
  data: ChartDataPoint[];
  startPrice?: number;
  height?: number;
  marketStart?: number;
  marketEnd?: number;
  showPriceToBeat?: boolean;
  showCheckpoints?: boolean;
  signals?: TradingSignal[];
  momentum?: MomentumSignal;
}

function SimpleStreamingChartComponent({
  symbol,
  data,
  startPrice,
  height = 120,
  marketStart,
  marketEnd,
  showPriceToBeat = true,
  showCheckpoints = false,
  signals = [],
  momentum,
}: SimpleStreamingChartProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  // Time ticker to force checkpoint status updates
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!showCheckpoints) return;
    const interval = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, [showCheckpoints]);

  // Calculate time labels relative to market start
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    return data.map((d) => ({
      ...d,
      elapsed: marketStart ? d.time - marketStart : 0,
      timeLabel: marketStart
        ? `${Math.floor((d.time - marketStart) / 60)}m`
        : "",
    }));
  }, [data, marketStart]);

  // Get current values for display
  const currentValues = useMemo(() => {
    if (!data || data.length === 0) return null;
    const last = data[data.length - 1];
    const first = data[0];
    const priceDelta = last.binancePrice - first.binancePrice;
    const priceDeltaPct = (priceDelta / first.binancePrice) * 100;
    return {
      binance: last.binancePrice,
      up: last.upPrice,
      down: last.downPrice,
      delta: priceDelta,
      deltaPct: priceDeltaPct,
    };
  }, [data]);

  // Calculate progress through the window
  const windowProgress = useMemo(() => {
    if (!marketStart || !marketEnd) return null;
    const now = Math.floor(Date.now() / 1000);
    const elapsed = now - marketStart;
    const total = marketEnd - marketStart;
    return {
      elapsed,
      total,
      pct: Math.min(100, (elapsed / total) * 100),
    };
  }, [marketStart, marketEnd, tick]);

  // Calculate checkpoint statuses for this symbol
  const checkpointStatuses = useMemo(() => {
    if (!marketStart) return [];
    const now = Math.floor(Date.now() / 1000);
    const elapsed = now - marketStart;
    const marketEndTime = marketEnd || marketStart + 900;

    // Filter signals for this symbol and current market window
    const currentMarketSignals = signals.filter(
      (s) => s.symbol === symbol && s.timestamp >= marketStart && s.timestamp <= marketEndTime
    );

    return CHECKPOINTS.map((cp) => {
      // Find signal triggered at this checkpoint
      const signal = currentMarketSignals.find((s) => {
        const sigCheckpoint = s.checkpoint?.replace(":", "").replace("m", "").replace("s", "");
        const cpLabel = cp.label.replace(":", "").replace("m", "");
        return sigCheckpoint === cpLabel ||
               s.checkpoint === cp.label ||
               (cp.seconds === 450 && (s.checkpoint === "7m30s" || s.checkpoint === "7:30m" || s.checkpoint === "7:30"));
      });

      const isPast = elapsed >= cp.seconds;
      const isActive = elapsed >= cp.seconds - 30 && elapsed < cp.seconds + 30;

      return {
        ...cp,
        signal,
        isPast,
        isActive,
        status: signal ? "triggered" : isPast ? "missed" : "pending",
      };
    });
  }, [marketStart, marketEnd, signals, symbol, tick]);

  if (!data || data.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-xs text-zinc-400"
        style={{ height }}
      >
        Loading...
      </div>
    );
  }

  // Calculate over/under relative to price to beat
  const priceComparison = useMemo(() => {
    if (!startPrice || !currentValues) return null;
    const diff = currentValues.binance - startPrice;
    const diffPct = (diff / startPrice) * 100;
    return {
      diff,
      diffPct,
      isOver: diff > 0,
    };
  }, [startPrice, currentValues]);

  return (
    <div className="relative">
      {/* Price Comparison Panel - Polymarket Style */}
      {showPriceToBeat && startPrice && currentValues && priceComparison && (
        <div className="flex items-center justify-between mb-2 p-2 bg-zinc-100 dark:bg-zinc-800/70 rounded-lg">
          {/* Price to Beat */}
          <div className="flex flex-col">
            <span className="text-[10px] text-zinc-500 uppercase tracking-wide">Price to Beat</span>
            <span className="text-lg font-bold font-mono text-violet-600 dark:text-violet-400">
              ${startPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>

          {/* Current Price */}
          <div className="flex flex-col items-center">
            <span className="text-[10px] text-zinc-500 uppercase tracking-wide">{symbol} Current</span>
            <span className="text-lg font-bold font-mono text-zinc-800 dark:text-zinc-100">
              ${currentValues.binance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>

          {/* Over/Under */}
          <div className="flex flex-col items-end">
            <span className="text-[10px] text-zinc-500 uppercase tracking-wide">
              {priceComparison.isOver ? "Over" : "Under"}
            </span>
            <span className={`text-lg font-bold font-mono ${priceComparison.isOver ? "text-green-500" : "text-red-500"}`}>
              {priceComparison.isOver ? "+" : ""}${Math.abs(priceComparison.diff).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center justify-between text-[10px] mb-1 px-1">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2 h-0.5 bg-amber-500 rounded"></span>
            <span className="text-zinc-500">{symbol}</span>
            {currentValues && (
              <span className={`font-mono ${currentValues.delta >= 0 ? "text-green-500" : "text-red-500"}`}>
                ({currentValues.delta >= 0 ? "+" : ""}{currentValues.deltaPct.toFixed(2)}%)
              </span>
            )}
          </span>
          {/* Price to Beat line indicator */}
          {showPriceToBeat && startPrice && (
            <span className="flex items-center gap-1">
              <span className="w-2 h-0.5 bg-violet-500 rounded border-dashed"></span>
              <span className="text-zinc-500">Target Line</span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1">
            <span className="w-2 h-0.5 bg-green-500 rounded"></span>
            <span className="text-zinc-400">UP</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-0.5 bg-red-500 rounded"></span>
            <span className="text-zinc-400">DN</span>
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="relative" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            {/* Vertical grid lines every 30 seconds */}
            {[30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 510, 540, 570, 600, 630, 660, 690, 720, 750, 780, 810, 840, 870].map((sec) => (
              <ReferenceLine
                key={`grid-${sec}`}
                x={sec}
                stroke={isDark ? "#27272a" : "#e4e4e7"}
                strokeWidth={1}
              />
            ))}
            <XAxis
              dataKey="elapsed"
              type="number"
              domain={[0, 900]}
              allowDataOverflow={true}
              ticks={showCheckpoints ? [] : [0, 180, 360, 540, 720, 900]}
              tick={showCheckpoints ? false : { fontSize: 9, fill: isDark ? "#71717a" : "#a1a1aa" }}
              tickFormatter={(v) => {
                const mins = Math.floor(v / 60);
                return `${mins}m`;
              }}
              stroke={isDark ? "#3f3f46" : "#e4e4e7"}
              tickLine={false}
              axisLine={false}
              height={showCheckpoints ? 5 : undefined}
            />
            <YAxis
              yAxisId="left"
              orientation="left"
              domain={["auto", "auto"]}
              tick={{ fontSize: 9, fill: isDark ? "#71717a" : "#a1a1aa" }}
              tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`}
              stroke={isDark ? "#3f3f46" : "#e4e4e7"}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[0, 1]}
              tick={{ fontSize: 9, fill: isDark ? "#71717a" : "#a1a1aa" }}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              stroke={isDark ? "#3f3f46" : "#e4e4e7"}
              tickLine={false}
              axisLine={false}
              width={35}
            />

            {/* Price to beat reference line */}
            {showPriceToBeat && startPrice && (
              <ReferenceLine
                yAxisId="left"
                y={startPrice}
                stroke="#8b5cf6"
                strokeDasharray="4 4"
                strokeWidth={1}
              />
            )}

            {/* Binance price line */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="binancePrice"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />

            {/* UP price line */}
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="upPrice"
              stroke="#22c55e"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />

            {/* DOWN price line */}
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="downPrice"
              stroke="#ef4444"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>

      </div>

      {/* Checkpoint indicators below chart - positioned to align with x-axis */}
      {showCheckpoints && checkpointStatuses.length > 0 && (
        <div className="relative mt-1" style={{ marginLeft: 45, marginRight: 40 }}>
          {checkpointStatuses.map((cp) => {
            // Position based on checkpoint time as percentage of 900s window
            const leftPct = (cp.seconds / 900) * 100;
            return (
              <div
                key={cp.label}
                className="absolute flex flex-col items-center"
                style={{ left: `${leftPct}%`, transform: "translateX(-50%)" }}
              >
                {/* Checkpoint indicator */}
                <div className="mb-0.5">
                  {cp.status === "triggered" ? (
                    cp.signal?.signal.includes("UP") ? (
                      <div className="w-5 h-5 rounded-full bg-green-500/20 flex items-center justify-center">
                        <TriangleUp className="text-green-500" />
                      </div>
                    ) : cp.signal?.signal.includes("DOWN") ? (
                      <div className="w-5 h-5 rounded-full bg-red-500/20 flex items-center justify-center">
                        <TriangleDown className="text-red-500" />
                      </div>
                    ) : (
                      <div className="w-5 h-5 rounded-full bg-zinc-500/20 flex items-center justify-center">
                        <TriangleRight className="text-zinc-400" />
                      </div>
                    )
                  ) : cp.isPast ? (
                    <div className="w-5 h-5 rounded-full bg-zinc-500/10 flex items-center justify-center">
                      <TriangleRight className="text-zinc-400" />
                    </div>
                  ) : (
                    <Clock className={`w-5 h-5 ${cp.isActive ? "text-blue-500 animate-pulse" : "text-zinc-300 dark:text-zinc-600"}`} />
                  )}
                </div>
                {/* Label */}
                <div className={`text-[8px] font-mono ${
                  cp.status === "triggered"
                    ? cp.signal?.signal.includes("UP")
                      ? "text-green-500 font-semibold"
                      : cp.signal?.signal.includes("DOWN")
                      ? "text-red-500 font-semibold"
                      : "text-zinc-500"
                    : cp.isActive
                    ? "text-blue-500 font-semibold"
                    : "text-zinc-400"
                }`}>
                  {cp.label}
                </div>
              </div>
            );
          })}
          {/* Spacer to maintain height */}
          <div className="h-8" />
        </div>
      )}
    </div>
  );
}

export const SimpleStreamingChart = memo(SimpleStreamingChartComponent);
