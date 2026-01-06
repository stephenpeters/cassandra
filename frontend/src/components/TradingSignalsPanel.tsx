"use client";

import { memo, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ChevronUp,
  ChevronDown,
  ChevronRight,
  MinusCircle,
  TrendingUp,
  TrendingDown,
  Clock,
} from "lucide-react";
import type { PaperSignal, MomentumSignal } from "@/types";

// Checkpoint definitions
const CHECKPOINTS = [
  { label: "3m", seconds: 180 },
  { label: "6m", seconds: 360 },
  { label: "7:30m", seconds: 450 },
  { label: "9m", seconds: 540 },
  { label: "12m", seconds: 720 },
];

interface TradingSignalsPanelProps {
  signals: PaperSignal[];
  momentum: Record<string, MomentumSignal>;
  selectedSymbol: string;
  marketStart?: number;  // Current market window start timestamp
  marketEnd?: number;    // Current market window end timestamp
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function getSignalIcon(signal: string) {
  if (signal.includes("UP")) {
    return <ChevronUp className="h-5 w-5 text-green-500" />;
  }
  if (signal.includes("DOWN")) {
    return <ChevronDown className="h-5 w-5 text-red-500" />;
  }
  return <MinusCircle className="h-5 w-5 text-zinc-500" />;
}

function getSignalColor(signal: string) {
  if (signal.includes("UP")) return "text-green-500";
  if (signal.includes("DOWN")) return "text-red-500";
  return "text-zinc-500";
}

function getSignalBgColor(signal: string) {
  if (signal.includes("UP")) return "bg-green-500/10 border-green-500/30";
  if (signal.includes("DOWN")) return "bg-red-500/10 border-red-500/30";
  return "bg-zinc-500/10 border-zinc-500/30";
}

// Format market window time for display
function formatMarketWindow(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZoneName: "short",
  });
}

// Generate market slug from timestamp
function getMarketSlug(symbol: string, timestamp: number): string {
  return `${symbol.toLowerCase()}-updown-15m-${timestamp}`;
}

function TradingSignalsPanelComponent({
  signals,
  momentum,
  selectedSymbol,
  marketStart,
  marketEnd,
}: TradingSignalsPanelProps) {
  // Get latest signal for selected symbol AND current market window only
  const latestSignal = signals.find((s) =>
    s.symbol === selectedSymbol &&
    (!marketStart || (s.timestamp >= marketStart && s.timestamp <= (marketEnd || marketStart + 900)))
  );

  // Get momentum for selected symbol
  const symbolMomentum = momentum[`${selectedSymbol}USDT`];

  // Get signals for current market only (filter by market window)
  const currentMarketSignals = marketStart
    ? signals.filter((s) =>
        s.symbol === selectedSymbol &&
        s.timestamp >= marketStart &&
        s.timestamp <= (marketEnd || marketStart + 900)
      )
    : signals.filter((s) => s.symbol === selectedSymbol).slice(0, 10);

  // Market window info for title
  const marketSlug = marketStart ? getMarketSlug(selectedSymbol, marketStart) : null;
  const marketWindowStr = marketStart ? formatMarketWindow(marketStart) : null;

  // Calculate checkpoint statuses
  const checkpointStatuses = useMemo(() => {
    const now = Math.floor(Date.now() / 1000);
    const elapsed = marketStart ? now - marketStart : 0;

    return CHECKPOINTS.map((cp) => {
      // Find signal triggered at this checkpoint
      const signal = currentMarketSignals.find((s) => {
        // Match checkpoint label (handle variations like "7m30s" vs "7:30m")
        const sigCheckpoint = s.checkpoint?.replace(":", "").replace("m", "").replace("s", "");
        const cpLabel = cp.label.replace(":", "").replace("m", "");
        return sigCheckpoint === cpLabel ||
               s.checkpoint === cp.label ||
               (cp.seconds === 450 && (s.checkpoint === "7m30s" || s.checkpoint === "7:30m"));
      });

      const isPast = elapsed >= cp.seconds;
      const isActive = elapsed >= cp.seconds - 30 && elapsed < cp.seconds + 30; // 30s window around checkpoint

      return {
        ...cp,
        signal,
        isPast,
        isActive,
        status: signal ? "triggered" : isPast ? "missed" : "pending",
      };
    });
  }, [marketStart, currentMarketSignals]);

  return (
    <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex flex-col gap-1">
          <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
            Trading Signals
          </CardTitle>
          {/* Market window identifier */}
          {marketSlug && marketWindowStr && (
            <div className="text-xs text-zinc-500 font-mono">
              {marketSlug} @ {marketWindowStr}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Checkpoint Timeline */}
        <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
          <div className="text-xs text-zinc-500 mb-3">Checkpoint Status</div>
          <div className="flex items-center justify-between gap-1">
            {checkpointStatuses.map((cp) => (
              <div key={cp.label} className="flex flex-col items-center flex-1">
                {/* Checkpoint indicator - triangular chevrons in circles */}
                <div className="mb-1">
                  {cp.status === "triggered" ? (
                    // Show UP/DOWN/HOLD chevron based on signal
                    cp.signal?.signal.includes("UP") ? (
                      <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                        <ChevronUp className="w-6 h-6 text-green-500" strokeWidth={3} />
                      </div>
                    ) : cp.signal?.signal.includes("DOWN") ? (
                      <div className="w-8 h-8 rounded-full bg-red-500/20 flex items-center justify-center">
                        <ChevronDown className="w-6 h-6 text-red-500" strokeWidth={3} />
                      </div>
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-zinc-500/20 flex items-center justify-center">
                        <ChevronRight className="w-6 h-6 text-zinc-400" strokeWidth={3} />
                      </div>
                    )
                  ) : cp.isPast ? (
                    // Checkpoint passed but no signal (missed) - grey right chevron
                    <div className="w-8 h-8 rounded-full bg-zinc-500/10 flex items-center justify-center">
                      <ChevronRight className="w-6 h-6 text-zinc-400" strokeWidth={3} />
                    </div>
                  ) : (
                    // Waiting for checkpoint - clock icon
                    <Clock className={`w-8 h-8 ${cp.isActive ? "text-blue-500 animate-pulse" : "text-zinc-300 dark:text-zinc-600"}`} />
                  )}
                </div>
                {/* Label */}
                <div className={`text-[10px] font-mono ${
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
                {/* Edge badge if triggered with signal */}
                {cp.signal && (
                  <div className={`mt-0.5 text-[10px] font-mono font-semibold ${
                    cp.signal.signal.includes("UP")
                      ? "text-green-500"
                      : cp.signal.signal.includes("DOWN")
                      ? "text-red-500"
                      : "text-zinc-500"
                  }`}>
                    {(cp.signal.edge * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
        {/* Current Signal for Selected Symbol */}
        {latestSignal ? (
          <div
            className={`p-4 rounded-lg border ${getSignalBgColor(
              latestSignal.signal
            )}`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                {getSignalIcon(latestSignal.signal)}
                <div>
                  <div className="text-lg font-bold text-zinc-800 dark:text-zinc-200">
                    {latestSignal.symbol}
                  </div>
                  <div className="text-xs text-zinc-500">
                    @ {latestSignal.checkpoint} checkpoint
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div
                  className={`text-xl font-bold ${getSignalColor(
                    latestSignal.signal
                  )}`}
                >
                  {latestSignal.signal.replace("_", " ")}
                </div>
                <div className="text-xs text-zinc-500">
                  {formatTime(latestSignal.timestamp)}
                </div>
              </div>
            </div>

            {/* Signal Details */}
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                <div className="text-xs text-zinc-500">Fair Value</div>
                <div className="font-mono font-semibold text-zinc-700 dark:text-zinc-300">
                  {(latestSignal.fair_value * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                <div className="text-xs text-zinc-500">Market Price</div>
                <div className="font-mono font-semibold text-zinc-700 dark:text-zinc-300">
                  {(latestSignal.market_price * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                <div className="text-xs text-zinc-500">Edge</div>
                <div
                  className={`font-mono font-semibold ${
                    latestSignal.edge > 0
                      ? "text-green-600 dark:text-green-400"
                      : "text-red-600 dark:text-red-400"
                  }`}
                >
                  {latestSignal.edge > 0 ? "+" : ""}
                  {(latestSignal.edge * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Confidence Bar */}
            <div className="mt-3">
              <div className="flex justify-between text-xs text-zinc-500 mb-1">
                <span>Signal Confidence</span>
                <span>{(latestSignal.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${
                    latestSignal.signal.includes("UP")
                      ? "bg-green-500"
                      : latestSignal.signal.includes("DOWN")
                      ? "bg-red-500"
                      : "bg-zinc-500"
                  }`}
                  style={{ width: `${latestSignal.confidence * 100}%` }}
                />
              </div>
            </div>

            {/* Momentum Data */}
            {latestSignal.momentum && (
              <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-zinc-700">
                <div className="text-xs text-zinc-500 mb-2">Momentum Factors</div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="flex items-center gap-1">
                    {latestSignal.momentum.direction === "UP" ? (
                      <TrendingUp className="h-3 w-3 text-green-500" />
                    ) : latestSignal.momentum.direction === "DOWN" ? (
                      <TrendingDown className="h-3 w-3 text-red-500" />
                    ) : (
                      <MinusCircle className="h-3 w-3 text-zinc-500" />
                    )}
                    <span className="text-zinc-600 dark:text-zinc-400">
                      {latestSignal.momentum.direction}
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-500">Vol: </span>
                    <span
                      className={
                        latestSignal.momentum.volume_delta > 0
                          ? "text-green-500"
                          : "text-red-500"
                      }
                    >
                      {latestSignal.momentum.volume_delta > 0 ? "+" : ""}
                      ${(latestSignal.momentum.volume_delta / 1000).toFixed(0)}K
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-500">Price: </span>
                    <span
                      className={
                        latestSignal.momentum.price_change_pct > 0
                          ? "text-green-500"
                          : "text-red-500"
                      }
                    >
                      {latestSignal.momentum.price_change_pct > 0 ? "+" : ""}
                      {latestSignal.momentum.price_change_pct.toFixed(3)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : null}

        {/* Signal History - Current Market Only */}
        {currentMarketSignals.length > 0 && (
          <div>
            <div className="text-xs text-zinc-500 mb-2">
              Signals This Market ({currentMarketSignals.length})
            </div>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {currentMarketSignals.map((sig, i) => (
                <div
                  key={`${sig.symbol}-${sig.timestamp}-${i}`}
                  className={`flex items-center justify-between p-2 rounded text-sm ${
                    sig.symbol === selectedSymbol
                      ? "bg-blue-500/10 dark:bg-blue-500/10"
                      : "bg-zinc-100 dark:bg-zinc-800/30"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {getSignalIcon(sig.signal)}
                    <span className="font-medium text-zinc-700 dark:text-zinc-300">
                      {sig.symbol}
                    </span>
                    <span className="text-xs text-zinc-500">
                      @ {sig.checkpoint}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Badge
                      className={`text-[10px] ${
                        sig.signal === "HOLD"
                          ? "bg-zinc-500/20 text-zinc-500"
                          : sig.signal.includes("UP")
                          ? "bg-green-500/20 text-green-500"
                          : "bg-red-500/20 text-red-500"
                      }`}
                    >
                      {sig.signal.replace("_", " ")}
                    </Badge>
                    <span className="text-xs text-zinc-500 font-mono">
                      {(sig.edge * 100).toFixed(1)}%
                    </span>
                    <span className="text-xs text-zinc-400">
                      {formatTime(sig.timestamp)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </CardContent>
    </Card>
  );
}

export const TradingSignalsPanel = memo(TradingSignalsPanelComponent);
