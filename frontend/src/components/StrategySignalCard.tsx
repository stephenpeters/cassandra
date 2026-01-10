"use client";

import { memo, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Target, Clock, TrendingUp, AlertCircle } from "lucide-react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import type { SniperStatus, SniperSignal } from "@/types";

interface StrategySignalCardProps {
  sniperStatus: Record<string, SniperStatus>;
  sniperSignals: SniperSignal[];
  selectedSymbol?: string;
}

function formatTime(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
}

function formatPrice(price: number): string {
  return `${(price * 100).toFixed(1)}c`;
}

function formatElapsed(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m${secs.toString().padStart(2, "0")}s`;
}

export const StrategySignalCard = memo(function StrategySignalCard({
  sniperStatus,
  sniperSignals,
  selectedSymbol,
}: StrategySignalCardProps) {
  // Get status for selected symbol or first available
  const currentStatus = useMemo(() => {
    if (selectedSymbol && sniperStatus[selectedSymbol]) {
      return sniperStatus[selectedSymbol];
    }
    // Return first available status
    const symbols = Object.keys(sniperStatus);
    return symbols.length > 0 ? sniperStatus[symbols[0]] : null;
  }, [sniperStatus, selectedSymbol]);

  // Filter signals for selected symbol if provided
  const relevantSignals = useMemo(() => {
    if (!selectedSymbol) return sniperSignals;
    return sniperSignals.filter((s) => s.symbol === selectedSymbol);
  }, [sniperSignals, selectedSymbol]);

  // Status badge styling
  const getStatusBadge = (status: SniperStatus | null) => {
    if (!status) {
      return (
        <Badge className="bg-zinc-500/20 text-zinc-500">
          No Data
        </Badge>
      );
    }

    switch (status.status) {
      case "disabled":
        return (
          <Badge className="bg-zinc-500/20 text-zinc-500">
            Disabled
          </Badge>
        );
      case "skip":
        return (
          <Badge className="bg-zinc-500/20 text-zinc-500">
            Skip
          </Badge>
        );
      case "waiting":
        return (
          <Badge className="bg-blue-500/20 text-blue-400">
            Waiting
          </Badge>
        );
      case "position_taken":
        return (
          <Badge className="bg-purple-500/20 text-purple-400">
            Position Taken
          </Badge>
        );
      case "ready":
        return (
          <Badge className="bg-green-500/20 text-green-400 animate-pulse">
            Ready
          </Badge>
        );
      case "no_signal":
        return (
          <Badge className="bg-orange-500/20 text-orange-400">
            No Signal
          </Badge>
        );
      default:
        return (
          <Badge className="bg-zinc-500/20 text-zinc-500">
            Unknown
          </Badge>
        );
    }
  };

  return (
    <Card className="border-zinc-200 dark:border-zinc-800">
      <CardHeader className="py-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="h-4 w-4 text-purple-500" />
            Sniper Strategy
          </CardTitle>
          {getStatusBadge(currentStatus)}
        </div>
      </CardHeader>
      <CardContent className="py-2 space-y-3">
        {/* Current Status */}
        {currentStatus && (
          <div className="p-2 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-500">
                {currentStatus.symbol}
              </span>
              <span className="text-zinc-400 font-mono">
                {currentStatus.reason}
              </span>
            </div>

            {/* Waiting progress */}
            {currentStatus.status === "waiting" && currentStatus.time_remaining !== undefined && (
              <div className="mt-2">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-zinc-500 flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    Time until active
                  </span>
                  <span className="font-mono text-blue-400">
                    {formatElapsed(currentStatus.time_remaining)}
                  </span>
                </div>
                <div className="w-full bg-zinc-200 dark:bg-zinc-700 rounded-full h-1.5">
                  <div
                    className="bg-blue-500 h-1.5 rounded-full transition-all"
                    style={{
                      width: `${Math.min(100, ((currentStatus.elapsed_sec || 0) / (currentStatus.min_elapsed_sec || 600)) * 100)}%`,
                    }}
                  />
                </div>
              </div>
            )}

            {/* Ready to signal */}
            {currentStatus.status === "ready" && (
              <div className="mt-2 flex items-center justify-between">
                <span className="text-green-400 font-medium flex items-center gap-1">
                  <TrendingUp className="h-3 w-3" />
                  {currentStatus.signal} @ {formatPrice(currentStatus.entry_price || 0)}
                </span>
                <span className="text-xs text-zinc-400">
                  EV: {(currentStatus.ev_pct || 0).toFixed(1)}%
                </span>
              </div>
            )}

            {/* Evaluations for no_signal */}
            {currentStatus.status === "no_signal" && currentStatus.evaluations && (
              <div className="mt-2 space-y-1">
                {currentStatus.evaluations.map((eval_data) => (
                  <div
                    key={eval_data.side}
                    className="flex items-center justify-between text-xs"
                  >
                    <span className={eval_data.side === "UP" ? "text-green-500" : "text-red-500"}>
                      {eval_data.side}
                    </span>
                    <span className="text-zinc-400">
                      {formatPrice(eval_data.price)}
                    </span>
                    <span className={eval_data.in_range ? "text-green-400" : "text-zinc-500"}>
                      {eval_data.in_range ? "In range" : "Out of range"}
                    </span>
                    <span className={eval_data.ev_ok ? "text-green-400" : "text-orange-400"}>
                      EV: {eval_data.ev_pct.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* No status available */}
        {!currentStatus && (
          <div className="p-3 text-center text-zinc-500 text-sm flex items-center justify-center gap-2">
            <AlertCircle className="h-4 w-4" />
            No Signal Generated
          </div>
        )}

        {/* Recent Signals */}
        {relevantSignals.length > 0 && (
          <div>
            <div className="text-xs text-zinc-500 mb-2">Recent Signals</div>
            <div className="space-y-1 max-h-24 overflow-y-auto">
              {relevantSignals.slice(0, 3).map((signal, i) => (
                <div
                  key={`${signal.symbol}-${signal.market_start}-${i}`}
                  className="flex items-center justify-between p-2 bg-zinc-50 dark:bg-zinc-800/30 rounded text-xs"
                >
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{signal.symbol}</span>
                    <Badge
                      className={`text-[10px] ${
                        signal.signal === "UP"
                          ? "bg-green-500/20 text-green-500"
                          : "bg-red-500/20 text-red-500"
                      }`}
                    >
                      {signal.signal}
                    </Badge>
                    <span className="text-zinc-400">
                      @ {formatPrice(signal.entry_price)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-zinc-400">
                    <span>EV: {(signal.ev_pct || 0).toFixed(1)}%</span>
                    <span>${signal.position_size.toFixed(0)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
});
