"use client";

import { memo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ArrowUpCircle,
  ArrowDownCircle,
  MinusCircle,
  TrendingUp,
  TrendingDown,
  Clock,
} from "lucide-react";
import type { PaperSignal, MomentumSignal } from "@/types";

interface TradingSignalsPanelProps {
  signals: PaperSignal[];
  momentum: Record<string, MomentumSignal>;
  selectedSymbol: string;
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
    return <ArrowUpCircle className="h-5 w-5 text-green-500" />;
  }
  if (signal.includes("DOWN")) {
    return <ArrowDownCircle className="h-5 w-5 text-red-500" />;
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

function TradingSignalsPanelComponent({
  signals,
  momentum,
  selectedSymbol,
}: TradingSignalsPanelProps) {
  // Get latest signal for selected symbol
  const latestSignal = signals.find((s) => s.symbol === selectedSymbol);

  // Get momentum for selected symbol
  const symbolMomentum = momentum[`${selectedSymbol}USDT`];

  // Get all recent signals (last 10)
  const recentSignals = signals.slice(0, 10);

  return (
    <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
            Trading Signals
          </CardTitle>
          <Badge className="bg-blue-500/20 text-blue-400 text-xs">
            Checkpoints: 3m, 7m, 10m, 12.5m
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
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
        ) : (
          <div className="p-4 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg text-center">
            <Clock className="h-8 w-8 text-zinc-400 mx-auto mb-2" />
            <div className="text-zinc-500 text-sm">
              Waiting for next checkpoint signal...
            </div>
            <div className="text-xs text-zinc-400 mt-1">
              Signals are generated at 3m, 7m, 10m, and 12.5m into each market
            </div>
          </div>
        )}

        {/* Signal History */}
        {recentSignals.length > 0 && (
          <div>
            <div className="text-xs text-zinc-500 mb-2">Recent Signals</div>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {recentSignals.map((sig, i) => (
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

        {/* Current Momentum Overview */}
        {symbolMomentum && (
          <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-zinc-500">
                Live {selectedSymbol} Momentum
              </span>
              <Badge
                className={`text-xs ${
                  symbolMomentum.direction === "UP"
                    ? "bg-green-500/20 text-green-400"
                    : symbolMomentum.direction === "DOWN"
                    ? "bg-red-500/20 text-red-400"
                    : "bg-zinc-600/20 text-zinc-400"
                }`}
              >
                {symbolMomentum.direction}
              </Badge>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-zinc-500">Vol Delta: </span>
                <span
                  className={`font-mono ${
                    symbolMomentum.volume_delta > 0
                      ? "text-green-500"
                      : "text-red-500"
                  }`}
                >
                  ${(symbolMomentum.volume_delta / 1000).toFixed(0)}K
                </span>
              </div>
              <div>
                <span className="text-zinc-500">Price: </span>
                <span
                  className={`font-mono ${
                    symbolMomentum.price_change_pct > 0
                      ? "text-green-500"
                      : "text-red-500"
                  }`}
                >
                  {symbolMomentum.price_change_pct > 0 ? "+" : ""}
                  {symbolMomentum.price_change_pct.toFixed(3)}%
                </span>
              </div>
              <div>
                <span className="text-zinc-500">Book: </span>
                <span
                  className={`font-mono ${
                    symbolMomentum.orderbook_imbalance > 0
                      ? "text-green-500"
                      : "text-red-500"
                  }`}
                >
                  {(symbolMomentum.orderbook_imbalance * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export const TradingSignalsPanel = memo(TradingSignalsPanelComponent);
