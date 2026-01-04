"use client";

import { memo } from "react";
import { Badge } from "@/components/ui/badge";
import type { MomentumSignal } from "@/types";

interface MomentumIndicatorProps {
  symbol: string;
  signal: MomentumSignal | undefined;
}

function MomentumIndicatorComponent({ symbol, signal }: MomentumIndicatorProps) {
  if (!signal) {
    return (
      <div className="text-zinc-500 text-sm">Loading momentum...</div>
    );
  }

  const directionColor = {
    UP: "bg-green-500",
    DOWN: "bg-red-500",
    NEUTRAL: "bg-zinc-500",
  }[signal.direction];

  const directionText = {
    UP: "BULLISH",
    DOWN: "BEARISH",
    NEUTRAL: "NEUTRAL",
  }[signal.direction];

  return (
    <div className="space-y-3">
      {/* Direction badge */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-zinc-400">{symbol}</span>
        <Badge className={`${directionColor} text-white`}>
          {directionText}
        </Badge>
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex justify-between text-xs text-zinc-500 mb-1">
          <span>Confidence</span>
          <span>{(signal.confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-zinc-200 dark:bg-zinc-800 rounded overflow-hidden">
          <div
            className={`h-full transition-all ${directionColor}`}
            style={{ width: `${signal.confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-zinc-100 dark:bg-zinc-800/50 rounded p-2">
          <div className="text-zinc-500">Volume Delta</div>
          <div
            className={`font-mono ${
              signal.volume_delta > 0 ? "text-green-400" : "text-red-400"
            }`}
          >
            {signal.volume_delta > 0 ? "+" : ""}
            ${(signal.volume_delta / 1000).toFixed(0)}K
          </div>
        </div>
        <div className="bg-zinc-100 dark:bg-zinc-800/50 rounded p-2">
          <div className="text-zinc-500">Price Chg</div>
          <div
            className={`font-mono ${
              signal.price_change_pct > 0 ? "text-green-400" : "text-red-400"
            }`}
          >
            {signal.price_change_pct > 0 ? "+" : ""}
            {signal.price_change_pct.toFixed(3)}%
          </div>
        </div>
        <div className="bg-zinc-100 dark:bg-zinc-800/50 rounded p-2 col-span-2">
          <div className="text-zinc-500">Order Book Imbalance</div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1.5 bg-zinc-300 dark:bg-zinc-700 rounded overflow-hidden">
              <div
                className={`h-full ${
                  signal.orderbook_imbalance > 0 ? "bg-green-500" : "bg-red-500"
                }`}
                style={{
                  width: `${Math.abs(signal.orderbook_imbalance) * 50 + 50}%`,
                  marginLeft:
                    signal.orderbook_imbalance < 0
                      ? `${50 - Math.abs(signal.orderbook_imbalance) * 50}%`
                      : "50%",
                }}
              />
            </div>
            <span
              className={`font-mono ${
                signal.orderbook_imbalance > 0
                  ? "text-green-400"
                  : "text-red-400"
              }`}
            >
              {(signal.orderbook_imbalance * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export const MomentumIndicator = memo(MomentumIndicatorComponent);
