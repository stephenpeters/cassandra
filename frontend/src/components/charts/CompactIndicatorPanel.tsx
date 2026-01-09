"use client";

import { memo } from "react";
import { Badge } from "@/components/ui/badge";
import { SimpleTooltip } from "@/components/ui/tooltip";
import type { MomentumSignal, OrderBookData } from "@/types";

interface CompactIndicatorPanelProps {
  symbol?: string;
  momentum: MomentumSignal | undefined;
  orderbook?: OrderBookData | undefined;
}

/**
 * Compact single-row indicator panel.
 *
 * Layout:
 * [BULLISH] | Conf: 72% | Vol: +$5K | Book: +12% | RSI: 45 | ADX: 28 | VWAP | ST
 */
function CompactIndicatorPanelComponent({
  symbol,
  momentum,
  orderbook,
}: CompactIndicatorPanelProps) {
  if (!momentum) {
    return (
      <div className="flex items-center gap-3 text-sm text-zinc-500 p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded">
        Loading indicators...
      </div>
    );
  }

  const directionColor = {
    UP: "bg-green-500",
    DOWN: "bg-red-500",
    NEUTRAL: "bg-zinc-500",
  }[momentum.direction];

  const directionText = {
    UP: "BULLISH",
    DOWN: "BEARISH",
    NEUTRAL: "NEUTRAL",
  }[momentum.direction];

  const imbalance = momentum.orderbook_imbalance ?? orderbook?.imbalance ?? 0;
  const confidence = momentum.confidence * 100;
  const volumeDelta = momentum.volume_delta / 1000; // In thousands
  const rsi = momentum.rsi ?? 50;
  const adx = momentum.adx ?? 0;
  const vwapSignal = momentum.vwap_signal ?? "NEUTRAL";
  const supertrendDir = momentum.supertrend_direction ?? "NEUTRAL";

  // Helper to format values with color
  const colorClass = (val: number) =>
    val > 0 ? "text-green-500" : val < 0 ? "text-red-500" : "text-zinc-400";

  const signalArrow = (signal: string) =>
    signal === "UP" ? "text-green-500" : signal === "DOWN" ? "text-red-500" : "text-zinc-400";

  return (
    <div className="flex items-center gap-3 text-sm p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded flex-wrap">
      {/* Direction badge */}
      <SimpleTooltip content="Overall market direction based on combined indicators">
        <Badge className={`${directionColor} text-white text-xs px-2.5 py-0.5 font-semibold`}>
          {directionText}
        </Badge>
      </SimpleTooltip>

      <span className="text-zinc-400">|</span>

      {/* Confidence */}
      <SimpleTooltip content="Signal confidence - how strongly indicators align">
        <span className="font-mono">
          <span className="text-zinc-500">Conf:</span>{" "}
          <span className={confidence > 60 ? "text-green-500" : confidence < 40 ? "text-red-500" : "text-yellow-500"}>
            {confidence.toFixed(0)}%
          </span>
        </span>
      </SimpleTooltip>

      <span className="text-zinc-400">|</span>

      {/* Volume Delta */}
      <SimpleTooltip content="Volume delta - buy volume minus sell volume (in thousands)">
        <span className="font-mono">
          <span className="text-zinc-500">Vol:</span>{" "}
          <span className={colorClass(volumeDelta)}>
            {volumeDelta >= 0 ? "+" : ""}{volumeDelta.toFixed(0)}K
          </span>
        </span>
      </SimpleTooltip>

      <span className="text-zinc-400">|</span>

      {/* Order Book Imbalance */}
      <SimpleTooltip content="Order book imbalance - positive means more buyers">
        <span className="font-mono">
          <span className="text-zinc-500">Book:</span>{" "}
          <span className={colorClass(imbalance)}>
            {imbalance >= 0 ? "+" : ""}{(imbalance * 100).toFixed(0)}%
          </span>
        </span>
      </SimpleTooltip>

      <span className="text-zinc-400">|</span>

      {/* RSI */}
      <SimpleTooltip content="RSI (Relative Strength Index) - overbought >70, oversold <30">
        <span className="font-mono">
          <span className="text-zinc-400 dark:text-zinc-300">RSI:</span>{" "}
          <span className={rsi > 70 ? "text-red-500 font-semibold" : rsi < 30 ? "text-green-500 font-semibold" : "text-zinc-700 dark:text-zinc-100"}>
            {rsi.toFixed(0)}
          </span>
        </span>
      </SimpleTooltip>

      <span className="text-zinc-400">|</span>

      {/* ADX */}
      <SimpleTooltip content="ADX (Average Directional Index) - trend strength. >25 = strong trend">
        <span className="font-mono">
          <span className="text-zinc-400 dark:text-zinc-300">ADX:</span>{" "}
          <span className={adx > 25 ? "text-purple-500 font-semibold" : "text-zinc-700 dark:text-zinc-100"}>
            {adx.toFixed(0)}
          </span>
        </span>
      </SimpleTooltip>

      <span className="text-zinc-400">|</span>

      {/* VWAP Signal */}
      <SimpleTooltip content="VWAP signal - price position relative to volume-weighted average">
        <span className="font-mono flex items-center gap-0.5">
          <span className={signalArrow(vwapSignal)}>
            {vwapSignal === "UP" ? "▲" : vwapSignal === "DOWN" ? "▼" : "●"}
          </span>
          <span className="text-zinc-500">VWAP</span>
        </span>
      </SimpleTooltip>

      <span className="text-zinc-400">|</span>

      {/* SuperTrend */}
      <SimpleTooltip content="SuperTrend indicator - trend following signal">
        <span className="font-mono flex items-center gap-0.5">
          <span className={signalArrow(supertrendDir)}>
            {supertrendDir === "UP" ? "▲" : supertrendDir === "DOWN" ? "▼" : "●"}
          </span>
          <span className="text-zinc-500">ST</span>
        </span>
      </SimpleTooltip>
    </div>
  );
}

export const CompactIndicatorPanel = memo(CompactIndicatorPanelComponent);
