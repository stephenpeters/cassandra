"use client";

import { memo } from "react";
import { Badge } from "@/components/ui/badge";
import { InfoTooltip } from "@/components/ui/info-tooltip";
import type { MomentumSignal, OrderBookData } from "@/types";

interface MarketAnalysisProps {
  symbol: string;
  momentum: MomentumSignal | undefined;
  orderbook: OrderBookData | undefined;
}

function MarketAnalysisComponent({ symbol, momentum, orderbook }: MarketAnalysisProps) {
  if (!momentum && !orderbook) {
    return (
      <div className="text-zinc-500 text-sm p-4 text-center">
        Loading market data...
      </div>
    );
  }

  const directionColor = momentum ? {
    UP: "bg-green-500",
    DOWN: "bg-red-500",
    NEUTRAL: "bg-zinc-500",
  }[momentum.direction] : "bg-zinc-500";

  const directionText = momentum ? {
    UP: "BULLISH",
    DOWN: "BEARISH",
    NEUTRAL: "NEUTRAL",
  }[momentum.direction] : "NEUTRAL";

  // Use momentum imbalance if available, else orderbook imbalance
  const imbalance = momentum?.orderbook_imbalance ?? orderbook?.imbalance ?? 0;

  return (
    <div className="space-y-3">
      {/* Header: Symbol + Direction */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-medium text-zinc-700 dark:text-zinc-200">{symbol}</span>
          <InfoTooltip content="Current momentum direction based on volume, price action, and order flow over the last 15 minutes." />
        </div>
        <Badge className={`${directionColor} text-white`}>
          {directionText}
        </Badge>
      </div>

      {/* Confidence bar */}
      {momentum && (
        <div>
          <div className="flex justify-between text-xs text-zinc-500 mb-1">
            <span className="flex items-center gap-1">
              Confidence
              <InfoTooltip content="How strongly the indicators align. Higher confidence = stronger signal. Above 60% is significant." />
            </span>
            <span className="font-mono">{(momentum.confidence * 100).toFixed(0)}%</span>
          </div>
          <div className="h-2 bg-zinc-200 dark:bg-zinc-800 rounded overflow-hidden">
            <div
              className={`h-full transition-all ${directionColor}`}
              style={{ width: `${momentum.confidence * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        {/* Volume Delta */}
        <div className="bg-zinc-100 dark:bg-zinc-800/50 rounded p-2">
          <div className="flex items-center gap-1 text-zinc-500 mb-1">
            Vol Δ
            <InfoTooltip content="Volume Delta: Buy volume minus sell volume. Positive = more buying pressure. Example: +$50K means buyers outpacing sellers by $50K." />
          </div>
          <div
            className={`font-mono font-medium ${
              (momentum?.volume_delta ?? 0) > 0 ? "text-green-500" : "text-red-500"
            }`}
          >
            {momentum ? (
              <>
                {momentum.volume_delta > 0 ? "+" : ""}
                ${(momentum.volume_delta / 1000).toFixed(0)}K
              </>
            ) : (
              "--"
            )}
          </div>
        </div>

        {/* Price Change */}
        <div className="bg-zinc-100 dark:bg-zinc-800/50 rounded p-2">
          <div className="flex items-center gap-1 text-zinc-500 mb-1">
            Price
            <InfoTooltip content="Price change in the last 15 minutes as a percentage. Positive = price is rising." />
          </div>
          <div
            className={`font-mono font-medium ${
              (momentum?.price_change_pct ?? 0) > 0 ? "text-green-500" : "text-red-500"
            }`}
          >
            {momentum ? (
              <>
                {momentum.price_change_pct > 0 ? "+" : ""}
                {momentum.price_change_pct.toFixed(3)}%
              </>
            ) : (
              "--"
            )}
          </div>
        </div>

        {/* Order Book Imbalance */}
        <div className="bg-zinc-100 dark:bg-zinc-800/50 rounded p-2">
          <div className="flex items-center gap-1 text-zinc-500 mb-1">
            Book
            <InfoTooltip content="Order Book Imbalance: Ratio of buy orders vs sell orders near current price. +40% means 70% bids vs 30% asks - more buyers waiting." />
          </div>
          <div
            className={`font-mono font-medium ${
              imbalance > 0 ? "text-green-500" : "text-red-500"
            }`}
          >
            {imbalance > 0 ? "+" : ""}
            {(imbalance * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Combined Imbalance Bar */}
      <div>
        <div className="flex justify-between text-xs text-zinc-500 mb-1">
          <span className="flex items-center gap-1">
            Market Pressure
            <InfoTooltip content="Visual representation of order book imbalance. Bar fills toward buying (green/right) or selling (red/left) pressure." />
          </span>
          <span className={`font-mono ${imbalance > 0 ? "text-green-500" : "text-red-500"}`}>
            {imbalance > 0 ? "Buyers" : "Sellers"}
          </span>
        </div>
        <div className="h-3 bg-zinc-200 dark:bg-zinc-700 rounded overflow-hidden relative">
          {/* Center line */}
          <div className="absolute left-1/2 top-0 bottom-0 w-px bg-zinc-400 dark:bg-zinc-500 z-10" />
          {/* Imbalance fill */}
          <div
            className={`absolute top-0 h-full transition-all ${
              imbalance >= 0 ? "bg-green-500" : "bg-red-500"
            }`}
            style={{
              left: imbalance >= 0 ? "50%" : `${50 + imbalance * 50}%`,
              width: `${Math.abs(imbalance) * 50}%`,
            }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-zinc-400 mt-0.5">
          <span>Sell</span>
          <span>Buy</span>
        </div>
      </div>

      {/* Order Book Depth (condensed) */}
      {orderbook && (
        <div className="border-t border-zinc-200 dark:border-zinc-700 pt-2">
          <div className="flex items-center gap-1 text-xs text-zinc-500 mb-2">
            Order Book Depth
            <InfoTooltip content="Top 3 price levels showing bid (buy) and ask (sell) orders. Larger bars = more volume at that price." />
          </div>
          <div className="grid grid-cols-2 gap-2 text-[10px]">
            {/* Bids */}
            <div className="space-y-0.5">
              <div className="text-green-500 font-medium">Bids (Buy)</div>
              {orderbook.bids.slice(0, 3).map(([price, size], i) => (
                <div key={`bid-${i}`} className="relative">
                  <div
                    className="absolute inset-y-0 right-0 bg-green-500/20 rounded-sm"
                    style={{ width: `${(size / Math.max(...orderbook.bids.slice(0, 3).map(b => b[1]))) * 100}%` }}
                  />
                  <div className="relative flex justify-between px-1 py-0.5">
                    <span className="text-green-400 font-mono">${price.toFixed(2)}</span>
                    <span className="text-zinc-400">{size.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
            {/* Asks */}
            <div className="space-y-0.5">
              <div className="text-red-500 font-medium">Asks (Sell)</div>
              {orderbook.asks.slice(0, 3).map(([price, size], i) => (
                <div key={`ask-${i}`} className="relative">
                  <div
                    className="absolute inset-y-0 left-0 bg-red-500/20 rounded-sm"
                    style={{ width: `${(size / Math.max(...orderbook.asks.slice(0, 3).map(a => a[1]))) * 100}%` }}
                  />
                  <div className="relative flex justify-between px-1 py-0.5">
                    <span className="text-red-400 font-mono">${price.toFixed(2)}</span>
                    <span className="text-zinc-400">{size.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          {/* Spread */}
          <div className="text-center text-[10px] text-zinc-500 mt-1">
            Spread: ${orderbook.spread.toFixed(3)} | Mid: ${orderbook.mid.toFixed(2)}
          </div>
        </div>
      )}
    </div>
  );
}

export const MarketAnalysis = memo(MarketAnalysisComponent);
