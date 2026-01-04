"use client";

import { memo } from "react";
import type { OrderBookData } from "@/types";

interface OrderBookChartProps {
  symbol: string;
  data: OrderBookData | undefined;
}

function OrderBookChartComponent({ symbol, data }: OrderBookChartProps) {
  if (!data) {
    return (
      <div className="h-40 flex items-center justify-center text-zinc-500">
        Loading order book...
      </div>
    );
  }

  const maxSize = Math.max(
    ...data.bids.map((b) => b[1]),
    ...data.asks.map((a) => a[1])
  );

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex justify-between text-xs text-zinc-400 px-1">
        <span>Price</span>
        <span>Size</span>
      </div>

      {/* Asks (sell orders) - reversed so lowest ask is at bottom */}
      <div className="space-y-0.5">
        {data.asks
          .slice(0, 5)
          .reverse()
          .map(([price, size], i) => (
            <div key={`ask-${i}`} className="relative flex justify-between text-xs">
              <div
                className="absolute right-0 top-0 h-full bg-red-500/20"
                style={{ width: `${(size / maxSize) * 100}%` }}
              />
              <span className="z-10 text-red-400 font-mono">
                ${price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </span>
              <span className="z-10 text-zinc-400 font-mono">
                {size.toFixed(4)}
              </span>
            </div>
          ))}
      </div>

      {/* Spread indicator */}
      <div className="flex justify-center items-center py-1 border-y border-zinc-300 dark:border-zinc-700">
        <span className="text-xs text-zinc-500">
          Spread: ${data.spread.toFixed(2)} | Mid: $
          {data.mid.toLocaleString(undefined, { minimumFractionDigits: 2 })}
        </span>
      </div>

      {/* Bids (buy orders) */}
      <div className="space-y-0.5">
        {data.bids.slice(0, 5).map(([price, size], i) => (
          <div key={`bid-${i}`} className="relative flex justify-between text-xs">
            <div
              className="absolute right-0 top-0 h-full bg-green-500/20"
              style={{ width: `${(size / maxSize) * 100}%` }}
            />
            <span className="z-10 text-green-400 font-mono">
              ${price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </span>
            <span className="z-10 text-zinc-400 font-mono">
              {size.toFixed(4)}
            </span>
          </div>
        ))}
      </div>

      {/* Imbalance indicator */}
      <div className="mt-2">
        <div className="flex justify-between text-xs text-zinc-500 mb-1">
          <span>Imbalance</span>
          <span
            className={
              data.imbalance > 0
                ? "text-green-400"
                : data.imbalance < 0
                ? "text-red-400"
                : ""
            }
          >
            {(data.imbalance * 100).toFixed(1)}%
          </span>
        </div>
        <div className="h-2 bg-zinc-200 dark:bg-zinc-800 rounded overflow-hidden">
          <div
            className={`h-full transition-all ${
              data.imbalance > 0 ? "bg-green-500" : "bg-red-500"
            }`}
            style={{
              width: `${Math.abs(data.imbalance) * 50 + 50}%`,
              marginLeft: data.imbalance < 0 ? `${50 - Math.abs(data.imbalance) * 50}%` : "50%",
            }}
          />
        </div>
      </div>
    </div>
  );
}

export const OrderBookChart = memo(OrderBookChartComponent);
