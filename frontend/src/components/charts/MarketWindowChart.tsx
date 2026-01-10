"use client";

import { useEffect, useRef, memo, useMemo } from "react";
import {
  createChart,
  ColorType,
  LineSeries,
  LineStyle,
} from "lightweight-charts";
import type {
  IChartApi,
  ISeriesApi,
  LineData,
  Time,
} from "lightweight-charts";
import { useTheme } from "@/components/ThemeProvider";
import type { TradingSignal } from "@/types";

export interface MarketWindowDataPoint {
  time: number;  // Unix timestamp in seconds
  binancePrice: number;
  upPrice: number;
  downPrice: number;
}

interface MarketWindowChartProps {
  symbol: string;
  data: MarketWindowDataPoint[];
  startPrice?: number;  // Binance price at market open (for scaling) - THIS IS THE "PRICE TO BEAT"
  priceRange?: number;  // ±range from start price (default 500)
  height?: number;
  marketStart?: number;  // Market window start timestamp
  marketEnd?: number;    // Market window end timestamp
  signals?: TradingSignal[];  // Trading signals to display as markers
  showPriceToBeat?: boolean;  // Show the "price to beat" annotation
}

function MarketWindowChartComponent({
  symbol,
  data,
  startPrice,
  priceRange = 500,
  height = 120,
  marketStart,
  marketEnd,
  signals = [],
  showPriceToBeat = true,
}: MarketWindowChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const binanceSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const priceToBeatSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const { theme } = useTheme();

  // Calculate price range for Binance axis
  const priceRangeCalc = useMemo(() => {
    if (startPrice) {
      return {
        min: startPrice - priceRange,
        max: startPrice + priceRange,
      };
    }
    if (data.length === 0) return { min: 0, max: 100000 };

    const prices = data.map((d) => d.binancePrice);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const padding = (maxPrice - minPrice) * 0.1 || 50;
    return {
      min: minPrice - padding,
      max: maxPrice + padding,
    };
  }, [data, startPrice, priceRange]);

  // Theme-aware colors
  const colors = theme === "dark" ? {
    text: "#9ca3af",
    grid: "#27272a",
    crosshair: "#6b7280",
    border: "#3f3f46",
    background: "transparent",
  } : {
    text: "#52525b",
    grid: "#e4e4e7",
    crosshair: "#a1a1aa",
    border: "#d4d4d8",
    background: "transparent",
  };

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: colors.background },
        textColor: colors.text,
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: colors.grid, style: LineStyle.Dotted },
        horzLines: { color: colors.grid, style: LineStyle.Dotted },
      },
      width: containerRef.current.clientWidth,
      height: height,
      crosshair: {
        mode: 0,  // No crosshair for mini chart
      },
      handleScroll: false,  // Disable scroll - fixed window
      handleScale: false,   // Disable zoom - fixed window
      timeScale: {
        borderColor: colors.border,
        timeVisible: true,
        secondsVisible: false,
        fixLeftEdge: true,
        fixRightEdge: true,
        lockVisibleTimeRangeOnResize: true,
        rightBarStaysOnScroll: false,
        shiftVisibleRangeOnNewBar: false,  // Don't shift when new data arrives
        tickMarkFormatter: (time: number) => {
          // Show elapsed time from market start if we have it
          if (marketStart) {
            const elapsed = time - marketStart;
            const mins = Math.floor(elapsed / 60);
            const secs = elapsed % 60;
            if (mins === 0) return `${secs}s`;
            if (secs === 0) return `${mins}m`;
            return `${mins}m${secs}s`;
          }
          const date = new Date(time * 1000);
          return date.toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
          });
        },
      },
      leftPriceScale: {
        borderColor: colors.border,
        visible: true,
        autoScale: false,
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      rightPriceScale: {
        visible: false,
      },
    });

    // Binance price series (left axis) - bold line
    const binanceSeries = chart.addSeries(LineSeries, {
      color: "#f59e0b",  // Amber for BTC
      lineWidth: 2,
      priceScaleId: "left",
      priceFormat: {
        type: "price",
        precision: 0,
        minMove: 1,
      },
      title: symbol,
    });

    // "Price to Beat" horizontal line (dashed) - the opening price
    const priceToBeatSeries = chart.addSeries(LineSeries, {
      color: "#8b5cf6",  // Purple
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceScaleId: "left",
      priceFormat: {
        type: "price",
        precision: 0,
        minMove: 1,
      },
      title: "Target",
      crosshairMarkerVisible: false,
    });

    // Configure left price scale (Binance)
    chart.priceScale("left").applyOptions({
      autoScale: false,
      scaleMargins: { top: 0.05, bottom: 0.05 },
    });

    chartRef.current = chart;
    binanceSeriesRef.current = binanceSeries;
    priceToBeatSeriesRef.current = priceToBeatSeries;

    // Handle resize
    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: containerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [height, theme, colors, symbol, marketStart]);

  // Update price scales when range changes
  useEffect(() => {
    if (!chartRef.current) return;

    // Set Binance price range - use autoScale for smooth movement
    chartRef.current.priceScale("left").applyOptions({
      autoScale: true,
      scaleMargins: { top: 0.1, bottom: 0.1 },
    });
  }, [priceRangeCalc]);

  // Update data and set fixed time range
  useEffect(() => {
    if (!binanceSeriesRef.current || !chartRef.current) return;

    if (data.length === 0) {
      binanceSeriesRef.current.setData([]);
      if (priceToBeatSeriesRef.current) priceToBeatSeriesRef.current.setData([]);
      return;
    }

    const binanceData: LineData<Time>[] = data.map((d) => ({
      time: d.time as Time,
      value: d.binancePrice,
    }));

    binanceSeriesRef.current.setData(binanceData);

    // Set "Price to Beat" horizontal line at startPrice
    if (priceToBeatSeriesRef.current && showPriceToBeat && startPrice && marketStart && marketEnd) {
      const priceToBeatData: LineData<Time>[] = [
        { time: marketStart as Time, value: startPrice },
        { time: marketEnd as Time, value: startPrice },
      ];
      priceToBeatSeriesRef.current.setData(priceToBeatData);
    }

    // Set fixed time range for full 15-min window
    if (marketStart && marketEnd) {
      // Force the visible range to the full market window
      const timeScale = chartRef.current.timeScale();
      timeScale.setVisibleRange({
        from: marketStart as Time,
        to: marketEnd as Time,
      });
      // Apply options to prevent any auto-scrolling
      timeScale.applyOptions({
        rightOffset: 0,
        barSpacing: timeScale.width() / ((marketEnd - marketStart) / 5), // ~5 sec per bar
      });
    }
  }, [data, marketStart, marketEnd, startPrice, showPriceToBeat]);

  // Filter signals for this symbol (for overlay display)
  const symbolSignals = useMemo(() => {
    return signals.filter(s => s.symbol === symbol);
  }, [signals, symbol]);

  // Get current values for display
  const currentValues = useMemo(() => {
    if (data.length === 0) return null;
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
  }, [marketStart, marketEnd]);

  return (
    <div className="relative">
      {/* Legend */}
      <div className="flex items-center justify-between text-[10px] mb-1 px-1">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2 h-0.5 bg-amber-500 rounded"></span>
            <span className="text-zinc-500">{symbol} (Binance)</span>
            {currentValues && (
              <span className="font-mono text-zinc-700 dark:text-zinc-300">
                ${currentValues.binance.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>
            )}
            {currentValues && (
              <span className={`font-mono ${currentValues.delta >= 0 ? "text-green-500" : "text-red-500"}`}>
                ({currentValues.delta >= 0 ? "+" : ""}{currentValues.deltaPct.toFixed(2)}%)
              </span>
            )}
          </span>
          {/* Chainlink target indicator */}
          {showPriceToBeat && startPrice && (
            <span className="flex items-center gap-1">
              <span className="w-2 h-0.5 bg-violet-500 rounded border-dashed"></span>
              <span className="text-zinc-500">Chainlink</span>
              <span className="font-mono text-violet-500">
                ${startPrice.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>
            </span>
          )}
        </div>
      </div>

      {/* Chart container with grey overlay for future time */}
      <div className="relative">
        <div ref={containerRef} />

        {/* Grey overlay for future/unfilled portion */}
        {windowProgress && windowProgress.pct < 100 && (
          <div
            className="absolute top-0 right-0 h-full bg-zinc-200/30 dark:bg-zinc-700/30 pointer-events-none"
            style={{
              width: `${Math.max(0, 100 - windowProgress.pct)}%`,
            }}
          />
        )}

        {/* Signal markers as overlays */}
        {marketStart && marketEnd && symbolSignals.map((sig, idx) => {
          const isUp = sig.signal === "BUY_UP" || sig.signal === "BUY_MORE_UP";
          const elapsed = sig.timestamp - marketStart;
          const total = marketEnd - marketStart;
          const leftPct = (elapsed / total) * 100;

          // Don't render if outside window
          if (leftPct < 0 || leftPct > 100) return null;

          return (
            <div
              key={`${sig.timestamp}-${idx}`}
              className="absolute pointer-events-none"
              style={{
                left: `${leftPct}%`,
                top: isUp ? "70%" : "10%",
                transform: "translateX(-50%)",
              }}
              title={`${isUp ? "UP" : "DOWN"} signal - Edge: ${(sig.edge * 100).toFixed(1)}%`}
            >
              {/* Triangle marker */}
              <div
                className={`w-0 h-0 border-l-[5px] border-r-[5px] border-l-transparent border-r-transparent ${
                  isUp
                    ? "border-b-[8px] border-b-green-500"
                    : "border-t-[8px] border-t-red-500"
                }`}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

export const MarketWindowChart = memo(MarketWindowChartComponent);
