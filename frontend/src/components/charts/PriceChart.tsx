"use client";

import { useEffect, useRef, memo } from "react";
import { createChart, ColorType, CandlestickSeries, HistogramSeries } from "lightweight-charts";
import type { IChartApi, ISeriesApi, CandlestickData, HistogramData, Time } from "lightweight-charts";
import type { CandleData } from "@/types";
import { useTheme } from "@/components/ThemeProvider";

interface PriceChartProps {
  symbol: string;
  candles: CandleData[];
  height?: number;
}

function PriceChartComponent({ symbol, candles, height = 300 }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const { theme } = useTheme();

  // Theme-aware colors
  const colors = theme === "dark" ? {
    text: "#9ca3af",
    grid: "#1f2937",
    crosshair: "#6b7280",
    border: "#374151",
  } : {
    text: "#374151",
    grid: "#e5e7eb",
    crosshair: "#9ca3af",
    border: "#d1d5db",
  };

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: colors.text,
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid },
      },
      width: containerRef.current.clientWidth,
      height: height,
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: colors.crosshair,
          style: 2,
        },
        horzLine: {
          width: 1,
          color: colors.crosshair,
          style: 2,
        },
      },
      timeScale: {
        borderColor: colors.border,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: colors.border,
      },
    });

    // Candlestick series - v5 API
    const candleSeries = chart.addSeries(
      CandlestickSeries,
      {
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderUpColor: "#22c55e",
        borderDownColor: "#ef4444",
        wickUpColor: "#22c55e",
        wickDownColor: "#ef4444",
      }
    );

    // Volume series - v5 API
    const volumeSeries = chart.addSeries(
      HistogramSeries,
      {
        color: "#3b82f6",
        priceFormat: {
          type: "volume",
        },
        priceScaleId: "volume",
      }
    );

    chart.priceScale("volume").applyOptions({
      scaleMargins: {
        top: 0.85,
        bottom: 0,
      },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

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
  }, [height, theme, colors.text, colors.grid, colors.crosshair, colors.border]);

  // Update data
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current || !candles.length)
      return;

    const candleData: CandlestickData<Time>[] = candles.map((c) => ({
      time: c.time as Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));

    const volumeData: HistogramData<Time>[] = candles.map((c) => ({
      time: c.time as Time,
      value: c.volume || 0,
      color: c.close >= c.open ? "#22c55e40" : "#ef444440",
    }));

    candleSeriesRef.current.setData(candleData);
    volumeSeriesRef.current.setData(volumeData);

    // Fit content
    chartRef.current?.timeScale().fitContent();
  }, [candles]);

  // Update last candle in real-time
  useEffect(() => {
    if (
      !candleSeriesRef.current ||
      !volumeSeriesRef.current ||
      !candles.length
    )
      return;

    const lastCandle = candles[candles.length - 1];
    if (!lastCandle) return;

    candleSeriesRef.current.update({
      time: lastCandle.time as Time,
      open: lastCandle.open,
      high: lastCandle.high,
      low: lastCandle.low,
      close: lastCandle.close,
    });

    volumeSeriesRef.current.update({
      time: lastCandle.time as Time,
      value: lastCandle.volume || 0,
      color: lastCandle.close >= lastCandle.open ? "#22c55e40" : "#ef444440",
    });
  }, [candles]);

  return (
    <div className="relative">
      <div className="absolute top-2 left-2 z-10 bg-zinc-200/90 dark:bg-zinc-900/80 text-zinc-800 dark:text-zinc-100 px-2 py-1 rounded text-sm font-mono">
        {symbol}
      </div>
      <div ref={containerRef} />
    </div>
  );
}

export const PriceChart = memo(PriceChartComponent);
