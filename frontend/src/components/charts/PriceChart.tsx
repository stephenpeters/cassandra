"use client";

import { useEffect, useRef, memo } from "react";
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  Time,
  ColorType,
} from "lightweight-charts";
import type { CandleData } from "@/types";

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

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      width: containerRef.current.clientWidth,
      height: height,
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: "#6b7280",
          style: 2,
        },
        horzLine: {
          width: 1,
          color: "#6b7280",
          style: 2,
        },
      },
      timeScale: {
        borderColor: "#374151",
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: "#374151",
      },
    });

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    // Volume series
    const volumeSeries = chart.addHistogramSeries({
      color: "#3b82f6",
      priceFormat: {
        type: "volume",
      },
      priceScaleId: "",
    });

    volumeSeries.priceScale().applyOptions({
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
  }, [height]);

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

    const volumeData = candles.map((c) => ({
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
      <div className="absolute top-2 left-2 z-10 bg-zinc-900/80 px-2 py-1 rounded text-sm font-mono">
        {symbol}
      </div>
      <div ref={containerRef} />
    </div>
  );
}

export const PriceChart = memo(PriceChartComponent);
