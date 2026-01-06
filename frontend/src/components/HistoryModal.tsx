"use client";

import { useState, useEffect, useMemo } from "react";
import { X, Database, Download, ChevronLeft, TrendingUp, TrendingDown, BarChart3 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { MarketWindowChart, type MarketWindowDataPoint } from "@/components/charts/MarketWindowChart";

interface HistoricalMarket {
  symbol: string;
  market_start: number;
  market_end: number;
  start_time_str: string;
  end_time_str: string;
  binance_open: number;
  binance_close: number;
  price_change: number;
  price_change_pct: number;
  resolution: "UP" | "DOWN";
  signals: Array<{
    symbol: string;
    side: string;
    edge: number;
    confidence: number;
    timestamp: number;
    reason: string;
  }>;
  chart_data: {
    data: MarketWindowDataPoint[];
    start_price: number;
  } | null;
  recorded_at: number;
}

interface HistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function HistoryModal({ isOpen, onClose }: HistoryModalProps) {
  const [markets, setMarkets] = useState<HistoricalMarket[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMarket, setSelectedMarket] = useState<HistoricalMarket | null>(null);
  const [filterSymbol, setFilterSymbol] = useState<string | null>(null);
  const [limit, setLimit] = useState(20);

  // Fetch historical data
  useEffect(() => {
    if (!isOpen) return;

    const fetchHistory = async () => {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams();
        if (filterSymbol) params.set("symbol", filterSymbol);
        params.set("limit", limit.toString());

        const res = await fetch(`http://localhost:8000/api/markets-15m/history?${params}`);
        if (!res.ok) throw new Error("Failed to fetch history");
        const data = await res.json();
        setMarkets(data.markets || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load history");
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [isOpen, filterSymbol, limit]);

  // Calculate stats
  const stats = useMemo(() => {
    if (markets.length === 0) return null;
    const upCount = markets.filter((m) => m.resolution === "UP").length;
    const avgChange = markets.reduce((acc, m) => acc + Math.abs(m.price_change_pct), 0) / markets.length;
    return {
      total: markets.length,
      upCount,
      downCount: markets.length - upCount,
      upPct: ((upCount / markets.length) * 100).toFixed(0),
      avgChange: avgChange.toFixed(3),
    };
  }, [markets]);

  // Download data
  const handleDownload = (format: "json" | "csv") => {
    if (markets.length === 0) return;

    if (format === "json") {
      const blob = new Blob([JSON.stringify(markets, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `polymarket-history-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } else {
      // CSV export (flattened)
      const headers = [
        "symbol",
        "start_time",
        "end_time",
        "binance_open",
        "binance_close",
        "price_change",
        "price_change_pct",
        "resolution",
        "signals_count",
      ];
      const rows = markets.map((m) => [
        m.symbol,
        m.start_time_str,
        m.end_time_str,
        m.binance_open,
        m.binance_close,
        m.price_change,
        m.price_change_pct,
        m.resolution,
        m.signals.length,
      ]);
      const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `polymarket-history-${new Date().toISOString().slice(0, 10)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white dark:bg-zinc-900 rounded-lg shadow-xl border border-zinc-300 dark:border-zinc-700 w-full max-w-4xl max-h-[85vh] overflow-hidden mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-700">
          <div className="flex items-center gap-2">
            {selectedMarket ? (
              <button
                onClick={() => setSelectedMarket(null)}
                className="p-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors mr-1"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
            ) : (
              <Database className="w-5 h-5 text-purple-500" />
            )}
            <h2 className="text-lg font-semibold">
              {selectedMarket
                ? `${selectedMarket.symbol} - ${selectedMarket.start_time_str}`
                : "Market History"}
            </h2>
            {!selectedMarket && stats && (
              <Badge variant="outline" className="text-xs ml-2">
                {stats.total} markets
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            {!selectedMarket && (
              <>
                <button
                  onClick={() => handleDownload("csv")}
                  className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-zinc-200 dark:bg-zinc-700 hover:bg-zinc-300 dark:hover:bg-zinc-600"
                  title="Download CSV"
                >
                  <Download className="w-3 h-3" />
                  CSV
                </button>
                <button
                  onClick={() => handleDownload("json")}
                  className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-zinc-200 dark:bg-zinc-700 hover:bg-zinc-300 dark:hover:bg-zinc-600"
                  title="Download JSON"
                >
                  <Download className="w-3 h-3" />
                  JSON
                </button>
              </>
            )}
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        {selectedMarket ? (
          // Detail view
          <div className="p-4 space-y-4 max-h-[calc(85vh-120px)] overflow-y-auto">
            {/* Market summary */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-2xl font-bold">{selectedMarket.symbol}</span>
                <Badge
                  className={`${
                    selectedMarket.resolution === "UP"
                      ? "bg-green-500/20 text-green-400"
                      : "bg-red-500/20 text-red-400"
                  }`}
                >
                  {selectedMarket.resolution === "UP" ? (
                    <TrendingUp className="w-4 h-4 mr-1" />
                  ) : (
                    <TrendingDown className="w-4 h-4 mr-1" />
                  )}
                  {selectedMarket.resolution}
                </Badge>
              </div>
              <div className="text-right">
                <div
                  className={`text-xl font-mono ${
                    selectedMarket.price_change >= 0 ? "text-green-500" : "text-red-500"
                  }`}
                >
                  {selectedMarket.price_change >= 0 ? "+" : ""}
                  {selectedMarket.price_change_pct.toFixed(4)}%
                </div>
                <div className="text-xs text-zinc-500">
                  ${selectedMarket.binance_open.toLocaleString()} → $
                  {selectedMarket.binance_close.toLocaleString()}
                </div>
              </div>
            </div>

            {/* Time info */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
                <div className="text-xs text-zinc-500 mb-1">Window Start</div>
                <div className="font-mono">{selectedMarket.start_time_str}</div>
              </div>
              <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
                <div className="text-xs text-zinc-500 mb-1">Window End</div>
                <div className="font-mono">{selectedMarket.end_time_str}</div>
              </div>
            </div>

            {/* Chart */}
            <div className="border border-zinc-300 dark:border-zinc-700 rounded-lg p-3 bg-zinc-50 dark:bg-zinc-800/30">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-4 h-4 text-zinc-500" />
                <span className="text-sm font-medium">Price vs Odds Chart</span>
              </div>
              {selectedMarket.chart_data && selectedMarket.chart_data.data.length > 0 ? (
                <MarketWindowChart
                  symbol={selectedMarket.symbol}
                  data={selectedMarket.chart_data.data}
                  startPrice={selectedMarket.chart_data.start_price}
                  priceRange={500}
                  height={200}
                />
              ) : (
                <div className="h-[200px] flex items-center justify-center text-sm text-zinc-500">
                  No chart data available for this market
                </div>
              )}
            </div>

            {/* Signals */}
            <div>
              <div className="text-sm font-medium text-zinc-500 mb-2">
                Trading Signals ({selectedMarket.signals.length})
              </div>
              {selectedMarket.signals.length > 0 ? (
                <div className="space-y-2">
                  {selectedMarket.signals.map((sig, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between p-2 bg-zinc-100 dark:bg-zinc-800/30 rounded text-sm"
                    >
                      <div className="flex items-center gap-2">
                        <Badge
                          className={`text-[10px] ${
                            sig.side === "UP"
                              ? "bg-green-500/20 text-green-400"
                              : "bg-red-500/20 text-red-400"
                          }`}
                        >
                          {sig.side}
                        </Badge>
                        <span className="text-xs text-zinc-400 max-w-[200px] truncate">
                          {sig.reason}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-xs">
                        <span className="text-zinc-500">Edge: {sig.edge.toFixed(1)}%</span>
                        <span className="text-zinc-400 font-mono">
                          {new Date(sig.timestamp * 1000).toLocaleTimeString("en-US", {
                            hour: "2-digit",
                            minute: "2-digit",
                            second: "2-digit",
                            hour12: false,
                          })}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-zinc-400 italic">No signals generated</div>
              )}
            </div>
          </div>
        ) : (
          // List view
          <div className="p-4 space-y-4 max-h-[calc(85vh-120px)] overflow-y-auto">
            {/* Filters */}
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setFilterSymbol(null)}
                  className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    filterSymbol === null
                      ? "bg-purple-500 text-white"
                      : "bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-400"
                  }`}
                >
                  All
                </button>
                {["BTC", "ETH", "SOL"].map((sym) => (
                  <button
                    key={sym}
                    onClick={() => setFilterSymbol(sym)}
                    className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                      filterSymbol === sym
                        ? "bg-purple-500 text-white"
                        : "bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-400"
                    }`}
                  >
                    {sym}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-zinc-500">Show:</span>
                <select
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value))}
                  className="px-2 py-1 rounded bg-zinc-200 dark:bg-zinc-700 text-xs"
                >
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={50}>50</option>
                </select>
              </div>
            </div>

            {/* Stats bar */}
            {stats && (
              <div className="flex items-center gap-4 p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg text-sm">
                <div>
                  <span className="text-zinc-500">Total: </span>
                  <span className="font-medium">{stats.total}</span>
                </div>
                <div>
                  <span className="text-green-500">{stats.upCount} UP</span>
                  <span className="text-zinc-400 mx-1">/</span>
                  <span className="text-red-500">{stats.downCount} DOWN</span>
                </div>
                <div>
                  <span className="text-zinc-500">UP Rate: </span>
                  <span className="font-medium">{stats.upPct}%</span>
                </div>
                <div>
                  <span className="text-zinc-500">Avg Move: </span>
                  <span className="font-medium">{stats.avgChange}%</span>
                </div>
              </div>
            )}

            {/* Market grid */}
            {loading ? (
              <div className="text-center text-zinc-500 py-8">Loading...</div>
            ) : error ? (
              <div className="text-center text-red-500 py-8">{error}</div>
            ) : markets.length === 0 ? (
              <div className="text-center text-zinc-500 py-8">
                No historical markets yet. Markets are stored as they resolve.
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {markets.map((market) => (
                  <button
                    key={`${market.symbol}-${market.market_start}`}
                    onClick={() => setSelectedMarket(market)}
                    className="p-3 rounded-lg bg-zinc-100 dark:bg-zinc-800/50 hover:bg-zinc-200 dark:hover:bg-zinc-800 transition-all text-left border border-transparent hover:border-purple-500/50"
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold">{market.symbol}</span>
                      <Badge
                        className={`text-[10px] ${
                          market.resolution === "UP"
                            ? "bg-green-500/20 text-green-400"
                            : "bg-red-500/20 text-red-400"
                        }`}
                      >
                        {market.resolution}
                      </Badge>
                    </div>

                    {/* Time */}
                    <div className="text-xs text-zinc-500 mb-2">
                      {new Date(market.market_start * 1000).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                      })}{" "}
                      {new Date(market.market_start * 1000).toLocaleTimeString("en-US", {
                        hour: "2-digit",
                        minute: "2-digit",
                        hour12: false,
                      })}
                    </div>

                    {/* Price change */}
                    <div
                      className={`text-sm font-mono ${
                        market.price_change >= 0 ? "text-green-500" : "text-red-500"
                      }`}
                    >
                      {market.price_change >= 0 ? "+" : ""}
                      {market.price_change_pct.toFixed(3)}%
                    </div>

                    {/* Mini info */}
                    <div className="flex items-center gap-2 mt-2 text-[10px] text-zinc-400">
                      {market.signals.length > 0 && (
                        <span>{market.signals.length} signals</span>
                      )}
                      {market.chart_data && <span>chart</span>}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-zinc-200 dark:border-zinc-700">
          <div className="text-xs text-zinc-500">
            Markets are stored as they resolve. Download data for offline analysis.
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded bg-zinc-200 dark:bg-zinc-700 text-sm font-medium hover:bg-zinc-300 dark:hover:bg-zinc-600 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
