"use client";

import { useState, useEffect } from "react";
import { X, Users, TrendingUp, Play, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { WhaleInfo, WhaleTrade } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface BacktestResult {
  whale_name: string;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number;
  roi_pct: number;
  win_rate: number;
  avg_pnl_per_trade: number;
}

interface RealPositionPnL {
  whale_name: string;
  wallet: string;
  total_positions: number;
  settled_positions: number;
  winning_trades: number;
  losing_trades: number;
  total_cost: number;
  current_value?: number;
  total_pnl: number;
  roi_pct: number;
  win_rate: number;
  recent_trade_count?: number;
  data_note?: string;
}

interface FollowingModalProps {
  isOpen: boolean;
  onClose: () => void;
  whales: WhaleInfo[];
  whaleTrades: WhaleTrade[];
  selectedWhale: string | null;
  onWhaleSelect: (whale: string | null) => void;
}

export function FollowingModal({
  isOpen,
  onClose,
  whales,
  whaleTrades,
  selectedWhale,
  onWhaleSelect,
}: FollowingModalProps) {
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestDays, setBacktestDays] = useState(7);
  const [realPnL, setRealPnL] = useState<RealPositionPnL | null>(null);
  const [realPnLLoading, setRealPnLLoading] = useState(false);

  // Fetch real position P&L when whale is selected
  const fetchRealPnL = async (whaleName: string) => {
    setRealPnLLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/whale-following/positions/${whaleName}`);
      if (res.ok) {
        const data = await res.json();
        setRealPnL(data);
      }
    } catch (e) {
      console.error("Failed to fetch real P&L:", e);
    } finally {
      setRealPnLLoading(false);
    }
  };

  // Run backtest for selected whale
  const runBacktest = async (whaleName: string) => {
    setBacktestLoading(true);
    try {
      const res = await fetch(
        `${API_URL}/api/whale-following/backtest/${whaleName}?days=${backtestDays}&position_size=25`
      );
      if (res.ok) {
        const data = await res.json();
        setBacktestResult(data);
      }
    } catch (e) {
      console.error("Backtest failed:", e);
    } finally {
      setBacktestLoading(false);
    }
  };

  // Fetch real P&L when whale changes
  useEffect(() => {
    setBacktestResult(null);
    setRealPnL(null);
    if (selectedWhale) {
      fetchRealPnL(selectedWhale);
    }
  }, [selectedWhale]);

  if (!isOpen) return null;

  // Count trades per whale
  const whaleTradeCount = whales.reduce((acc, whale) => {
    acc[whale.name] = whaleTrades.filter((t) => t.whale === whale.name).length;
    return acc;
  }, {} as Record<string, number>);

  // Get recent trades for selected whale
  const recentTrades = selectedWhale
    ? whaleTrades.filter((t) => t.whale === selectedWhale).slice(0, 10)
    : whaleTrades.slice(0, 10);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white dark:bg-zinc-900 rounded-lg shadow-xl border border-zinc-300 dark:border-zinc-700 w-full max-w-2xl max-h-[80vh] overflow-hidden mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-700">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5 text-cyan-500" />
            <h2 className="text-lg font-semibold">Following</h2>
            <Badge variant="outline" className="text-xs">
              {whales.length} whales
            </Badge>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4 max-h-[60vh] overflow-y-auto">
          {/* Whale selector */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => onWhaleSelect(null)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                selectedWhale === null
                  ? "bg-cyan-500 text-white"
                  : "bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-300 dark:hover:bg-zinc-600"
              }`}
            >
              All
            </button>
            {whales.map((whale) => (
              <button
                key={whale.name}
                onClick={() => onWhaleSelect(whale.name)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                  selectedWhale === whale.name
                    ? "bg-cyan-500 text-white"
                    : "bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-300 dark:hover:bg-zinc-600"
                }`}
              >
                {whale.name}
                <span className="text-xs opacity-70">
                  ({whaleTradeCount[whale.name] || 0})
                </span>
              </button>
            ))}
          </div>

          {/* Whale cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {whales
              .filter((w) => !selectedWhale || w.name === selectedWhale)
              .map((whale) => (
                <div
                  key={whale.name}
                  className={`p-3 rounded-lg border transition-all cursor-pointer ${
                    selectedWhale === whale.name
                      ? "border-cyan-500 bg-cyan-50 dark:bg-cyan-900/20"
                      : "border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800/50 hover:border-zinc-300 dark:hover:border-zinc-600"
                  }`}
                  onClick={() =>
                    onWhaleSelect(selectedWhale === whale.name ? null : whale.name)
                  }
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-cyan-600 dark:text-cyan-400">
                      {whale.name}
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {whaleTradeCount[whale.name] || 0} trades
                    </Badge>
                  </div>
                  <div className="text-xs font-mono text-zinc-500 mb-2">
                    {whale.address}
                  </div>
                  {whale.strategy && (
                    <div className="text-xs text-zinc-400">{whale.strategy}</div>
                  )}
                </div>
              ))}
          </div>

          {/* Real Position P&L - Primary display */}
          {selectedWhale && (
            <div className="p-4 bg-gradient-to-r from-green-500/10 to-cyan-500/10 rounded-lg border border-green-500/20">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-green-500" />
                  <h3 className="text-sm font-semibold">Real Position P&L</h3>
                  <Badge variant="outline" className="text-[10px] bg-green-500/10 text-green-600 border-green-500/30">
                    Actual Data
                  </Badge>
                </div>
                {realPnLLoading && <Loader2 className="w-4 h-4 animate-spin text-green-500" />}
              </div>

              {realPnL && (
                <>
                  <div className="grid grid-cols-4 gap-3 mb-3">
                    <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                      <div className="text-[10px] text-zinc-500 mb-1">Positions</div>
                      <div className="text-lg font-bold">{realPnL.total_positions}</div>
                      <div className="text-[10px] text-zinc-400">{realPnL.settled_positions} settled</div>
                    </div>
                    <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                      <div className="text-[10px] text-zinc-500 mb-1">Win Rate</div>
                      <div className={`text-lg font-bold ${realPnL.win_rate >= 0.5 ? "text-green-500" : "text-red-500"}`}>
                        {realPnL.settled_positions > 0 ? `${(realPnL.win_rate * 100).toFixed(0)}%` : "—"}
                      </div>
                      <div className="text-[10px] text-zinc-400">{realPnL.winning_trades}W / {realPnL.losing_trades}L</div>
                    </div>
                    <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                      <div className="text-[10px] text-zinc-500 mb-1">Total P&L</div>
                      <div className={`text-lg font-bold ${realPnL.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                        ${realPnL.total_pnl.toFixed(0)}
                      </div>
                      <div className="text-[10px] text-zinc-400">${realPnL.total_cost.toFixed(0)} cost</div>
                    </div>
                    <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                      <div className="text-[10px] text-zinc-500 mb-1">ROI</div>
                      <div className={`text-lg font-bold ${realPnL.roi_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                        {realPnL.total_cost > 0 ? `${realPnL.roi_pct.toFixed(1)}%` : "—"}
                      </div>
                    </div>
                  </div>

                  {/* Activity and Performance Summary */}
                  <div className="space-y-2">
                    {realPnL.recent_trade_count && realPnL.recent_trade_count > 0 && (
                      <div className="p-2 bg-cyan-500/10 rounded text-xs text-cyan-600 dark:text-cyan-400">
                        <span className="font-medium">{realPnL.recent_trade_count.toLocaleString()}</span> recent trades
                        {realPnL.recent_trade_count > 100 && " — algorithmic trader"}
                      </div>
                    )}

                    {realPnL.total_positions > 0 && (
                      <div className="p-2 bg-white/30 dark:bg-zinc-800/30 rounded text-xs">
                        <span className="text-zinc-500">Current Session: </span>
                        <span className="text-zinc-700 dark:text-zinc-300">
                          <span className={realPnL.total_pnl >= 0 ? "text-green-500 font-semibold" : "text-red-500 font-semibold"}>
                            ${realPnL.total_pnl >= 0 ? "+" : ""}{realPnL.total_pnl.toFixed(2)}
                          </span>{" "}
                          P&L ({realPnL.winning_trades}W/{realPnL.losing_trades}L) across {realPnL.total_positions} positions
                        </span>
                      </div>
                    )}

                    {realPnL.data_note && (
                      <div className="p-2 bg-amber-500/10 rounded text-[10px] text-amber-600 dark:text-amber-400 italic">
                        Note: {realPnL.data_note}
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Simulated Backtest Section - Secondary */}
          {selectedWhale && (
            <div className="p-4 bg-gradient-to-r from-purple-500/10 to-zinc-500/10 rounded-lg border border-purple-500/20">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Play className="w-4 h-4 text-purple-500" />
                  <h3 className="text-sm font-semibold">Simulated Backtest</h3>
                  <Badge variant="outline" className="text-[10px] bg-purple-500/10 text-purple-600 border-purple-500/30">
                    Estimated
                  </Badge>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={backtestDays}
                    onChange={(e) => setBacktestDays(parseInt(e.target.value))}
                    className="text-xs px-2 py-1 rounded bg-zinc-200 dark:bg-zinc-700 border-none"
                  >
                    <option value={3}>3 days</option>
                    <option value={7}>7 days</option>
                    <option value={14}>14 days</option>
                    <option value={30}>30 days</option>
                  </select>
                  <button
                    onClick={() => runBacktest(selectedWhale)}
                    disabled={backtestLoading}
                    className="flex items-center gap-1 px-3 py-1 rounded bg-purple-500 text-white text-xs font-medium hover:bg-purple-600 disabled:opacity-50 transition-colors"
                  >
                    {backtestLoading ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      <Play className="w-3 h-3" />
                    )}
                    Run
                  </button>
                </div>
              </div>

              <p className="text-xs text-zinc-500 mb-3">
                Simulates following {selectedWhale}&apos;s trades with $25 positions (uses estimated outcomes)
              </p>

              {backtestResult && (
                <div className="grid grid-cols-4 gap-3">
                  <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                    <div className="text-[10px] text-zinc-500 mb-1">Trades</div>
                    <div className="text-lg font-bold">{backtestResult.total_trades}</div>
                  </div>
                  <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                    <div className="text-[10px] text-zinc-500 mb-1">Win Rate</div>
                    <div className={`text-lg font-bold ${backtestResult.win_rate >= 0.5 ? "text-green-500" : "text-red-500"}`}>
                      {(backtestResult.win_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                    <div className="text-[10px] text-zinc-500 mb-1">Total P&L</div>
                    <div className={`text-lg font-bold ${backtestResult.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                      ${backtestResult.total_pnl.toFixed(0)}
                    </div>
                  </div>
                  <div className="p-2 bg-white/50 dark:bg-zinc-800/50 rounded">
                    <div className="text-[10px] text-zinc-500 mb-1">ROI</div>
                    <div className={`text-lg font-bold ${backtestResult.roi_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                      {backtestResult.roi_pct.toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              {backtestResult && backtestResult.total_trades > 0 && (
                <div className="mt-3 p-2 bg-white/30 dark:bg-zinc-800/30 rounded text-xs">
                  <span className="text-zinc-500">Simulated: </span>
                  <span className="text-zinc-700 dark:text-zinc-300">
                    Following with $25 trades: {" "}
                    <span className={backtestResult.total_pnl >= 0 ? "text-green-500" : "text-red-500"}>
                      ${backtestResult.total_pnl.toFixed(2)}
                    </span>{" "}
                    ({backtestResult.winning_trades}W / {backtestResult.losing_trades}L).
                    <span className="text-zinc-400 italic"> Results may differ from real P&L above.</span>
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Recent trades */}
          {recentTrades.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-zinc-500 mb-2">
                Recent Trades {selectedWhale && `(${selectedWhale})`}
              </h3>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {recentTrades.map((trade, i) => (
                  <div
                    key={`${trade.tx_hash}-${i}`}
                    className="flex items-center justify-between text-xs p-2 bg-zinc-100 dark:bg-zinc-800/30 rounded"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-cyan-500">{trade.whale}</span>
                      <Badge
                        className={`text-[10px] ${
                          trade.side === "BUY"
                            ? "bg-green-500/20 text-green-400"
                            : "bg-red-500/20 text-red-400"
                        }`}
                      >
                        {trade.side}
                      </Badge>
                      <span className="text-zinc-500 truncate max-w-[150px]">
                        {trade.market}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-zinc-400 font-mono">
                        ${trade.usd_value.toFixed(0)}
                      </span>
                      <span className="text-zinc-500">
                        {new Date(trade.timestamp * 1000).toLocaleTimeString("en-US", {
                          hour: "2-digit",
                          minute: "2-digit",
                          hour12: false,
                        })}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end p-4 border-t border-zinc-200 dark:border-zinc-700">
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
