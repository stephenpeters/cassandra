"use client";

import { memo, useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  BarChart3,
  Settings,
  RotateCcw,
  AlertTriangle,
  Clock,
  ShieldAlert,
  Wallet,
} from "lucide-react";
import type { TradingAccount, TradingSignal, TradingConfig, LiveTradingStatus } from "@/types";

interface TradingCardProps {
  account: TradingAccount | null;
  signals: TradingSignal[];
  config: TradingConfig | null;
  liveStatus: LiveTradingStatus | null;
  onToggle: () => void;
  onReset: () => void;
  onConfigUpdate: (config: Partial<TradingConfig>) => void;
  onModeChange: (mode: "paper" | "live", apiKey: string) => Promise<{ success: boolean; error?: string }>;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;
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

function formatSlug(slug: string): string {
  // Parse slug like "btc-updown-15m-1767760200" to "BTC @ 11:30 AM ET"
  const parts = slug.split("-");
  if (parts.length < 4) return slug;

  const symbol = parts[0].toUpperCase();
  const timestamp = parseInt(parts[parts.length - 1], 10);

  if (isNaN(timestamp)) return slug;

  const date = new Date(timestamp * 1000);
  const timeStr = date.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZone: "America/New_York",
  });

  return `${symbol} @ ${timeStr} ET`;
}

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}

function TradingCardComponent({
  account,
  signals,
  config,
  liveStatus,
  onToggle,
  onReset,
  onConfigUpdate,
  onModeChange,
}: TradingCardProps) {
  const [showSettings, setShowSettings] = useState(false);
  const [showModeModal, setShowModeModal] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [modeError, setModeError] = useState<string | null>(null);
  const [isChangingMode, setIsChangingMode] = useState(false);
  const [uptime, setUptime] = useState(0);
  const startTimeRef = useRef<number>(Date.now());

  // Load API key from localStorage on mount
  useEffect(() => {
    const savedKey = localStorage.getItem("predmkt_api_key");
    if (savedKey) setApiKey(savedKey);
  }, []);

  const tradingMode = liveStatus?.mode || "paper";
  const isLiveMode = tradingMode === "live";

  const handleModeToggle = () => {
    if (isLiveMode) {
      // Switching to paper mode - show confirmation
      setShowModeModal(true);
    } else {
      // Switching to live mode - show confirmation with warnings
      setShowModeModal(true);
    }
  };

  const confirmModeChange = async () => {
    if (!apiKey) {
      setModeError("API key is required");
      return;
    }

    setIsChangingMode(true);
    setModeError(null);

    const newMode = isLiveMode ? "paper" : "live";
    const result = await onModeChange(newMode, apiKey);

    if (result.success) {
      // Save API key to localStorage on success
      localStorage.setItem("predmkt_api_key", apiKey);
      setShowModeModal(false);
    } else {
      setModeError(result.error || "Failed to change mode");
    }

    setIsChangingMode(false);
  };

  // Track uptime since component mount (proxy for session start)
  useEffect(() => {
    const interval = setInterval(() => {
      setUptime(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  if (!account) {
    return (
      <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
        <CardContent className="flex items-center justify-center h-40">
          <span className="text-zinc-500">Loading paper trading...</span>
        </CardContent>
      </Card>
    );
  }

  const isEnabled = config?.enabled ?? false;
  const isProfit = account.total_pnl >= 0;
  const todayIsProfit = account.today_pnl >= 0;

  return (
    <>
    <Card className={`bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800 ${isLiveMode ? "ring-2 ring-red-500" : ""}`}>
      {/* Mode Banner */}
      <div className={`px-4 py-2 text-center font-bold text-sm ${
        isLiveMode
          ? "bg-red-600 text-white"
          : "bg-zinc-200 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400"
      }`}>
        {isLiveMode ? (
          <span className="flex items-center justify-center gap-2">
            <ShieldAlert className="h-4 w-4" />
            LIVE TRADING - REAL MONEY
            {liveStatus?.wallet && (
              <span className="ml-2 font-mono">
                (${liveStatus.wallet.usdc_balance.toLocaleString(undefined, { minimumFractionDigits: 2 })})
              </span>
            )}
          </span>
        ) : (
          <span className="flex items-center justify-center gap-2">
            📝 PAPER TRADING - SIMULATED
          </span>
        )}
      </div>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
                Trading
              </CardTitle>
              {account.trading_halted && (
                <Badge className="bg-red-500/20 text-red-500">
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  Halted
                </Badge>
              )}
              {liveStatus?.kill_switch_active && (
                <Badge className="bg-orange-500/20 text-orange-500">
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  Kill Switch
                </Badge>
              )}
            </div>
            {/* Uptime display */}
            <div className="flex items-center gap-1 text-xs text-zinc-500">
              <Clock className="h-3 w-3" />
              <span className="font-mono">{formatUptime(uptime)}</span>
              <span className="text-zinc-400">running</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Mode Toggle Button */}
            <button
              onClick={handleModeToggle}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                isLiveMode
                  ? "bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50"
                  : "bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700"
              }`}
              title={isLiveMode ? "Switch to Paper Mode" : "Switch to Live Mode"}
            >
              {isLiveMode ? "Go Paper" : "Go Live"}
            </button>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-1.5 rounded bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
              title="Settings"
            >
              <Settings className="h-4 w-4 text-zinc-600 dark:text-zinc-400" />
            </button>
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-500">
                {isEnabled ? "Active" : "Paused"}
              </span>
              <Switch
                checked={isEnabled}
                onCheckedChange={onToggle}
              />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Account Summary - Different display for Paper vs Live mode */}
        {isLiveMode ? (
          /* Live Mode - Show wallet balance only */
          <div className="grid grid-cols-2 gap-3">
            {/* Live Wallet Balance */}
            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg col-span-2">
              <div className="flex items-center gap-1 text-xs text-red-600 dark:text-red-400 mb-1">
                <Wallet className="h-3 w-3" />
                Live Wallet (USDC)
              </div>
              <div className="text-2xl font-mono font-semibold text-red-600 dark:text-red-400">
                {liveStatus?.wallet
                  ? formatCurrency(liveStatus.wallet.usdc_balance)
                  : "Loading..."}
              </div>
              <div className="text-xs text-zinc-500 mt-1">
                Real funds on Polymarket
              </div>
            </div>

            {/* Open Positions Count */}
            <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
              <div className="text-xs text-zinc-500 mb-1">Open Positions</div>
              <div className="text-lg font-mono font-semibold text-zinc-800 dark:text-zinc-200">
                {liveStatus?.open_positions ?? 0}
              </div>
            </div>

            {/* CLOB Status */}
            <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
              <div className="text-xs text-zinc-500 mb-1">CLOB Status</div>
              <div className={`text-lg font-mono font-semibold ${
                liveStatus?.clob_connected
                  ? "text-green-600 dark:text-green-400"
                  : "text-red-600 dark:text-red-400"
              }`}>
                {liveStatus?.clob_connected ? "Connected" : "Disconnected"}
              </div>
            </div>
          </div>
        ) : (
          /* Paper Mode - Show simulated account stats */
          <div className="grid grid-cols-2 gap-3">
            {/* Balance */}
            <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
              <div className="flex items-center gap-1 text-xs text-zinc-500 mb-1">
                <DollarSign className="h-3 w-3" />
                Paper Balance
              </div>
              <div className="text-lg font-mono font-semibold text-zinc-800 dark:text-zinc-200">
                {formatCurrency(account.balance)}
              </div>
            </div>

            {/* Total P&L */}
            <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
              <div className="flex items-center gap-1 text-xs text-zinc-500 mb-1">
                {isProfit ? (
                  <TrendingUp className="h-3 w-3 text-green-500" />
                ) : (
                  <TrendingDown className="h-3 w-3 text-red-500" />
                )}
                Total P&L
              </div>
              <div
                className={`text-lg font-mono font-semibold ${
                  isProfit ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
                }`}
              >
                {formatCurrency(account.total_pnl)}
                <span className="text-xs ml-1">
                  ({formatPercent(account.total_return_pct)})
                </span>
              </div>
            </div>

            {/* Today P&L */}
            <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
              <div className="flex items-center gap-1 text-xs text-zinc-500 mb-1">
                <BarChart3 className="h-3 w-3" />
                Today
              </div>
              <div
                className={`text-lg font-mono font-semibold ${
                  todayIsProfit ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
                }`}
              >
                {formatCurrency(account.today_pnl)}
              </div>
            </div>

            {/* Win Rate */}
            <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg">
              <div className="text-xs text-zinc-500 mb-1">Win Rate</div>
              <div className="text-lg font-mono font-semibold text-zinc-800 dark:text-zinc-200">
                {account.win_rate.toFixed(1)}%
                <span className="text-xs text-zinc-500 ml-1">
                  ({account.winning_trades}W / {account.losing_trades}L)
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Open Positions - Only show in Paper mode */}
        {!isLiveMode && account.positions.length > 0 && (
          <div>
            <div className="text-xs text-zinc-500 mb-2">Open Positions</div>
            <div className="space-y-1">
              {account.positions.map((pos) => (
                <div
                  key={pos.id}
                  className="flex flex-col p-2 bg-zinc-100 dark:bg-zinc-800/30 rounded text-sm"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-zinc-700 dark:text-zinc-300">
                        {pos.symbol}
                      </span>
                      <Badge
                        className={`text-[10px] ${
                          pos.side === "UP"
                            ? "bg-green-500/20 text-green-500"
                            : "bg-red-500/20 text-red-500"
                        }`}
                      >
                        {pos.side}
                      </Badge>
                      <span className="text-xs text-zinc-500">@ {pos.checkpoint}</span>
                    </div>
                    <div className="text-right">
                      <span className="font-mono text-zinc-600 dark:text-zinc-400">
                        {formatCurrency(pos.cost_basis)}
                      </span>
                    </div>
                  </div>
                  {pos.slug && (
                    <div className="mt-1 text-[10px] font-mono text-zinc-400 truncate">
                      {formatSlug(pos.slug)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recent Signals - Paper mode only */}
        {!isLiveMode && signals.length > 0 && (
          <div>
            <div className="text-xs text-zinc-500 mb-2">Recent Signals</div>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {signals.slice(0, 5).map((sig, i) => (
                <div
                  key={`${sig.symbol}-${sig.timestamp}-${i}`}
                  className="flex flex-col p-2 bg-zinc-100 dark:bg-zinc-800/30 rounded text-xs"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-zinc-700 dark:text-zinc-300">
                        {sig.symbol}
                      </span>
                      <span className="text-zinc-500">@ {sig.checkpoint}</span>
                      <Badge
                        className={`text-[10px] ${
                          sig.signal === "HOLD"
                            ? "bg-zinc-500/20 text-zinc-500"
                            : sig.signal.includes("UP")
                            ? "bg-green-500/20 text-green-500"
                            : "bg-red-500/20 text-red-500"
                        }`}
                      >
                        {sig.signal}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-zinc-500">
                        Edge: {(sig.edge * 100).toFixed(1)}%
                      </span>
                      <span className="text-zinc-400">{formatTime(sig.timestamp)}</span>
                    </div>
                  </div>
                  {sig.slug && (
                    <div className="mt-1 text-[10px] font-mono text-zinc-400 truncate">
                      {formatSlug(sig.slug)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recent Trades - Paper mode only */}
        {!isLiveMode && account.recent_trades.length > 0 && (
          <div>
            <div className="text-xs text-zinc-500 mb-2">Recent Trades</div>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {account.recent_trades.slice(0, 5).map((trade) => (
                <div
                  key={trade.id}
                  className="flex flex-col p-2 bg-zinc-100 dark:bg-zinc-800/30 rounded text-xs"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-zinc-700 dark:text-zinc-300">
                        {trade.symbol}
                      </span>
                      <Badge
                        className={`text-[10px] ${
                          trade.side === "UP"
                            ? "bg-green-500/20 text-green-500"
                            : "bg-red-500/20 text-red-500"
                        }`}
                      >
                        {trade.side}
                      </Badge>
                      <span className="text-zinc-500">
                        {trade.resolution === trade.side ? "WIN" : "LOSS"}
                      </span>
                    </div>
                    <div
                      className={`font-mono ${
                        trade.pnl >= 0
                          ? "text-green-600 dark:text-green-400"
                          : "text-red-600 dark:text-red-400"
                      }`}
                    >
                      {trade.pnl >= 0 ? "+" : ""}
                      {formatCurrency(trade.pnl)}
                    </div>
                  </div>
                  {trade.slug && (
                    <div className="mt-1 text-[10px] font-mono text-zinc-400 truncate">
                      {formatSlug(trade.slug)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Settings Panel - Paper mode only */}
        {!isLiveMode && showSettings && config && (
          <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg space-y-3">
            <div className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
              Settings
            </div>

            {/* Asset Selection */}
            <div>
              <label className="text-xs text-zinc-500 block mb-2">
                Enabled Assets (no restart needed)
              </label>
              <div className="flex gap-4">
                {[
                  { symbol: "BTC", viable: true, volume: "$165K" },
                  { symbol: "ETH", viable: true, volume: "$52K" },
                ].map(({ symbol, viable, volume }) => {
                  const enabled = config.enabled_assets?.includes(symbol) ?? symbol === "BTC";
                  return (
                    <label
                      key={symbol}
                      className={`flex items-center gap-2 p-2 rounded border cursor-pointer transition-colors ${
                        enabled
                          ? "bg-green-500/10 border-green-500/30"
                          : "bg-zinc-200 dark:bg-zinc-800 border-zinc-300 dark:border-zinc-700"
                      } ${!viable ? "opacity-50 cursor-not-allowed" : ""}`}
                    >
                      <input
                        type="checkbox"
                        checked={enabled}
                        disabled={!viable}
                        onChange={(e) => {
                          const current = config.enabled_assets || ["BTC"];
                          const updated = e.target.checked
                            ? [...current, symbol]
                            : current.filter((s) => s !== symbol);
                          // Ensure at least one asset is enabled
                          if (updated.length > 0) {
                            onConfigUpdate({ enabled_assets: updated });
                          }
                        }}
                        className="rounded border-zinc-400"
                      />
                      <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                        {symbol}
                      </span>
                      <span className="text-[10px] text-zinc-500">
                        {volume}
                      </span>
                    </label>
                  );
                })}
              </div>
              <p className="text-[10px] text-zinc-500 mt-1">
                SOL/XRP/DOGE disabled due to low volume (&lt;$20K)
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <label className="text-xs text-zinc-500">Starting Balance</label>
                <input
                  type="number"
                  value={config.starting_balance}
                  onChange={(e) =>
                    onConfigUpdate({ starting_balance: parseFloat(e.target.value) })
                  }
                  className="w-full mt-1 px-2 py-1 bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded text-sm font-mono"
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500">Max Position $</label>
                <input
                  type="number"
                  step="100"
                  value={config.max_position_usd || 5000}
                  onChange={(e) =>
                    onConfigUpdate({ max_position_usd: parseFloat(e.target.value) })
                  }
                  className="w-full mt-1 px-2 py-1 bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded text-sm font-mono"
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500">Slippage %</label>
                <input
                  type="number"
                  step="0.1"
                  value={config.slippage_pct}
                  onChange={(e) =>
                    onConfigUpdate({ slippage_pct: parseFloat(e.target.value) })
                  }
                  className="w-full mt-1 px-2 py-1 bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded text-sm font-mono"
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500">Max Position %</label>
                <input
                  type="number"
                  step="0.5"
                  value={config.max_position_pct}
                  onChange={(e) =>
                    onConfigUpdate({ max_position_pct: parseFloat(e.target.value) })
                  }
                  className="w-full mt-1 px-2 py-1 bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded text-sm font-mono"
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500">Daily Loss Limit %</label>
                <input
                  type="number"
                  step="1"
                  value={config.daily_loss_limit_pct}
                  onChange={(e) =>
                    onConfigUpdate({ daily_loss_limit_pct: parseFloat(e.target.value) })
                  }
                  className="w-full mt-1 px-2 py-1 bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded text-sm font-mono"
                />
              </div>
            </div>

            <button
              onClick={onReset}
              className="flex items-center gap-1 px-3 py-1.5 text-xs bg-red-500/10 hover:bg-red-500/20 text-red-600 dark:text-red-400 rounded transition-colors"
            >
              <RotateCcw className="h-3 w-3" />
              Reset Account
            </button>
          </div>
        )}

        {/* Halted Warning */}
        {account.trading_halted && (
          <div className="p-3 bg-red-500/10 rounded-lg">
            <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-sm">
              <AlertTriangle className="h-4 w-4" />
              {account.halt_reason}
            </div>
          </div>
        )}
      </CardContent>
    </Card>

    {/* Mode Change Confirmation Modal */}
    {showModeModal && (
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <div
          className="absolute inset-0 bg-black/50 backdrop-blur-sm"
          onClick={() => setShowModeModal(false)}
        />
        <div className="relative bg-white dark:bg-zinc-900 rounded-lg shadow-xl border border-zinc-300 dark:border-zinc-700 w-full max-w-md mx-4 p-6">
          <h3 className="text-lg font-semibold mb-4">
            {isLiveMode ? "Switch to Paper Mode?" : "⚠️ Switch to LIVE Mode?"}
          </h3>

          {!isLiveMode && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4">
              <div className="flex items-start gap-2 text-red-600 dark:text-red-400">
                <ShieldAlert className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="font-medium">You are about to enable LIVE trading!</p>
                  <ul className="mt-2 list-disc list-inside space-y-1">
                    <li>Real USDC will be used for trades</li>
                    <li>All trades will execute on Polymarket</li>
                    <li>Losses are real and permanent</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          <div className="mb-4">
            <label className="text-sm text-zinc-500 block mb-2">API Key (required)</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API key"
              className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700 rounded text-sm font-mono"
            />
            <p className="text-xs text-zinc-400 mt-1">
              Your API key is stored locally and never sent to third parties.
            </p>
          </div>

          {modeError && (
            <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 rounded text-sm text-red-600 dark:text-red-400">
              {modeError}
            </div>
          )}

          <div className="flex justify-end gap-3">
            <button
              onClick={() => setShowModeModal(false)}
              className="px-4 py-2 text-sm text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={confirmModeChange}
              disabled={isChangingMode || !apiKey}
              className={`px-4 py-2 text-sm rounded transition-colors ${
                isLiveMode
                  ? "bg-zinc-600 hover:bg-zinc-700 text-white"
                  : "bg-red-600 hover:bg-red-700 text-white"
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {isChangingMode
                ? "Switching..."
                : isLiveMode
                ? "Switch to Paper"
                : "Enable LIVE Trading"}
            </button>
          </div>
        </div>
      </div>
    )}
    </>
  );
}

// TradingCard supports both paper and live trading modes
export const TradingCard = memo(TradingCardComponent);
// Backwards compatibility alias (deprecated)
export const PaperTradingCard = TradingCard;
