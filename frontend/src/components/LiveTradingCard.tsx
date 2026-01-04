"use client";

import { useState, useEffect } from "react";
import {
  LiveTradingStatus,
  TradingMode,
  LiveOrder,
  CircuitBreaker,
  WalletBalance,
} from "@/types";

interface LiveTradingCardProps {
  status: LiveTradingStatus | null;
  orders: LiveOrder[];
  onModeChange: (mode: TradingMode) => void;
  onKillSwitch: (activate: boolean, reason?: string) => void;
  onResetCircuitBreaker: () => void;
  onSetAssets: (assets: string[]) => void;
}

export function LiveTradingCard({
  status,
  orders,
  onModeChange,
  onKillSwitch,
  onResetCircuitBreaker,
  onSetAssets,
}: LiveTradingCardProps) {
  const [selectedMode, setSelectedMode] = useState<TradingMode>("paper");
  const [showConfirmLive, setShowConfirmLive] = useState(false);

  useEffect(() => {
    if (status) {
      setSelectedMode(status.mode);
    }
  }, [status]);

  if (!status) {
    return (
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-lg font-semibold text-foreground mb-2">
          Live Trading
        </h3>
        <p className="text-muted-foreground text-sm">Loading...</p>
      </div>
    );
  }

  const handleModeChange = (mode: TradingMode) => {
    if (mode === "live" && selectedMode !== "live") {
      setShowConfirmLive(true);
    } else {
      setSelectedMode(mode);
      onModeChange(mode);
    }
  };

  const confirmLiveMode = () => {
    setSelectedMode("live");
    onModeChange("live");
    setShowConfirmLive(false);
  };

  const modeColors: Record<TradingMode, string> = {
    paper: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    shadow: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    live: "bg-green-500/20 text-green-400 border-green-500/30",
  };

  const statusColor =
    status.kill_switch_active || status.circuit_breaker.triggered
      ? "bg-red-500/20 border-red-500/50"
      : "bg-card border-border";

  return (
    <div className={`rounded-lg p-4 border ${statusColor}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-foreground">Live Trading</h3>
        <span
          className={`px-2 py-1 rounded text-xs font-medium border ${modeColors[status.mode]}`}
        >
          {status.mode.toUpperCase()}
        </span>
      </div>

      {/* Kill Switch Alert */}
      {status.kill_switch_active && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 mb-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-red-400 font-semibold">KILL SWITCH ACTIVE</p>
              <p className="text-red-300 text-sm">All trading halted</p>
            </div>
            <button
              onClick={() => onKillSwitch(false)}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-sm"
            >
              Resume
            </button>
          </div>
        </div>
      )}

      {/* Circuit Breaker Alert */}
      {status.circuit_breaker.triggered && !status.kill_switch_active && (
        <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-3 mb-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-orange-400 font-semibold">CIRCUIT BREAKER</p>
              <p className="text-orange-300 text-sm">
                {status.circuit_breaker.reason}
              </p>
            </div>
            <button
              onClick={onResetCircuitBreaker}
              className="px-3 py-1 bg-orange-600 hover:bg-orange-700 text-white rounded text-sm"
            >
              Reset
            </button>
          </div>
        </div>
      )}

      {/* Wallet Balance (Live Mode Only) */}
      {status.wallet && (
        <div className="bg-muted/30 rounded-lg p-3 mb-4">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-muted-foreground">USDC Balance:</span>
              <span className="text-foreground ml-2">
                ${status.wallet.usdc_balance.toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Locked:</span>
              <span className="text-foreground ml-2">
                ${status.wallet.collateral_locked.toLocaleString()}
              </span>
            </div>
            <div className="col-span-2">
              <span className="text-muted-foreground">Available:</span>
              <span className="text-green-400 ml-2 font-medium">
                ${status.wallet.available_for_trading.toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Mode Selection */}
      <div className="mb-4">
        <label className="block text-sm text-muted-foreground mb-2">
          Trading Mode
        </label>
        <div className="flex gap-2">
          {(["paper", "shadow", "live"] as TradingMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => handleModeChange(mode)}
              className={`flex-1 py-2 px-3 rounded text-sm font-medium transition-colors ${
                selectedMode === mode
                  ? modeColors[mode]
                  : "bg-muted/50 text-muted-foreground hover:bg-muted"
              }`}
            >
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          {selectedMode === "paper" && "Simulated trades only - no real money"}
          {selectedMode === "shadow" &&
            "Live signals, paper execution - validation mode"}
          {selectedMode === "live" && "Real money trades - use with caution!"}
        </p>
      </div>

      {/* Asset Selection */}
      <div className="mb-4">
        <label className="block text-sm text-muted-foreground mb-2">
          Enabled Assets
        </label>
        <div className="flex gap-2">
          {["BTC", "ETH", "SOL", "XRP", "DOGE"].map((asset) => {
            const isEnabled = status.enabled_assets.includes(asset);
            const isViable = asset === "BTC" || asset === "ETH";
            return (
              <button
                key={asset}
                onClick={() => {
                  const newAssets = isEnabled
                    ? status.enabled_assets.filter((a) => a !== asset)
                    : [...status.enabled_assets, asset];
                  if (newAssets.length > 0) {
                    onSetAssets(newAssets);
                  }
                }}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  isEnabled
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted/50 text-muted-foreground hover:bg-muted"
                } ${!isViable ? "opacity-50" : ""}`}
                title={!isViable ? "Low volume - not recommended" : ""}
              >
                {asset}
              </button>
            );
          })}
        </div>
      </div>

      {/* Status Info */}
      <div className="grid grid-cols-2 gap-4 text-sm mb-4">
        <div>
          <span className="text-muted-foreground">Open Positions:</span>
          <span className="text-foreground ml-2">{status.open_positions}</span>
        </div>
        <div>
          <span className="text-muted-foreground">CLOB:</span>
          <span
            className={`ml-2 ${status.clob_connected ? "text-green-400" : "text-red-400"}`}
          >
            {status.clob_connected ? "Connected" : "Disconnected"}
          </span>
        </div>
        <div>
          <span className="text-muted-foreground">Daily Loss:</span>
          <span className="text-foreground ml-2">
            ${status.circuit_breaker.daily_loss_usd.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-muted-foreground">Losses:</span>
          <span className="text-foreground ml-2">
            {status.circuit_breaker.consecutive_losses}
          </span>
        </div>
      </div>

      {/* Kill Switch */}
      {!status.kill_switch_active && (
        <button
          onClick={() => onKillSwitch(true, "Manual activation")}
          className="w-full py-2 bg-red-600 hover:bg-red-700 text-white rounded font-medium transition-colors"
        >
          EMERGENCY STOP
        </button>
      )}

      {/* Recent Orders */}
      {orders.length > 0 && (
        <div className="mt-4 pt-4 border-t border-border">
          <h4 className="text-sm font-medium text-muted-foreground mb-2">
            Recent Orders
          </h4>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {orders.slice(0, 5).map((order) => (
              <div
                key={order.id}
                className="flex items-center justify-between text-xs"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={
                      order.side === "UP" ? "text-green-400" : "text-red-400"
                    }
                  >
                    {order.symbol} {order.side}
                  </span>
                  <span className="text-muted-foreground">
                    ${order.size_usd.toFixed(0)}
                  </span>
                </div>
                <span
                  className={`px-1.5 py-0.5 rounded text-xs ${
                    order.status === "filled"
                      ? "bg-green-500/20 text-green-400"
                      : order.status === "failed"
                        ? "bg-red-500/20 text-red-400"
                        : order.status === "shadow"
                          ? "bg-yellow-500/20 text-yellow-400"
                          : "bg-muted text-muted-foreground"
                  }`}
                >
                  {order.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Confirm Live Mode Modal */}
      {showConfirmLive && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border border-border rounded-lg p-6 max-w-md mx-4">
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Enable Live Trading?
            </h3>
            <p className="text-muted-foreground mb-4">
              This will enable real money trading. Make sure you have:
            </p>
            <ul className="text-sm text-muted-foreground mb-4 space-y-1">
              <li>- USDC balance in your Polymarket wallet</li>
              <li>- Approved token allowances for trading</li>
              <li>- Verified your risk limits are appropriate</li>
              <li>- Tested thoroughly in shadow mode first</li>
            </ul>
            <div className="flex gap-2">
              <button
                onClick={() => setShowConfirmLive(false)}
                className="flex-1 py-2 bg-muted text-foreground rounded"
              >
                Cancel
              </button>
              <button
                onClick={confirmLiveMode}
                className="flex-1 py-2 bg-green-600 hover:bg-green-700 text-white rounded font-medium"
              >
                Enable Live
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
