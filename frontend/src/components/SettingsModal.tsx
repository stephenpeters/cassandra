"use client";

import { useState, useEffect } from "react";
import { X, Settings, Check, Clock } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { InfoTooltip } from "@/components/ui/info-tooltip";
import { SimpleTooltip } from "@/components/ui/tooltip";
import type { TradingConfig } from "@/types";

// Market timeframes - 15m is active, others planned
const MARKET_TIMEFRAMES = [
  { id: "15m", label: "15 Min", active: true, description: "Currently available" },
  { id: "1h", label: "1 Hour", active: false, description: "Coming soon" },
  { id: "4h", label: "4 Hour", active: false, description: "Coming soon" },
  { id: "1d", label: "Daily", active: false, description: "Coming soon" },
];

// Available symbols with volume tiers
const MARKET_SYMBOLS = [
  { symbol: "BTC", tier: "primary", volume: "165K", recommended: true },
  { symbol: "ETH", tier: "primary", volume: "52K", recommended: true },
  { symbol: "SOL", tier: "secondary", volume: "15K", recommended: false },
  { symbol: "XRP", tier: "secondary", volume: "8K", recommended: false },
  { symbol: "DOGE", tier: "secondary", volume: "5K", recommended: false },
];

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: TradingConfig | null;
  onConfigUpdate: (config: Partial<TradingConfig>) => void;
}

function formatSecondsToTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs}s`;
}

export function SettingsModal({ isOpen, onClose, config, onConfigUpdate }: SettingsModalProps) {
  const [localConfig, setLocalConfig] = useState<Partial<TradingConfig>>({});

  useEffect(() => {
    if (config) {
      setLocalConfig({
        min_edge_pct: config.min_edge_pct,
        min_time_remaining_sec: config.min_time_remaining_sec,
        cooldown_sec: config.cooldown_sec,
        entry_time_up_sec: config.entry_time_up_sec ?? 450,
        entry_time_down_sec: config.entry_time_down_sec ?? 450,
        require_volume_confirmation: config.require_volume_confirmation,
        require_orderbook_confirmation: config.require_orderbook_confirmation,
        min_volume_delta_usd: config.min_volume_delta_usd,
        min_orderbook_imbalance: config.min_orderbook_imbalance,
        enabled_assets: config.enabled_assets,
        signal_checkpoints: config.signal_checkpoints ?? [180, 300, 450, 540, 720],
        active_checkpoint: config.active_checkpoint ?? 450,
      });
    }
  }, [config]);

  // Available checkpoint options (seconds)
  const CHECKPOINT_OPTIONS = [
    { sec: 180, label: "3m", description: "Early - high risk, high reward" },
    { sec: 300, label: "5m", description: "Moderate timing" },
    { sec: 450, label: "7m30s", description: "Default - balanced" },
    { sec: 540, label: "9m", description: "Best EV historically" },
    { sec: 720, label: "12m", description: "Conservative - less time" },
  ];

  const formatCheckpointLabel = (sec: number): string => {
    const mins = Math.floor(sec / 60);
    const secs = sec % 60;
    if (secs === 30) return `${mins}m30s`;
    if (secs > 0) return `${mins}m${secs}s`;
    return `${mins}m`;
  };

  if (!isOpen) return null;

  const handleSave = () => {
    onConfigUpdate(localConfig);
    onClose();
  };

  const handleAssetToggle = (asset: string) => {
    const current = localConfig.enabled_assets || [];
    const updated = current.includes(asset)
      ? current.filter((a) => a !== asset)
      : [...current, asset];
    setLocalConfig({ ...localConfig, enabled_assets: updated });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white dark:bg-zinc-900 rounded-lg shadow-xl border border-zinc-300 dark:border-zinc-700 w-full max-w-lg max-h-[80vh] overflow-y-auto mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-700">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-zinc-500" />
            <h2 className="text-lg font-semibold">Trading Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-6">
          {/* Strategy Overview */}
          <div className="p-3 bg-blue-500/10 dark:bg-blue-500/10 rounded-lg border border-blue-500/20">
            <p className="text-xs text-blue-600 dark:text-blue-400">
              <strong>Strategy:</strong> Latency arbitrage. When Binance price moves during a 15-min window,
              Polymarket odds lag behind by 30-60 seconds. We buy when the gap (edge) exceeds your threshold.
            </p>
          </div>

          {/* Section 1: Market Grid (Symbol x Timeframe) */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 border-b border-zinc-200 dark:border-zinc-700 pb-1">
              Markets
            </h3>
            <p className="text-xs text-zinc-500">
              Select which markets to trade. Green = enabled. Grey = coming soon.
            </p>

            {/* Market Grid */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr>
                    <th className="text-left py-1 px-2 text-zinc-500">Symbol</th>
                    {MARKET_TIMEFRAMES.map((tf) => (
                      <th key={tf.id} className="text-center py-1 px-2">
                        <SimpleTooltip content={tf.description}>
                          <span className={tf.active ? "text-zinc-700 dark:text-zinc-300" : "text-zinc-400"}>
                            {tf.label}
                          </span>
                        </SimpleTooltip>
                      </th>
                    ))}
                    <th className="text-right py-1 px-2 text-zinc-500">Vol/24h</th>
                  </tr>
                </thead>
                <tbody>
                  {MARKET_SYMBOLS.map(({ symbol, tier, volume, recommended }) => (
                    <tr key={symbol} className="border-t border-zinc-200 dark:border-zinc-700">
                      <td className="py-2 px-2">
                        <div className="flex items-center gap-1">
                          <span className="font-medium">{symbol}</span>
                          {recommended && (
                            <Badge className="text-[9px] bg-green-500/20 text-green-500">rec</Badge>
                          )}
                        </div>
                      </td>
                      {MARKET_TIMEFRAMES.map((tf) => (
                        <td key={tf.id} className="text-center py-2 px-2">
                          {tf.active ? (
                            <button
                              onClick={() => handleAssetToggle(symbol)}
                              className={`w-6 h-6 rounded flex items-center justify-center transition-colors ${
                                (localConfig.enabled_assets || []).includes(symbol)
                                  ? "bg-green-500 text-white"
                                  : "bg-zinc-200 dark:bg-zinc-700 text-zinc-400 hover:bg-zinc-300 dark:hover:bg-zinc-600"
                              }`}
                            >
                              {(localConfig.enabled_assets || []).includes(symbol) ? (
                                <Check className="w-3 h-3" />
                              ) : null}
                            </button>
                          ) : (
                            <SimpleTooltip content="Coming soon - not yet on Polymarket">
                              <div className="w-6 h-6 rounded bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center cursor-not-allowed">
                                <Clock className="w-3 h-3 text-zinc-400" />
                              </div>
                            </SimpleTooltip>
                          )}
                        </td>
                      ))}
                      <td className="text-right py-2 px-2 text-zinc-500 font-mono">
                        ${volume}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-[10px] text-zinc-500">
              Higher volume = tighter spreads. BTC/ETH recommended for best execution.
            </p>
          </div>

          {/* Section 2: When to Trade */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 border-b border-zinc-200 dark:border-zinc-700 pb-1">
              When to Trade
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-zinc-500">Min Edge %</span>
                  <InfoTooltip content="How much Polymarket must lag Binance before trading. Higher = fewer but safer trades." />
                </div>
                <input
                  type="number"
                  value={localConfig.min_edge_pct || 5}
                  onChange={(e) => setLocalConfig({ ...localConfig, min_edge_pct: parseFloat(e.target.value) })}
                  className="w-full px-3 py-2 rounded bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 text-sm"
                  step="0.5"
                  min="1"
                  max="20"
                />
                <p className="text-[10px] text-zinc-500 mt-1">5% = conservative, 3% = aggressive</p>
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-zinc-500">Cooldown (sec)</span>
                  <InfoTooltip content="Wait time between trades on same asset. Prevents overtrading on volatile swings." />
                </div>
                <input
                  type="number"
                  value={localConfig.cooldown_sec || 30}
                  onChange={(e) => setLocalConfig({ ...localConfig, cooldown_sec: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 rounded bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 text-sm"
                  step="10"
                  min="10"
                  max="300"
                />
              </div>
            </div>

            {/* Entry Window */}
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium">Entry Window</span>
                <InfoTooltip content="Only consider entries after this time into the 15-min window. Later = more confirmation but less time for position." />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-xs text-zinc-500">UP entries after</span>
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min="180"
                      max="720"
                      step="30"
                      value={localConfig.entry_time_up_sec || 450}
                      onChange={(e) => setLocalConfig({ ...localConfig, entry_time_up_sec: parseInt(e.target.value) })}
                      className="flex-1"
                    />
                    <span className="text-xs font-mono w-12">
                      {formatSecondsToTime(localConfig.entry_time_up_sec || 450)}
                    </span>
                  </div>
                </div>
                <div>
                  <span className="text-xs text-zinc-500">DOWN entries after</span>
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min="180"
                      max="720"
                      step="30"
                      value={localConfig.entry_time_down_sec || 450}
                      onChange={(e) => setLocalConfig({ ...localConfig, entry_time_down_sec: parseInt(e.target.value) })}
                      className="flex-1"
                    />
                    <span className="text-xs font-mono w-12">
                      {formatSecondsToTime(localConfig.entry_time_down_sec || 450)}
                    </span>
                  </div>
                </div>
              </div>
              <p className="text-[10px] text-zinc-500 mt-1">
                7m30s is default. Earlier entries give more time but less price confirmation.
              </p>
            </div>

            {/* Stop buffer */}
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-zinc-500">Stop entries before close</span>
                <InfoTooltip content="Stop entering new positions this many seconds before market closes to avoid being caught in final volatility." />
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  value={localConfig.min_time_remaining_sec || 120}
                  onChange={(e) => setLocalConfig({ ...localConfig, min_time_remaining_sec: parseInt(e.target.value) })}
                  className="w-24 px-3 py-2 rounded bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 text-sm"
                  step="30"
                  min="60"
                  max="300"
                />
                <span className="text-xs text-zinc-500">seconds</span>
              </div>
            </div>
          </div>

          {/* Section 3: Signal Checkpoints */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 border-b border-zinc-200 dark:border-zinc-700 pb-1">
              Signal Checkpoints
            </h3>
            <p className="text-xs text-zinc-500">
              Signals are only generated at these times into the 15-min window. Select which checkpoints to monitor.
            </p>

            {/* Checkpoint selection */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium">Signal Times</span>
                <InfoTooltip content="Select which checkpoints will generate signals. The active checkpoint (highlighted) is where trades execute." />
              </div>
              <div className="grid grid-cols-5 gap-2">
                {CHECKPOINT_OPTIONS.map(({ sec, label, description }) => {
                  const isEnabled = (localConfig.signal_checkpoints || []).includes(sec);
                  const isActive = localConfig.active_checkpoint === sec;

                  return (
                    <button
                      key={sec}
                      onClick={() => {
                        const current = localConfig.signal_checkpoints || [];
                        if (isEnabled) {
                          // Don't allow disabling the active checkpoint
                          if (isActive) return;
                          setLocalConfig({
                            ...localConfig,
                            signal_checkpoints: current.filter((s) => s !== sec),
                          });
                        } else {
                          setLocalConfig({
                            ...localConfig,
                            signal_checkpoints: [...current, sec].sort((a, b) => a - b),
                          });
                        }
                      }}
                      className={`relative px-2 py-2 rounded-lg text-xs font-medium transition-all ${
                        isEnabled
                          ? isActive
                            ? "bg-purple-500 text-white ring-2 ring-purple-300"
                            : "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                          : "bg-zinc-200 dark:bg-zinc-700 text-zinc-500"
                      }`}
                      title={description}
                    >
                      {label}
                      {isActive && (
                        <span className="absolute -top-1 -right-1 w-2 h-2 bg-green-500 rounded-full" />
                      )}
                    </button>
                  );
                })}
              </div>
              <p className="text-[10px] text-zinc-400">
                Click to enable/disable. Purple = active (trades execute here).
              </p>
            </div>

            {/* Active checkpoint selection */}
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium">Execute Trades At</span>
                <InfoTooltip content="Select which checkpoint will actually execute trades. Other enabled checkpoints only generate display signals." />
              </div>
              <div className="flex flex-wrap gap-2">
                {CHECKPOINT_OPTIONS.filter(({ sec }) =>
                  (localConfig.signal_checkpoints || []).includes(sec)
                ).map(({ sec, label, description }) => {
                  const isActive = localConfig.active_checkpoint === sec;

                  return (
                    <button
                      key={sec}
                      onClick={() => {
                        setLocalConfig({
                          ...localConfig,
                          active_checkpoint: sec,
                        });
                      }}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        isActive
                          ? "bg-purple-500 text-white"
                          : "bg-zinc-200 dark:bg-zinc-700 text-zinc-500 hover:bg-zinc-300 dark:hover:bg-zinc-600"
                      }`}
                      title={description}
                    >
                      {label}
                      {sec === 540 && <span className="ml-1 text-[9px] opacity-70">(Best EV)</span>}
                      {sec === 450 && <span className="ml-1 text-[9px] opacity-70">(Default)</span>}
                    </button>
                  );
                })}
              </div>
              <p className="text-[10px] text-zinc-400 mt-1">
                9m historically shows best expected value. 7m30s is the balanced default.
              </p>
            </div>
          </div>

          {/* Section 4: Signal Filters */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 border-b border-zinc-200 dark:border-zinc-700 pb-1">
              Signal Filters
            </h3>
            <p className="text-xs text-zinc-500">
              Require these confirmations to filter out false signals from random price noise.
            </p>
            <div className="space-y-2">
              <label className="flex items-start gap-2 cursor-pointer p-2 rounded bg-zinc-50 dark:bg-zinc-800/50">
                <input
                  type="checkbox"
                  checked={localConfig.require_volume_confirmation ?? true}
                  onChange={(e) => setLocalConfig({ ...localConfig, require_volume_confirmation: e.target.checked })}
                  className="rounded mt-0.5"
                />
                <div>
                  <span className="text-sm font-medium">Volume confirmation</span>
                  <p className="text-xs text-zinc-500">Buy volume must support direction (buying pressure for UP, selling for DOWN)</p>
                </div>
              </label>
              <label className="flex items-start gap-2 cursor-pointer p-2 rounded bg-zinc-50 dark:bg-zinc-800/50">
                <input
                  type="checkbox"
                  checked={localConfig.require_orderbook_confirmation ?? true}
                  onChange={(e) => setLocalConfig({ ...localConfig, require_orderbook_confirmation: e.target.checked })}
                  className="rounded mt-0.5"
                />
                <div>
                  <span className="text-sm font-medium">Order book confirmation</span>
                  <p className="text-xs text-zinc-500">Bid/ask imbalance must support direction (more bids for UP, more asks for DOWN)</p>
                </div>
              </label>
            </div>
          </div>

          {/* Section 5: Market Display */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 border-b border-zinc-200 dark:border-zinc-700 pb-1">
              Market Display
            </h3>
            <p className="text-xs text-zinc-500">
              Configure how market data is displayed. Chart updates every 5 seconds.
            </p>
            <div className="p-3 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-zinc-600 dark:text-zinc-400">Data update interval</span>
                <span className="font-mono text-zinc-500">5 sec (backend)</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-zinc-600 dark:text-zinc-400">Market refresh</span>
                <span className="font-mono text-zinc-500">2 sec (Polymarket API)</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-zinc-600 dark:text-zinc-400">Price precision</span>
                <span className="font-mono text-zinc-500">0.1%</span>
              </div>
              <p className="text-[10px] text-zinc-400 pt-2 border-t border-zinc-200 dark:border-zinc-700">
                Note: Polymarket odds update slower than Binance prices. This lag is the basis of the latency arbitrage strategy.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 p-4 border-t border-zinc-200 dark:border-zinc-700">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded bg-zinc-200 dark:bg-zinc-700 text-sm font-medium hover:bg-zinc-300 dark:hover:bg-zinc-600 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 rounded bg-purple-500 text-white text-sm font-medium hover:bg-purple-600 transition-colors"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}
