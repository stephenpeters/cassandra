"use client";

import { useState, useCallback, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { InfoTooltip } from "@/components/ui/info-tooltip";
import { Settings, Save, RotateCcw, AlertTriangle } from "lucide-react";
import type { PaperTradingConfig } from "@/types";

// Convert seconds to mm:ss display
function formatSecondsToTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m${secs > 0 ? ` ${secs}s` : ""}`;
}

interface TradingSettingsPanelProps {
  config: PaperTradingConfig | null;
  onConfigUpdate: (config: Partial<PaperTradingConfig>) => Promise<void>;
}

const AVAILABLE_ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"];

export function TradingSettingsPanel({
  config,
  onConfigUpdate,
}: TradingSettingsPanelProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [localConfig, setLocalConfig] = useState<Partial<PaperTradingConfig>>({});
  const [hasChanges, setHasChanges] = useState(false);

  // Initialize local config when config prop changes
  useEffect(() => {
    if (config) {
      setLocalConfig({
        min_edge_pct: config.min_edge_pct ?? 5,
        min_time_remaining_sec: config.min_time_remaining_sec ?? 120,
        cooldown_sec: config.cooldown_sec ?? 30,
        entry_time_up_sec: config.entry_time_up_sec ?? 450,
        entry_time_down_sec: config.entry_time_down_sec ?? 450,
        max_position_pct: config.max_position_pct ?? 2,
        max_position_usd: config.max_position_usd ?? 5000,
        daily_loss_limit_pct: config.daily_loss_limit_pct ?? 10,
        enabled_assets: config.enabled_assets ?? ["BTC"],
        require_volume_confirmation: config.require_volume_confirmation ?? true,
        require_orderbook_confirmation: config.require_orderbook_confirmation ?? true,
        min_volume_delta_usd: config.min_volume_delta_usd ?? 10000,
        min_orderbook_imbalance: config.min_orderbook_imbalance ?? 0.1,
      });
    }
  }, [config]);

  const updateLocalConfig = useCallback(
    (key: keyof PaperTradingConfig, value: unknown) => {
      setLocalConfig((prev) => ({ ...prev, [key]: value }));
      setHasChanges(true);
    },
    []
  );

  const toggleAsset = useCallback(
    (asset: string) => {
      setLocalConfig((prev) => {
        const current = prev.enabled_assets || [];
        const newAssets = current.includes(asset)
          ? current.filter((a) => a !== asset)
          : [...current, asset];
        return { ...prev, enabled_assets: newAssets.length > 0 ? newAssets : ["BTC"] };
      });
      setHasChanges(true);
    },
    []
  );

  const handleSave = useCallback(async () => {
    setIsSaving(true);
    try {
      await onConfigUpdate(localConfig);
      setHasChanges(false);
      setIsEditing(false);
    } catch (e) {
      console.error("Failed to save config:", e);
    } finally {
      setIsSaving(false);
    }
  }, [localConfig, onConfigUpdate]);

  const handleReset = useCallback(() => {
    if (config) {
      setLocalConfig({
        min_edge_pct: config.min_edge_pct ?? 5,
        min_time_remaining_sec: config.min_time_remaining_sec ?? 120,
        cooldown_sec: config.cooldown_sec ?? 30,
        entry_time_up_sec: config.entry_time_up_sec ?? 450,
        entry_time_down_sec: config.entry_time_down_sec ?? 450,
        max_position_pct: config.max_position_pct ?? 2,
        max_position_usd: config.max_position_usd ?? 5000,
        daily_loss_limit_pct: config.daily_loss_limit_pct ?? 10,
        enabled_assets: config.enabled_assets ?? ["BTC"],
        require_volume_confirmation: config.require_volume_confirmation ?? true,
        require_orderbook_confirmation: config.require_orderbook_confirmation ?? true,
        min_volume_delta_usd: config.min_volume_delta_usd ?? 10000,
        min_orderbook_imbalance: config.min_orderbook_imbalance ?? 0.1,
      });
      setHasChanges(false);
    }
  }, [config]);

  if (!config) {
    return (
      <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
        <CardContent className="p-6 text-center text-zinc-500">
          Loading settings...
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-zinc-500" />
            <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
              Trading Settings
            </CardTitle>
          </div>
          <div className="flex items-center gap-2">
            {hasChanges && (
              <Badge className="bg-yellow-500/20 text-yellow-400 text-xs">
                Unsaved
              </Badge>
            )}
            {!isEditing ? (
              <Button
                size="sm"
                variant="outline"
                onClick={() => setIsEditing(true)}
                className="text-xs"
              >
                Edit
              </Button>
            ) : (
              <div className="flex gap-1">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleReset}
                  disabled={isSaving}
                  className="text-xs"
                >
                  <RotateCcw className="w-3 h-3 mr-1" />
                  Reset
                </Button>
                <Button
                  size="sm"
                  onClick={handleSave}
                  disabled={isSaving || !hasChanges}
                  className="text-xs bg-blue-600 hover:bg-blue-700"
                >
                  <Save className="w-3 h-3 mr-1" />
                  {isSaving ? "Saving..." : "Save"}
                </Button>
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Enabled Assets */}
        <div>
          <label className="text-xs text-zinc-500 uppercase tracking-wider block mb-2">
            Enabled Assets
          </label>
          <div className="flex gap-2 flex-wrap">
            {AVAILABLE_ASSETS.map((asset) => {
              const isEnabled = localConfig.enabled_assets?.includes(asset);
              return (
                <button
                  key={asset}
                  onClick={() => isEditing && toggleAsset(asset)}
                  disabled={!isEditing}
                  className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                    isEnabled
                      ? "bg-blue-600 text-white"
                      : "bg-zinc-200 dark:bg-zinc-700 text-zinc-500 dark:text-zinc-400"
                  } ${isEditing ? "cursor-pointer hover:opacity-80" : "cursor-default"}`}
                >
                  {asset}
                </button>
              );
            })}
          </div>
          <p className="text-xs text-zinc-400 mt-1">
            Changes apply immediately without restart
          </p>
        </div>

        {/* Entry Timing */}
        <div className="border-t border-zinc-200 dark:border-zinc-700 pt-4">
          <h4 className="flex items-center gap-1 text-xs text-zinc-500 uppercase tracking-wider mb-3">
            Entry Timing
            <InfoTooltip content="When to consider entering trades within the 15-minute market window. Earlier = more risk but more time for price to move. Later = more certainty but less time." />
          </h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="flex items-center gap-1 text-xs text-zinc-400 mb-1">
                <span className="text-green-500">UP</span> Entry Time
                <InfoTooltip content="Seconds into the 15-min window before considering UP trades. Example: 450 = wait until 7m30s. Earlier entries are riskier but have more upside." />
              </label>
              <input
                type="number"
                value={localConfig.entry_time_up_sec ?? 450}
                onChange={(e) =>
                  updateLocalConfig("entry_time_up_sec", parseInt(e.target.value))
                }
                disabled={!isEditing}
                step="30"
                min="60"
                max="780"
                className="w-full px-2 py-1.5 text-sm bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded disabled:opacity-50"
              />
              <p className="text-[10px] text-zinc-400 mt-0.5">
                {formatSecondsToTime(localConfig.entry_time_up_sec ?? 450)} into window
              </p>
            </div>
            <div>
              <label className="flex items-center gap-1 text-xs text-zinc-400 mb-1">
                <span className="text-red-500">DOWN</span> Entry Time
                <InfoTooltip content="Seconds into the 15-min window before considering DOWN trades. Can be different from UP if you want asymmetric entries." />
              </label>
              <input
                type="number"
                value={localConfig.entry_time_down_sec ?? 450}
                onChange={(e) =>
                  updateLocalConfig("entry_time_down_sec", parseInt(e.target.value))
                }
                disabled={!isEditing}
                step="30"
                min="60"
                max="780"
                className="w-full px-2 py-1.5 text-sm bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded disabled:opacity-50"
              />
              <p className="text-[10px] text-zinc-400 mt-0.5">
                {formatSecondsToTime(localConfig.entry_time_down_sec ?? 450)} into window
              </p>
            </div>
          </div>
        </div>

        {/* Latency Arbitrage Settings */}
        <div className="border-t border-zinc-200 dark:border-zinc-700 pt-4">
          <h4 className="flex items-center gap-1 text-xs text-zinc-500 uppercase tracking-wider mb-3">
            Latency Arbitrage
            <InfoTooltip content="Settings for exploiting the delay between Binance price moves and Polymarket odds adjustments. The 'edge' is the difference between calculated fair value and market price." />
          </h4>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="flex items-center gap-1 text-xs text-zinc-400 mb-1">
                Min Edge %
                <InfoTooltip content="Minimum edge percentage required to trigger a trade. Higher = fewer but higher-quality trades. Example: 5% means fair value must be 5% above market price." />
              </label>
              <input
                type="number"
                value={localConfig.min_edge_pct ?? 5}
                onChange={(e) =>
                  updateLocalConfig("min_edge_pct", parseFloat(e.target.value))
                }
                disabled={!isEditing}
                step="0.5"
                min="1"
                max="20"
                className="w-full px-2 py-1.5 text-sm bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded disabled:opacity-50"
              />
              <p className="text-[10px] text-zinc-400 mt-0.5">
                Trigger threshold
              </p>
            </div>
            <div>
              <label className="flex items-center gap-1 text-xs text-zinc-400 mb-1">
                Min Time (sec)
                <InfoTooltip content="Stop trading when this many seconds remain in the window. Avoids entering too close to resolution when odds are already priced in." />
              </label>
              <input
                type="number"
                value={localConfig.min_time_remaining_sec ?? 120}
                onChange={(e) =>
                  updateLocalConfig(
                    "min_time_remaining_sec",
                    parseInt(e.target.value)
                  )
                }
                disabled={!isEditing}
                step="30"
                min="30"
                max="600"
                className="w-full px-2 py-1.5 text-sm bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded disabled:opacity-50"
              />
              <p className="text-[10px] text-zinc-400 mt-0.5">
                No trades after
              </p>
            </div>
            <div>
              <label className="flex items-center gap-1 text-xs text-zinc-400 mb-1">
                Cooldown (sec)
                <InfoTooltip content="Minimum seconds between trades on the same symbol. Prevents overtrading on volatile moves." />
              </label>
              <input
                type="number"
                value={localConfig.cooldown_sec ?? 30}
                onChange={(e) =>
                  updateLocalConfig("cooldown_sec", parseInt(e.target.value))
                }
                disabled={!isEditing}
                step="10"
                min="10"
                max="300"
                className="w-full px-2 py-1.5 text-sm bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded disabled:opacity-50"
              />
              <p className="text-[10px] text-zinc-400 mt-0.5">
                Between trades
              </p>
            </div>
          </div>
        </div>

        {/* Confirmation Requirements */}
        <div className="border-t border-zinc-200 dark:border-zinc-700 pt-4">
          <h4 className="flex items-center gap-1 text-xs text-zinc-500 uppercase tracking-wider mb-3">
            Signal Confirmation
            <InfoTooltip content="Additional requirements before executing a trade. Volume and order book confirmations ensure momentum aligns with the trade direction." />
          </h4>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <span className="flex items-center gap-1 text-sm text-zinc-700 dark:text-zinc-300">
                  Require Volume Confirmation
                  <InfoTooltip content="Volume delta must support the trade direction. If buying UP, net buying volume must exceed the threshold. Filters out low-conviction signals." />
                </span>
                <p className="text-[10px] text-zinc-400">
                  Min delta: ${(localConfig.min_volume_delta_usd ?? 10000).toLocaleString()}
                </p>
              </div>
              <Switch
                checked={localConfig.require_volume_confirmation ?? true}
                onCheckedChange={(v) =>
                  updateLocalConfig("require_volume_confirmation", v)
                }
                disabled={!isEditing}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <span className="flex items-center gap-1 text-sm text-zinc-700 dark:text-zinc-300">
                  Require Order Book Confirmation
                  <InfoTooltip content="Order book imbalance must support the trade direction. If buying UP, there must be more buyers than sellers near current price." />
                </span>
                <p className="text-[10px] text-zinc-400">
                  Min imbalance: {((localConfig.min_orderbook_imbalance ?? 0.1) * 100).toFixed(0)}%
                </p>
              </div>
              <Switch
                checked={localConfig.require_orderbook_confirmation ?? true}
                onCheckedChange={(v) =>
                  updateLocalConfig("require_orderbook_confirmation", v)
                }
                disabled={!isEditing}
              />
            </div>
          </div>
        </div>

        {/* Risk Management */}
        <div className="border-t border-zinc-200 dark:border-zinc-700 pt-4">
          <h4 className="flex items-center gap-1 text-xs text-zinc-500 uppercase tracking-wider mb-3">
            Risk Management
            <InfoTooltip content="Position sizing and loss limits to protect capital. These limits apply to both paper and live trading." />
          </h4>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="flex items-center gap-1 text-xs text-zinc-400 mb-1">
                Max Position %
                <InfoTooltip content="Maximum percentage of account balance to risk on a single position. Example: 2% of $10,000 = $200 max per trade." />
              </label>
              <input
                type="number"
                value={localConfig.max_position_pct ?? 2}
                onChange={(e) =>
                  updateLocalConfig(
                    "max_position_pct",
                    parseFloat(e.target.value)
                  )
                }
                disabled={!isEditing}
                step="0.5"
                min="0.5"
                max="10"
                className="w-full px-2 py-1.5 text-sm bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded disabled:opacity-50"
              />
              <p className="text-[10px] text-zinc-400 mt-0.5">
                Of account
              </p>
            </div>
            <div>
              <label className="flex items-center gap-1 text-xs text-zinc-400 mb-1">
                Max Position $
                <InfoTooltip content="Hard dollar cap per position, regardless of account size. Limited by Polymarket liquidity - large orders cause slippage." />
              </label>
              <input
                type="number"
                value={localConfig.max_position_usd ?? 5000}
                onChange={(e) =>
                  updateLocalConfig(
                    "max_position_usd",
                    parseInt(e.target.value)
                  )
                }
                disabled={!isEditing}
                step="500"
                min="100"
                max="50000"
                className="w-full px-2 py-1.5 text-sm bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded disabled:opacity-50"
              />
              <p className="text-[10px] text-zinc-400 mt-0.5">
                Hard cap
              </p>
            </div>
            <div>
              <label className="flex items-center gap-1 text-xs text-zinc-400 mb-1">
                Daily Loss Limit %
                <InfoTooltip content="If daily losses exceed this percentage of starting balance, all trading halts until next day. Prevents catastrophic drawdowns." />
              </label>
              <input
                type="number"
                value={localConfig.daily_loss_limit_pct ?? 10}
                onChange={(e) =>
                  updateLocalConfig(
                    "daily_loss_limit_pct",
                    parseFloat(e.target.value)
                  )
                }
                disabled={!isEditing}
                step="1"
                min="1"
                max="50"
                className="w-full px-2 py-1.5 text-sm bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded disabled:opacity-50"
              />
              <p className="text-[10px] text-zinc-400 mt-0.5">
                Halts trading
              </p>
            </div>
          </div>
        </div>

        {/* Warning */}
        {isEditing && (
          <div className="flex items-start gap-2 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
            <AlertTriangle className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
            <p className="text-xs text-yellow-600 dark:text-yellow-400">
              Changes take effect on the next market window. Current positions are not affected.
              Settings are persisted to file and survive server restarts.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
