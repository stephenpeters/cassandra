"use client";

import { useState, useEffect } from "react";
import { X, Sliders, TrendingUp, TrendingDown, Activity } from "lucide-react";
import { InfoTooltip } from "@/components/ui/info-tooltip";
import type { TradingConfig } from "@/types";

interface StrategySettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: TradingConfig | null;
  onConfigUpdate: (config: Partial<TradingConfig>) => void;
}

interface IndicatorConfig {
  key: keyof TradingConfig;
  name: string;
  description: string;
  icon: "trend" | "activity";
}

const INDICATORS: IndicatorConfig[] = [
  {
    key: "use_edge",
    name: "Edge (Latency Gap)",
    description: "The price difference between Binance spot price movement and Polymarket odds. Example: If BTC rises 0.1% on Binance but Polymarket UP odds only show 52%, there's ~8% edge if fair value should be 60%.",
    icon: "trend",
  },
  {
    key: "use_volume_delta",
    name: "Volume Delta",
    description: "Net directional volume = Buy volume minus Sell volume. Positive = more buyers, confirms upward momentum. Example: +$50K means buyers outpacing sellers by $50K in the window.",
    icon: "activity",
  },
  {
    key: "use_orderbook",
    name: "Order Book Imbalance",
    description: "Ratio of bid orders vs ask orders near current price. +40% means 70% bids vs 30% asks - more buyers waiting. Shows where large orders are stacked.",
    icon: "activity",
  },
  {
    key: "use_vwap",
    name: "VWAP",
    description: "Volume Weighted Average Price - the average price weighted by volume traded. Price above VWAP = bullish (buyers paying up), below = bearish (sellers dumping).",
    icon: "trend",
  },
  {
    key: "use_rsi",
    name: "RSI",
    description: "Relative Strength Index (0-100). Below 30 = oversold (likely to bounce UP). Above 70 = overbought (likely to pull DOWN). Measures momentum exhaustion.",
    icon: "activity",
  },
  {
    key: "use_adx",
    name: "ADX",
    description: "Average Directional Index (0-100). Measures trend STRENGTH not direction. Above 25 = strong trend (momentum likely to continue). Below 20 = ranging/choppy.",
    icon: "activity",
  },
  {
    key: "use_supertrend",
    name: "Supertrend",
    description: "ATR-based trend indicator. Uses volatility bands to give clear UP/DOWN signal. When price closes above upper band = bullish trend, below lower band = bearish trend.",
    icon: "trend",
  },
];

export function StrategySettingsModal({
  isOpen,
  onClose,
  config,
  onConfigUpdate,
}: StrategySettingsModalProps) {
  const [localConfig, setLocalConfig] = useState<Partial<TradingConfig>>({});

  useEffect(() => {
    if (config) {
      setLocalConfig({
        // Tiered confirmation
        min_confirmations: config.min_confirmations ?? 2,
        partial_size_pct: config.partial_size_pct ?? 50,
        edge_mandatory: config.edge_mandatory ?? false,

        // Indicator toggles
        use_edge: config.use_edge ?? true,
        use_volume_delta: config.use_volume_delta ?? true,
        use_orderbook: config.use_orderbook ?? true,
        use_vwap: config.use_vwap ?? true,
        use_rsi: config.use_rsi ?? true,
        use_adx: config.use_adx ?? true,
        use_supertrend: config.use_supertrend ?? true,

        // Thresholds
        min_edge_pct: config.min_edge_pct ?? 5,
        min_volume_delta_usd: config.min_volume_delta_usd ?? 10000,
        min_orderbook_imbalance: config.min_orderbook_imbalance ?? 0.1,
        rsi_oversold: config.rsi_oversold ?? 30,
        rsi_overbought: config.rsi_overbought ?? 70,
        adx_trend_threshold: config.adx_trend_threshold ?? 25,
        supertrend_multiplier: config.supertrend_multiplier ?? 3.0,
      });
    }
  }, [config]);

  if (!isOpen) return null;

  const handleSave = () => {
    onConfigUpdate(localConfig);
    onClose();
  };

  const handleToggle = (key: keyof TradingConfig) => {
    setLocalConfig({ ...localConfig, [key]: !localConfig[key] });
  };

  // Count enabled indicators
  const enabledCount = INDICATORS.filter(
    (ind) => localConfig[ind.key] as boolean
  ).length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white dark:bg-zinc-900 rounded-lg shadow-xl border border-zinc-300 dark:border-zinc-700 w-full max-w-xl max-h-[85vh] overflow-y-auto mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-700">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-purple-500" />
            <h2 className="text-lg font-semibold">Strategy Settings</h2>
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
          <div className="p-3 bg-purple-500/10 rounded-lg border border-purple-500/20">
            <p className="text-xs text-purple-600 dark:text-purple-400">
              <strong>Tiered Confirmation:</strong> Instead of requiring ALL indicators,
              configure how many must agree before entering a trade. More confirmations = safer but fewer trades.
            </p>
          </div>

          {/* Section 1: Confirmation Requirements */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 border-b border-zinc-200 dark:border-zinc-700 pb-1">
              Confirmation Requirements
            </h3>

            <div className="grid grid-cols-2 gap-4">
              {/* Min Confirmations */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm font-medium">Min Confirmations</span>
                  <InfoTooltip content="Minimum indicators that must confirm the trade direction. 2 = aggressive, 3 = balanced, 4+ = conservative." />
                </div>
                <div className="flex gap-2">
                  {[2, 3, 4].map((n) => (
                    <button
                      key={n}
                      onClick={() => setLocalConfig({ ...localConfig, min_confirmations: n })}
                      className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                        localConfig.min_confirmations === n
                          ? "bg-purple-500 text-white"
                          : "bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-300 dark:hover:bg-zinc-600"
                      }`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
                <p className="text-[10px] text-zinc-500 mt-1">
                  {enabledCount} indicators enabled. Need {localConfig.min_confirmations} to trade.
                </p>
              </div>

              {/* Partial Size */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm font-medium">Partial Size %</span>
                  <InfoTooltip content="Position size when you have exactly min_confirmations. Full size when you have more." />
                </div>
                <div className="flex gap-2">
                  {[25, 50, 75].map((pct) => (
                    <button
                      key={pct}
                      onClick={() => setLocalConfig({ ...localConfig, partial_size_pct: pct })}
                      className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                        localConfig.partial_size_pct === pct
                          ? "bg-purple-500 text-white"
                          : "bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-300 dark:hover:bg-zinc-600"
                      }`}
                    >
                      {pct}%
                    </button>
                  ))}
                </div>
                <p className="text-[10px] text-zinc-500 mt-1">
                  Use reduced size when confidence is lower.
                </p>
              </div>
            </div>

            {/* Edge Mandatory */}
            <label className="flex items-center gap-3 p-3 rounded-lg bg-zinc-50 dark:bg-zinc-800/50 cursor-pointer">
              <input
                type="checkbox"
                checked={localConfig.edge_mandatory as boolean}
                onChange={(e) => setLocalConfig({ ...localConfig, edge_mandatory: e.target.checked })}
                className="rounded"
              />
              <div className="flex-1">
                <span className="text-sm font-medium">Require Edge for all trades</span>
                <p className="text-xs text-zinc-500">
                  If enabled, edge must always be present. Other indicators alone won't trigger trades.
                </p>
              </div>
            </label>
          </div>

          {/* Section 2: Indicator Toggles */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 border-b border-zinc-200 dark:border-zinc-700 pb-1">
              Active Indicators
            </h3>
            <p className="text-xs text-zinc-500">
              Enable/disable indicators. Disabled indicators won't count toward confirmations.
            </p>

            <div className="grid grid-cols-2 gap-2">
              {INDICATORS.map((indicator) => {
                const isEnabled = localConfig[indicator.key] as boolean;
                const Icon = indicator.icon === "trend" ? TrendingUp : Activity;

                return (
                  <button
                    key={indicator.key}
                    onClick={() => handleToggle(indicator.key)}
                    className={`flex items-start gap-2 p-2.5 rounded-lg text-left transition-all ${
                      isEnabled
                        ? "bg-purple-500/10 border border-purple-500/30"
                        : "bg-zinc-100 dark:bg-zinc-800 border border-transparent"
                    }`}
                  >
                    <Icon
                      className={`w-4 h-4 mt-0.5 ${
                        isEnabled ? "text-purple-500" : "text-zinc-400"
                      }`}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1">
                        <span
                          className={`text-xs font-medium truncate ${
                            isEnabled
                              ? "text-purple-600 dark:text-purple-400"
                              : "text-zinc-600 dark:text-zinc-400"
                          }`}
                        >
                          {indicator.name}
                        </span>
                        {isEnabled && (
                          <span className="w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                        )}
                      </div>
                      <p className="text-[10px] text-zinc-500 line-clamp-2">
                        {indicator.description}
                      </p>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 3: Threshold Settings */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 border-b border-zinc-200 dark:border-zinc-700 pb-1">
              Indicator Thresholds
            </h3>

            {/* Edge Threshold */}
            {localConfig.use_edge && (
              <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-purple-500" />
                    <span className="text-sm font-medium">Edge Threshold</span>
                    <InfoTooltip content="Minimum price gap between Binance movement and Polymarket odds to trigger a trade. Example: 5% edge means you need at least 5 percentage points of mispricing (e.g., fair value 60% but market shows 55%)." />
                  </div>
                  <span className="text-sm font-mono text-purple-500">
                    {localConfig.min_edge_pct}%
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="15"
                  step="0.5"
                  value={localConfig.min_edge_pct || 5}
                  onChange={(e) =>
                    setLocalConfig({ ...localConfig, min_edge_pct: parseFloat(e.target.value) })
                  }
                  className="w-full"
                />
                <p className="text-[10px] text-zinc-500">
                  Lower = more trades, higher = safer trades
                </p>
              </div>
            )}

            {/* Volume Delta Threshold */}
            {localConfig.use_volume_delta && (
              <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-blue-500" />
                    <span className="text-sm font-medium">Volume Delta</span>
                  </div>
                  <span className="text-sm font-mono text-blue-500">
                    ${((localConfig.min_volume_delta_usd || 10000) / 1000).toFixed(0)}K
                  </span>
                </div>
                <input
                  type="range"
                  min="1000"
                  max="50000"
                  step="1000"
                  value={localConfig.min_volume_delta_usd || 10000}
                  onChange={(e) =>
                    setLocalConfig({ ...localConfig, min_volume_delta_usd: parseInt(e.target.value) })
                  }
                  className="w-full"
                />
                <p className="text-[10px] text-zinc-500">
                  Minimum net directional volume in USD
                </p>
              </div>
            )}

            {/* Orderbook Imbalance */}
            {localConfig.use_orderbook && (
              <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-green-500" />
                    <span className="text-sm font-medium">Orderbook Imbalance</span>
                  </div>
                  <span className="text-sm font-mono text-green-500">
                    {((localConfig.min_orderbook_imbalance || 0.1) * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.5"
                  step="0.05"
                  value={localConfig.min_orderbook_imbalance || 0.1}
                  onChange={(e) =>
                    setLocalConfig({ ...localConfig, min_orderbook_imbalance: parseFloat(e.target.value) })
                  }
                  className="w-full"
                />
                <p className="text-[10px] text-zinc-500">
                  Bid/ask ratio difference required
                </p>
              </div>
            )}

            {/* RSI Thresholds */}
            {localConfig.use_rsi && (
              <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg space-y-2">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-orange-500" />
                  <span className="text-sm font-medium">RSI Levels</span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="text-green-500">Oversold</span>
                      <span className="font-mono">{localConfig.rsi_oversold}</span>
                    </div>
                    <input
                      type="range"
                      min="10"
                      max="40"
                      step="5"
                      value={localConfig.rsi_oversold || 30}
                      onChange={(e) =>
                        setLocalConfig({ ...localConfig, rsi_oversold: parseInt(e.target.value) })
                      }
                      className="w-full"
                    />
                  </div>
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="text-red-500">Overbought</span>
                      <span className="font-mono">{localConfig.rsi_overbought}</span>
                    </div>
                    <input
                      type="range"
                      min="60"
                      max="90"
                      step="5"
                      value={localConfig.rsi_overbought || 70}
                      onChange={(e) =>
                        setLocalConfig({ ...localConfig, rsi_overbought: parseInt(e.target.value) })
                      }
                      className="w-full"
                    />
                  </div>
                </div>
                <p className="text-[10px] text-zinc-500">
                  RSI &lt; oversold = bullish, RSI &gt; overbought = bearish
                </p>
              </div>
            )}

            {/* ADX Threshold */}
            {localConfig.use_adx && (
              <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-cyan-500" />
                    <span className="text-sm font-medium">ADX Trend Strength</span>
                  </div>
                  <span className="text-sm font-mono text-cyan-500">
                    {localConfig.adx_trend_threshold}
                  </span>
                </div>
                <input
                  type="range"
                  min="15"
                  max="40"
                  step="5"
                  value={localConfig.adx_trend_threshold || 25}
                  onChange={(e) =>
                    setLocalConfig({ ...localConfig, adx_trend_threshold: parseInt(e.target.value) })
                  }
                  className="w-full"
                />
                <p className="text-[10px] text-zinc-500">
                  ADX above threshold = strong trend detected
                </p>
              </div>
            )}

            {/* Supertrend Multiplier */}
            {localConfig.use_supertrend && (
              <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-pink-500" />
                    <span className="text-sm font-medium">Supertrend ATR Multiplier</span>
                  </div>
                  <span className="text-sm font-mono text-pink-500">
                    {localConfig.supertrend_multiplier}x
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="0.5"
                  value={localConfig.supertrend_multiplier || 3}
                  onChange={(e) =>
                    setLocalConfig({ ...localConfig, supertrend_multiplier: parseFloat(e.target.value) })
                  }
                  className="w-full"
                />
                <p className="text-[10px] text-zinc-500">
                  Higher = less sensitive, fewer signals
                </p>
              </div>
            )}
          </div>

          {/* Current Configuration Summary */}
          <div className="p-3 bg-zinc-100 dark:bg-zinc-800 rounded-lg">
            <h4 className="text-xs font-semibold text-zinc-600 dark:text-zinc-400 mb-2">
              Current Configuration
            </h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex justify-between">
                <span className="text-zinc-500">Active indicators:</span>
                <span className="font-mono text-purple-500">{enabledCount}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Min confirmations:</span>
                <span className="font-mono text-purple-500">{localConfig.min_confirmations}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Partial position:</span>
                <span className="font-mono text-purple-500">{localConfig.partial_size_pct}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Edge mandatory:</span>
                <span className={`font-mono ${localConfig.edge_mandatory ? "text-green-500" : "text-zinc-500"}`}>
                  {localConfig.edge_mandatory ? "Yes" : "No"}
                </span>
              </div>
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
