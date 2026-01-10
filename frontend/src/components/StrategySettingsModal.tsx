"use client";

import { useState, useEffect, useCallback } from "react";
import { X, Sliders, Target, TrendingDown, Clock, Users, Zap, AlertTriangle } from "lucide-react";
import { InfoTooltip } from "@/components/ui/info-tooltip";
import type { TradingConfig, LiveTradingStatus } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type TabType = "sniper" | "dip_arb" | "latency_arb" | "copy_trading" | "latency_gap" | "live_tools";

interface StrategyConfig {
  enabled: boolean;
  markets: string[];
  settings: Record<string, unknown>;
}

interface StrategySettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: TradingConfig | null;
  onConfigUpdate: (config: Partial<TradingConfig>) => void;
  liveStatus?: LiveTradingStatus | null;
  onTestOrder?: (symbol: string, side: "UP" | "DOWN", amount: number, apiKey: string) => Promise<{ success: boolean; error?: string; order?: unknown }>;
  onSetAllowances?: (apiKey: string) => Promise<{ success: boolean; error?: string }>;
}

const DEFAULT_CONFIGS: Record<string, StrategyConfig> = {
  sniper: { enabled: true, markets: ["BTC"], settings: { min_price: 0.75, max_price: 0.98, min_elapsed_sec: 600, min_ev_pct: 3.0 } },
  dip_arb: { enabled: false, markets: ["BTC"], settings: { dip_threshold: 0.15, window_minutes: 2, sum_target: 0.95 } },
  latency_arb: { enabled: false, markets: ["BTC"], settings: { min_move_pct: 0.3, take_profit_pct: 6.0, max_entry_price: 0.65 } },
  copy_trading: { enabled: false, markets: ["BTC"], settings: { position_multiplier: 0.5, max_position_usd: 100 } },
  latency_gap: { enabled: true, markets: ["BTC", "ETH"], settings: { min_edge_pct: 5.0, min_confidence: 0.6 } },
};

export function StrategySettingsModal({
  isOpen,
  onClose,
  config,
  onConfigUpdate,
  liveStatus,
  onTestOrder,
  onSetAllowances,
}: StrategySettingsModalProps) {
  const [activeTab, setActiveTab] = useState<TabType>("sniper");
  const [localConfig, setLocalConfig] = useState<Partial<TradingConfig>>({});
  const [saving, setSaving] = useState(false);

  // Strategy configs
  const [sniperConfig, setSniperConfig] = useState<StrategyConfig>(DEFAULT_CONFIGS.sniper);
  const [dipArbConfig, setDipArbConfig] = useState<StrategyConfig>(DEFAULT_CONFIGS.dip_arb);
  const [latencyArbConfig, setLatencyArbConfig] = useState<StrategyConfig>(DEFAULT_CONFIGS.latency_arb);
  const [copyTradingConfig, setCopyTradingConfig] = useState<StrategyConfig>(DEFAULT_CONFIGS.copy_trading);
  const [latencyGapConfig, setLatencyGapConfig] = useState<StrategyConfig>(DEFAULT_CONFIGS.latency_gap);

  // Manual order state
  const [testSymbol, setTestSymbol] = useState<"BTC" | "ETH" | "SOL">("BTC");
  const [testSide, setTestSide] = useState<"UP" | "DOWN">("UP");
  const [testAmount, setTestAmount] = useState(5);
  const [testOrderLoading, setTestOrderLoading] = useState(false);
  const [testOrderResult, setTestOrderResult] = useState<{ success: boolean; error?: string } | null>(null);
  const [allowancesLoading, setAllowancesLoading] = useState(false);
  const [allowancesResult, setAllowancesResult] = useState<{ success: boolean; error?: string } | null>(null);
  const [apiKey, setApiKey] = useState("");

  // Load API key
  useEffect(() => {
    const savedKey = localStorage.getItem("predmkt_api_key");
    if (savedKey) setApiKey(savedKey);
  }, [isOpen]);

  // Load strategies from API
  const loadStrategies = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/strategies`);
      if (res.ok) {
        const data = await res.json();
        const strategies = data.strategies || {};

        if (strategies.sniper) setSniperConfig({ enabled: strategies.sniper.enabled, markets: strategies.sniper.markets, settings: strategies.sniper.settings });
        if (strategies.dip_arb) setDipArbConfig({ enabled: strategies.dip_arb.enabled, markets: strategies.dip_arb.markets, settings: strategies.dip_arb.settings });
        if (strategies.latency_arb) setLatencyArbConfig({ enabled: strategies.latency_arb.enabled, markets: strategies.latency_arb.markets, settings: strategies.latency_arb.settings });
        if (strategies.copy_trading) setCopyTradingConfig({ enabled: strategies.copy_trading.enabled, markets: strategies.copy_trading.markets, settings: strategies.copy_trading.settings });
        if (strategies.latency_gap) setLatencyGapConfig({ enabled: strategies.latency_gap.enabled, markets: strategies.latency_gap.markets, settings: strategies.latency_gap.settings });
      }
    } catch (e) {
      console.error("Failed to load strategies:", e);
    }
  }, []);

  useEffect(() => {
    if (isOpen) loadStrategies();
  }, [isOpen, loadStrategies]);

  useEffect(() => {
    if (config) setLocalConfig(config);
  }, [config]);

  if (!isOpen) return null;

  // Save strategy to backend
  const saveStrategy = async (name: string, cfg: StrategyConfig) => {
    const key = apiKey || "paper-mode";
    await fetch(`${API_URL}/api/strategies/${name}/enable?enabled=${cfg.enabled}`, { method: "POST", headers: { "X-API-Key": key } });
    await fetch(`${API_URL}/api/strategies/${name}/markets`, { method: "POST", headers: { "Content-Type": "application/json", "X-API-Key": key }, body: JSON.stringify(cfg.markets) });
    if (Object.keys(cfg.settings).length > 0) {
      await fetch(`${API_URL}/api/strategies/${name}/settings`, { method: "POST", headers: { "Content-Type": "application/json", "X-API-Key": key }, body: JSON.stringify(cfg.settings) });
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveStrategy("sniper", sniperConfig);
      await saveStrategy("dip_arb", dipArbConfig);
      await saveStrategy("latency_arb", latencyArbConfig);
      await saveStrategy("copy_trading", copyTradingConfig);
      await saveStrategy("latency_gap", latencyGapConfig);
      onConfigUpdate(localConfig);
      onClose();
    } catch (error) {
      console.error("[Save] Failed:", error);
      alert(`Save failed: ${error}`);
    } finally {
      setSaving(false);
    }
  };

  const handleTestOrder = async () => {
    if (!onTestOrder || !apiKey) return;
    setTestOrderLoading(true);
    setTestOrderResult(null);
    try {
      const result = await onTestOrder(testSymbol, testSide, testAmount, apiKey);
      setTestOrderResult(result);
      if (result.success) setTimeout(() => setTestOrderResult(null), 5000);
    } catch {
      setTestOrderResult({ success: false, error: "Unexpected error" });
    } finally {
      setTestOrderLoading(false);
    }
  };

  const handleSetAllowances = async () => {
    if (!onSetAllowances || !apiKey) return;
    setAllowancesLoading(true);
    setAllowancesResult(null);
    try {
      const result = await onSetAllowances(apiKey);
      setAllowancesResult(result);
      if (result.success) setTimeout(() => setAllowancesResult(null), 5000);
    } catch {
      setAllowancesResult({ success: false, error: "Unexpected error" });
    } finally {
      setAllowancesLoading(false);
    }
  };

  const isLiveMode = liveStatus?.mode === "live";

  const tabs: { id: TabType; label: string; icon: React.ReactNode; color: string }[] = [
    { id: "sniper", label: "Sniper", icon: <Target className="w-4 h-4" />, color: "amber" },
    { id: "dip_arb", label: "DipArb", icon: <TrendingDown className="w-4 h-4" />, color: "emerald" },
    { id: "latency_arb", label: "LatencyArb", icon: <Clock className="w-4 h-4" />, color: "cyan" },
    { id: "copy_trading", label: "CopyTrade", icon: <Users className="w-4 h-4" />, color: "pink" },
    { id: "latency_gap", label: "LatencyGap", icon: <Zap className="w-4 h-4" />, color: "purple" },
    ...(isLiveMode ? [{ id: "live_tools" as TabType, label: "Live", icon: <AlertTriangle className="w-4 h-4" />, color: "red" }] : []),
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-white dark:bg-zinc-900 rounded-lg shadow-xl border border-zinc-300 dark:border-zinc-700 w-full max-w-2xl max-h-[85vh] overflow-hidden mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-700">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-purple-500" />
            <h2 className="text-lg font-semibold">Strategy Settings</h2>
          </div>
          <button onClick={onClose} className="p-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800"><X className="w-5 h-5" /></button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-zinc-200 dark:border-zinc-700 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium whitespace-nowrap transition-colors ${
                activeTab === tab.id
                  ? `text-${tab.color}-500 border-b-2 border-${tab.color}-500 bg-${tab.color}-500/10`
                  : "text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto max-h-[calc(85vh-180px)]">
          {/* Sniper Tab */}
          {activeTab === "sniper" && (
            <div className="space-y-4">
              <div className="p-3 bg-amber-500/10 rounded-lg border border-amber-500/20">
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  <strong>Sniper Strategy:</strong> Buy high-probability outcomes (75c-98c) after 10 minutes when EV &gt; 3%.
                </p>
              </div>
              <label className="flex items-center gap-3 p-3 rounded-lg bg-zinc-50 dark:bg-zinc-800/50">
                <input type="checkbox" checked={sniperConfig.enabled} onChange={(e) => setSniperConfig({ ...sniperConfig, enabled: e.target.checked })} className="rounded" />
                <span className="text-sm font-medium">Enable Sniper Strategy</span>
              </label>
              <div className="space-y-3">
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Min Price</span><span className="font-mono text-amber-500">{String(sniperConfig.settings.min_price)}c</span></div>
                  <input type="range" min="0.5" max="0.9" step="0.05" value={Number(sniperConfig.settings.min_price)} onChange={(e) => setSniperConfig({ ...sniperConfig, settings: { ...sniperConfig.settings, min_price: parseFloat(e.target.value) } })} className="w-full" />
                </div>
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Max Price</span><span className="font-mono text-amber-500">{String(sniperConfig.settings.max_price)}c</span></div>
                  <input type="range" min="0.9" max="0.99" step="0.01" value={Number(sniperConfig.settings.max_price)} onChange={(e) => setSniperConfig({ ...sniperConfig, settings: { ...sniperConfig.settings, max_price: parseFloat(e.target.value) } })} className="w-full" />
                </div>
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Min Elapsed (sec)</span><span className="font-mono text-amber-500">{String(sniperConfig.settings.min_elapsed_sec)}s</span></div>
                  <input type="range" min="300" max="840" step="60" value={Number(sniperConfig.settings.min_elapsed_sec)} onChange={(e) => setSniperConfig({ ...sniperConfig, settings: { ...sniperConfig.settings, min_elapsed_sec: parseInt(e.target.value) } })} className="w-full" />
                </div>
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Min EV %</span><span className="font-mono text-amber-500">{String(sniperConfig.settings.min_ev_pct)}%</span></div>
                  <input type="range" min="1" max="10" step="0.5" value={Number(sniperConfig.settings.min_ev_pct)} onChange={(e) => setSniperConfig({ ...sniperConfig, settings: { ...sniperConfig.settings, min_ev_pct: parseFloat(e.target.value) } })} className="w-full" />
                </div>
              </div>
            </div>
          )}

          {/* DipArb Tab */}
          {activeTab === "dip_arb" && (
            <div className="space-y-4">
              <div className="p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                <p className="text-xs text-emerald-600 dark:text-emerald-400">
                  <strong>DipArb Strategy:</strong> Two-leg flash crash arbitrage. Buy dip, then hedge opposite when leg1_price + opposite_ask ≤ sum_target.
                </p>
              </div>
              <label className="flex items-center gap-3 p-3 rounded-lg bg-zinc-50 dark:bg-zinc-800/50">
                <input type="checkbox" checked={dipArbConfig.enabled} onChange={(e) => setDipArbConfig({ ...dipArbConfig, enabled: e.target.checked })} className="rounded" />
                <span className="text-sm font-medium">Enable DipArb Strategy</span>
              </label>
              <div className="space-y-3">
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Dip Threshold</span><span className="font-mono text-emerald-500">{(Number(dipArbConfig.settings.dip_threshold) * 100).toFixed(0)}%</span></div>
                  <input type="range" min="0.05" max="0.3" step="0.01" value={Number(dipArbConfig.settings.dip_threshold)} onChange={(e) => setDipArbConfig({ ...dipArbConfig, settings: { ...dipArbConfig.settings, dip_threshold: parseFloat(e.target.value) } })} className="w-full" />
                </div>
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Window Minutes</span><span className="font-mono text-emerald-500">{String(dipArbConfig.settings.window_minutes)}m</span></div>
                  <input type="range" min="1" max="10" step="1" value={Number(dipArbConfig.settings.window_minutes)} onChange={(e) => setDipArbConfig({ ...dipArbConfig, settings: { ...dipArbConfig.settings, window_minutes: parseInt(e.target.value) } })} className="w-full" />
                </div>
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Sum Target</span><span className="font-mono text-emerald-500">{String(dipArbConfig.settings.sum_target)}</span></div>
                  <input type="range" min="0.8" max="1.0" step="0.01" value={Number(dipArbConfig.settings.sum_target)} onChange={(e) => setDipArbConfig({ ...dipArbConfig, settings: { ...dipArbConfig.settings, sum_target: parseFloat(e.target.value) } })} className="w-full" />
                </div>
              </div>
            </div>
          )}

          {/* LatencyArb Tab */}
          {activeTab === "latency_arb" && (
            <div className="space-y-4">
              <div className="p-3 bg-cyan-500/10 rounded-lg border border-cyan-500/20">
                <p className="text-xs text-cyan-600 dark:text-cyan-400">
                  <strong>LatencyArb Strategy:</strong> Exploit 30-second Binance→Polymarket lag. Buy when Binance moves, take profit at 6%.
                </p>
              </div>
              <label className="flex items-center gap-3 p-3 rounded-lg bg-zinc-50 dark:bg-zinc-800/50">
                <input type="checkbox" checked={latencyArbConfig.enabled} onChange={(e) => setLatencyArbConfig({ ...latencyArbConfig, enabled: e.target.checked })} className="rounded" />
                <span className="text-sm font-medium">Enable LatencyArb Strategy</span>
              </label>
              <div className="space-y-3">
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Min Binance Move %</span><span className="font-mono text-cyan-500">{String(latencyArbConfig.settings.min_move_pct)}%</span></div>
                  <input type="range" min="0.1" max="1.0" step="0.1" value={Number(latencyArbConfig.settings.min_move_pct)} onChange={(e) => setLatencyArbConfig({ ...latencyArbConfig, settings: { ...latencyArbConfig.settings, min_move_pct: parseFloat(e.target.value) } })} className="w-full" />
                </div>
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Take Profit %</span><span className="font-mono text-cyan-500">{String(latencyArbConfig.settings.take_profit_pct)}%</span></div>
                  <input type="range" min="2" max="15" step="0.5" value={Number(latencyArbConfig.settings.take_profit_pct)} onChange={(e) => setLatencyArbConfig({ ...latencyArbConfig, settings: { ...latencyArbConfig.settings, take_profit_pct: parseFloat(e.target.value) } })} className="w-full" />
                </div>
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Max Entry Price</span><span className="font-mono text-cyan-500">{(Number(latencyArbConfig.settings.max_entry_price) * 100).toFixed(0)}c</span></div>
                  <input type="range" min="0.5" max="0.8" step="0.05" value={Number(latencyArbConfig.settings.max_entry_price)} onChange={(e) => setLatencyArbConfig({ ...latencyArbConfig, settings: { ...latencyArbConfig.settings, max_entry_price: parseFloat(e.target.value) } })} className="w-full" />
                </div>
              </div>
            </div>
          )}

          {/* CopyTrading Tab */}
          {activeTab === "copy_trading" && (
            <div className="space-y-4">
              <div className="p-3 bg-pink-500/10 rounded-lg border border-pink-500/20">
                <p className="text-xs text-pink-600 dark:text-pink-400">
                  <strong>Copy Trading:</strong> Follow successful traders' positions with configurable position scaling.
                </p>
              </div>
              <label className="flex items-center gap-3 p-3 rounded-lg bg-zinc-50 dark:bg-zinc-800/50">
                <input type="checkbox" checked={copyTradingConfig.enabled} onChange={(e) => setCopyTradingConfig({ ...copyTradingConfig, enabled: e.target.checked })} className="rounded" />
                <span className="text-sm font-medium">Enable Copy Trading</span>
              </label>
              <div className="space-y-3">
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Position Multiplier</span><span className="font-mono text-pink-500">{String(copyTradingConfig.settings.position_multiplier)}x</span></div>
                  <input type="range" min="0.1" max="2.0" step="0.1" value={Number(copyTradingConfig.settings.position_multiplier)} onChange={(e) => setCopyTradingConfig({ ...copyTradingConfig, settings: { ...copyTradingConfig.settings, position_multiplier: parseFloat(e.target.value) } })} className="w-full" />
                </div>
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Max Position USD</span><span className="font-mono text-pink-500">${String(copyTradingConfig.settings.max_position_usd)}</span></div>
                  <input type="range" min="10" max="500" step="10" value={Number(copyTradingConfig.settings.max_position_usd)} onChange={(e) => setCopyTradingConfig({ ...copyTradingConfig, settings: { ...copyTradingConfig.settings, max_position_usd: parseInt(e.target.value) } })} className="w-full" />
                </div>
              </div>
            </div>
          )}

          {/* LatencyGap Tab */}
          {activeTab === "latency_gap" && (
            <div className="space-y-4">
              <div className="p-3 bg-purple-500/10 rounded-lg border border-purple-500/20">
                <p className="text-xs text-purple-600 dark:text-purple-400">
                  <strong>Latency Gap:</strong> Trade based on Binance price leading Polymarket odds. Checkpoint-based entries.
                </p>
              </div>
              <label className="flex items-center gap-3 p-3 rounded-lg bg-zinc-50 dark:bg-zinc-800/50">
                <input type="checkbox" checked={latencyGapConfig.enabled} onChange={(e) => setLatencyGapConfig({ ...latencyGapConfig, enabled: e.target.checked })} className="rounded" />
                <span className="text-sm font-medium">Enable Latency Gap Strategy</span>
              </label>
              <div className="space-y-3">
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Min Edge %</span><span className="font-mono text-purple-500">{String(latencyGapConfig.settings.min_edge_pct)}%</span></div>
                  <input type="range" min="1" max="15" step="0.5" value={Number(latencyGapConfig.settings.min_edge_pct)} onChange={(e) => setLatencyGapConfig({ ...latencyGapConfig, settings: { ...latencyGapConfig.settings, min_edge_pct: parseFloat(e.target.value) } })} className="w-full" />
                </div>
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg">
                  <div className="flex justify-between text-sm mb-2"><span>Min Confidence</span><span className="font-mono text-purple-500">{(Number(latencyGapConfig.settings.min_confidence) * 100).toFixed(0)}%</span></div>
                  <input type="range" min="0.3" max="0.9" step="0.05" value={Number(latencyGapConfig.settings.min_confidence)} onChange={(e) => setLatencyGapConfig({ ...latencyGapConfig, settings: { ...latencyGapConfig.settings, min_confidence: parseFloat(e.target.value) } })} className="w-full" />
                </div>
              </div>
            </div>
          )}

          {/* Live Tools Tab */}
          {activeTab === "live_tools" && isLiveMode && (
            <div className="space-y-4">
              <div className="p-3 bg-red-500/10 rounded-lg border border-red-500/20">
                <p className="text-xs text-red-600 dark:text-red-400">
                  <strong>WARNING:</strong> These controls execute real trades with real money.
                </p>
              </div>

              {/* Allowances */}
              {onSetAllowances && (
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg space-y-2">
                  <span className="text-sm font-medium">Token Allowances</span>
                  <button onClick={handleSetAllowances} disabled={allowancesLoading || !apiKey} className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded disabled:opacity-50">
                    {allowancesLoading ? "Setting..." : "Set Token Allowances"}
                  </button>
                  {allowancesResult && <div className={`p-2 rounded text-sm ${allowancesResult.success ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>{allowancesResult.success ? "Success!" : allowancesResult.error}</div>}
                </div>
              )}

              {/* Test Order */}
              {onTestOrder && (
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg space-y-3">
                  <span className="text-sm font-medium">Manual Test Order</span>
                  <div className="flex gap-2">
                    {(["BTC", "ETH", "SOL"] as const).map((s) => (
                      <button key={s} onClick={() => setTestSymbol(s)} className={`flex-1 py-1.5 text-sm rounded ${testSymbol === s ? "bg-amber-500 text-white" : "bg-zinc-200 dark:bg-zinc-700"}`}>{s}</button>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <button onClick={() => setTestSide("UP")} className={`flex-1 py-1.5 text-sm rounded ${testSide === "UP" ? "bg-green-500 text-white" : "bg-zinc-200 dark:bg-zinc-700"}`}>▲ UP</button>
                    <button onClick={() => setTestSide("DOWN")} className={`flex-1 py-1.5 text-sm rounded ${testSide === "DOWN" ? "bg-red-500 text-white" : "bg-zinc-200 dark:bg-zinc-700"}`}>▼ DOWN</button>
                  </div>
                  <input type="number" min="1" max="100" value={testAmount} onChange={(e) => setTestAmount(Math.min(100, Math.max(1, parseInt(e.target.value) || 1)))} className="w-full px-3 py-1.5 bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700 rounded text-sm" />
                  <button onClick={handleTestOrder} disabled={testOrderLoading || !apiKey} className="w-full px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded disabled:opacity-50">
                    {testOrderLoading ? "Placing..." : `Place ${testSide} $${testAmount}`}
                  </button>
                  {testOrderResult && <div className={`p-2 rounded text-sm ${testOrderResult.success ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>{testOrderResult.success ? "Order placed!" : testOrderResult.error}</div>}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-zinc-200 dark:border-zinc-700">
          <div className="text-xs text-zinc-500">
            Active: {[sniperConfig.enabled && "Sniper", dipArbConfig.enabled && "DipArb", latencyArbConfig.enabled && "LatencyArb", copyTradingConfig.enabled && "CopyTrade", latencyGapConfig.enabled && "LatencyGap"].filter(Boolean).join(", ") || "None"}
          </div>
          <div className="flex gap-2">
            <button onClick={onClose} className="px-4 py-2 rounded bg-zinc-200 dark:bg-zinc-700 text-sm font-medium hover:bg-zinc-300 dark:hover:bg-zinc-600">Cancel</button>
            <button onClick={handleSave} disabled={saving} className="px-4 py-2 rounded bg-purple-500 text-white text-sm font-medium hover:bg-purple-600 disabled:opacity-50">
              {saving ? "Saving..." : "Save Changes"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
