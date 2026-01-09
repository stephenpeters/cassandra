"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { useWebSocket } from "@/hooks/useWebSocket";
import { ConnectionStatus } from "@/components/ConnectionStatus";
import { ThemeToggle } from "@/components/ThemeToggle";
import { PolymarketDashboard } from "@/components/PolymarketDashboard";
import { TradingCard } from "@/components/TradingCard";
import { SettingsModal } from "@/components/SettingsModal";
import { StrategySettingsModal } from "@/components/StrategySettingsModal";
import { FollowingModal } from "@/components/FollowingModal";
import { HistoryModal } from "@/components/HistoryModal";
import { WhaleTradesTable } from "@/components/WhaleTradesTable";
import { Settings, Users, Database, Sliders } from "lucide-react";

export default function Home() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTC");
  const [selectedWhale, setSelectedWhale] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [strategyOpen, setStrategyOpen] = useState(false);
  const [followingOpen, setFollowingOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);

  const {
    isConnected,
    orderbooks,
    momentum,
    whaleTrades,
    whales,
    markets15m,
    marketTrades,
    chartData,
    paperAccount,
    paperSignals,
    paperConfig,
    liveStatus,
    togglePaperTrading,
    resetTradingAccount,
    factoryReset,
    updatePaperConfig,
    setTradingMode,
    placeTestOrder,
    setAllowances,
    activateKillSwitch,
    deactivateKillSwitch,
  } = useWebSocket();

  // Filter trades by selected whale
  const filteredTrades = selectedWhale
    ? whaleTrades.filter((t) => t.whale === selectedWhale)
    : whaleTrades;

  // Count significant whale trades (>$1000)
  const significantTrades = filteredTrades.filter((t) => t.usd_value > 1000).length;

  // Get current market for selected symbol
  const currentMarket = markets15m?.active?.[selectedSymbol];

  return (
    <div className="min-h-screen bg-zinc-100 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 transition-colors">
      {/* Header */}
      <header className="border-b border-zinc-300 dark:border-zinc-800 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
              Cassandra Polymarket Crypto Bot
            </h1>
          </div>
          <div className="flex items-center gap-3">
            {/* History button */}
            <button
              onClick={() => setHistoryOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-200 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-colors text-sm"
            >
              <Database className="w-4 h-4 text-purple-500" />
              <span className="hidden sm:inline">History</span>
            </button>
            {/* Strategy button (includes Copy Trading) */}
            <button
              onClick={() => setStrategyOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-200 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-colors text-sm"
            >
              <Sliders className="w-4 h-4 text-purple-500" />
              <span className="hidden sm:inline">Strategy</span>
            </button>
            {/* Settings button */}
            <button
              onClick={() => setSettingsOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-200 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-colors text-sm"
            >
              <Settings className="w-4 h-4 text-zinc-500" />
              <span className="hidden sm:inline">Settings</span>
            </button>
            <ThemeToggle />
            <ConnectionStatus isConnected={isConnected} />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Polymarket Dashboard - Summary cards + detail view */}
        <PolymarketDashboard
          markets15m={markets15m}
          marketTrades={marketTrades}
          momentum={momentum}
          chartData={chartData}
          orderbooks={orderbooks}
          positions={paperAccount?.positions || []}
          selectedSymbol={selectedSymbol}
          onSymbolSelect={setSelectedSymbol}
          signals={paperSignals}
        />

        {/* Trading Card - supports both Paper and Live modes */}
        <TradingCard
          account={paperAccount}
          signals={paperSignals}
          config={paperConfig}
          liveStatus={liveStatus}
          markets15m={markets15m}
          onToggle={togglePaperTrading}
          onReset={resetTradingAccount}
          onFactoryReset={factoryReset}
          onConfigUpdate={updatePaperConfig}
          onModeChange={setTradingMode}
          onSetAllowances={setAllowances}
          onKillSwitch={async (activate: boolean, reason?: string) => {
            const apiKey = localStorage.getItem("predmkt_api_key") || "";
            if (activate) {
              return activateKillSwitch(reason || "Manual UI activation", apiKey);
            } else {
              return deactivateKillSwitch(apiKey);
            }
          }}
          onManualOrder={async (order) => {
            const apiKey = localStorage.getItem("predmkt_api_key") || "";
            return placeTestOrder(order.symbol, order.side, order.amount, apiKey);
          }}
        />

        {/* Copy Trades - Simplified */}
        <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
          <CardHeader>
            <div className="flex items-center justify-between flex-wrap gap-4">
              <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
                Copy Trades
              </CardTitle>
              <div className="flex items-center gap-2 flex-wrap">
                <button
                  onClick={() => setSelectedWhale(null)}
                  className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    selectedWhale === null
                      ? "bg-cyan-600 text-white"
                      : "bg-zinc-200 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-300 dark:hover:bg-zinc-700"
                  }`}
                >
                  All
                </button>
                {whales.slice(0, 3).map((whale) => (
                  <button
                    key={whale.name}
                    onClick={() => setSelectedWhale(whale.name)}
                    className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                      selectedWhale === whale.name
                        ? "bg-cyan-600 text-white"
                        : "bg-zinc-200 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-300 dark:hover:bg-zinc-700"
                    }`}
                  >
                    {whale.name}
                  </button>
                ))}
                {whales.length > 3 && (
                  <button
                    onClick={() => setFollowingOpen(true)}
                    className="px-2 py-1 rounded text-xs text-zinc-500 hover:text-zinc-400"
                  >
                    +{whales.length - 3} more
                  </button>
                )}
                {significantTrades > 0 && (
                  <Badge className="bg-yellow-500/20 text-yellow-400">
                    {significantTrades} large
                  </Badge>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="crypto">
              <TabsList className="bg-zinc-200 dark:bg-zinc-800 mb-4">
                <TabsTrigger value="crypto">Crypto</TabsTrigger>
                <TabsTrigger value="all">All</TabsTrigger>
                <TabsTrigger value="large">&gt;$1K</TabsTrigger>
              </TabsList>
              <TabsContent value="crypto">
                <WhaleTradesTable
                  trades={filteredTrades.filter(
                    (t) =>
                      t.market.toLowerCase().includes("bitcoin") ||
                      t.market.toLowerCase().includes("btc") ||
                      t.market.toLowerCase().includes("eth") ||
                      t.market.toLowerCase().includes("sol")
                  )}
                />
              </TabsContent>
              <TabsContent value="all">
                <WhaleTradesTable trades={filteredTrades} />
              </TabsContent>
              <TabsContent value="large">
                <WhaleTradesTable
                  trades={filteredTrades.filter((t) => t.usd_value > 1000)}
                />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-300 dark:border-zinc-800 px-6 py-4 mt-8">
        <div className="max-w-7xl mx-auto text-center text-xs text-zinc-600 dark:text-zinc-500">
          Real-time data from Binance + Polymarket | Not financial advice
        </div>
      </footer>

      {/* Modals */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        config={paperConfig}
        onConfigUpdate={updatePaperConfig}
      />
      <StrategySettingsModal
        isOpen={strategyOpen}
        onClose={() => setStrategyOpen(false)}
        config={paperConfig}
        onConfigUpdate={updatePaperConfig}
        liveStatus={liveStatus}
        onTestOrder={placeTestOrder}
        onSetAllowances={setAllowances}
      />
      <FollowingModal
        isOpen={followingOpen}
        onClose={() => setFollowingOpen(false)}
        whales={whales}
        whaleTrades={whaleTrades}
        selectedWhale={selectedWhale}
        onWhaleSelect={setSelectedWhale}
      />
      <HistoryModal
        isOpen={historyOpen}
        onClose={() => setHistoryOpen(false)}
      />
    </div>
  );
}
