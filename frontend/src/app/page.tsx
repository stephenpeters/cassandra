"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { useWebSocket } from "@/hooks/useWebSocket";
import { PriceChart } from "@/components/charts/PriceChart";
import { OrderBookChart } from "@/components/charts/OrderBookChart";
import { MomentumIndicator } from "@/components/charts/MomentumIndicator";
import { WhaleTradesTable } from "@/components/WhaleTradesTable";
import { ConnectionStatus } from "@/components/ConnectionStatus";
import { ThemeToggle } from "@/components/ThemeToggle";
import { PolymarketCard } from "@/components/PolymarketCard";
import { Markets15MinStatus } from "@/components/Markets15MinStatus";
import { PaperTradingCard } from "@/components/PaperTradingCard";
import { TradingSignalsPanel } from "@/components/TradingSignalsPanel";
import { Maximize2, Minimize2 } from "lucide-react";

export default function Home() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTC");
  const [selectedWhale, setSelectedWhale] = useState<string | null>(null);
  const [chartExpanded, setChartExpanded] = useState(false);

  const {
    isConnected,
    candles,
    orderbooks,
    momentum,
    whaleTrades,
    whales,
    symbols,
    markets15m,
    marketTrades,
    paperAccount,
    paperSignals,
    paperConfig,
    togglePaperTrading,
    resetPaperAccount,
    updatePaperConfig,
  } = useWebSocket();

  // Map symbols to Binance format for data lookup
  const symbolToBinance: Record<string, string> = {
    BTC: "BTCUSDT",
    ETH: "ETHUSDT",
    SOL: "SOLUSDT",
    XRP: "XRPUSDT",
    DOGE: "DOGEUSDT",
  };

  const binanceSymbol = symbolToBinance[selectedSymbol];
  const currentCandles = candles[selectedSymbol] || [];
  const currentOrderbook = orderbooks[binanceSymbol];
  const currentMomentum = momentum[binanceSymbol];

  // Filter trades by selected whale
  const filteredTrades = selectedWhale
    ? whaleTrades.filter((t) => t.whale === selectedWhale)
    : whaleTrades;

  // Count significant whale trades (>$1000)
  const significantTrades = filteredTrades.filter((t) => t.usd_value > 1000).length;

  return (
    <div className="min-h-screen bg-zinc-100 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 transition-colors">
      {/* Header */}
      <header className="border-b border-zinc-300 dark:border-zinc-800 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              Polymarket Whale Tracker
            </h1>
            <Badge variant="outline" className="text-xs">
              Crypto 15-Min Markets
            </Badge>
          </div>
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <ConnectionStatus isConnected={isConnected} />
            <div className="text-xs text-zinc-600 dark:text-zinc-500">
              Tracking {whales.length} whales
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto p-6 space-y-6">
        {/* 15-Minute Markets Status - at top */}
        <Markets15MinStatus
          markets15m={markets15m}
          marketTrades={marketTrades}
          momentum={momentum}
          selectedSymbol={selectedSymbol}
          onSymbolSelect={setSelectedSymbol}
        />

        {/* Paper Trading Row - 2 columns */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Trading Signals Panel */}
          <TradingSignalsPanel
            signals={paperSignals}
            momentum={momentum}
            selectedSymbol={selectedSymbol}
          />

          {/* Paper Trading Card */}
          <PaperTradingCard
            account={paperAccount}
            signals={paperSignals}
            config={paperConfig}
            onToggle={togglePaperTrading}
            onReset={resetPaperAccount}
            onConfigUpdate={updatePaperConfig}
          />
        </div>

        {/* Main grid - 4 columns */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Price chart - collapsible */}
          <Card className={`bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800 ${chartExpanded ? "lg:col-span-4" : "lg:col-span-1"}`}>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
                  {selectedSymbol}/USDT
                </CardTitle>
                <button
                  onClick={() => setChartExpanded(!chartExpanded)}
                  className="p-1.5 rounded bg-zinc-200 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-colors"
                  title={chartExpanded ? "Minimize" : "Maximize"}
                >
                  {chartExpanded ? (
                    <Minimize2 className="h-4 w-4 text-zinc-600 dark:text-zinc-400" />
                  ) : (
                    <Maximize2 className="h-4 w-4 text-zinc-600 dark:text-zinc-400" />
                  )}
                </button>
              </div>
            </CardHeader>
            <CardContent>
              <PriceChart
                symbol={`${selectedSymbol}/USDT`}
                candles={currentCandles}
                height={chartExpanded ? 350 : 200}
              />
            </CardContent>
          </Card>

          {/* Momentum - only show when chart is minimized */}
          {!chartExpanded && (
            <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
                  Momentum
                </CardTitle>
              </CardHeader>
              <CardContent>
                <MomentumIndicator
                  symbol={selectedSymbol}
                  signal={currentMomentum}
                />
              </CardContent>
            </Card>
          )}

          {/* Order book - only show when chart is minimized */}
          {!chartExpanded && (
            <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
                  Order Book
                </CardTitle>
              </CardHeader>
              <CardContent>
                <OrderBookChart
                  symbol={selectedSymbol}
                  data={currentOrderbook}
                />
              </CardContent>
            </Card>
          )}

          {/* Polymarket 15-min markets - only show when chart is minimized */}
          {!chartExpanded && (
            <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
                  Polymarket
                </CardTitle>
              </CardHeader>
              <CardContent>
                <PolymarketCard
                  symbol={selectedSymbol}
                  whaleTrades={whaleTrades}
                  momentum={currentMomentum}
                />
              </CardContent>
            </Card>
          )}
        </div>

        {/* Whale selector + trades */}
        <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
          <CardHeader>
            <div className="flex items-center justify-between flex-wrap gap-4">
              <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
                Whale Trades
              </CardTitle>
              <div className="flex items-center gap-2 flex-wrap">
                {/* Whale selector */}
                <div className="flex gap-1 flex-wrap">
                  <button
                    onClick={() => setSelectedWhale(null)}
                    className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                      selectedWhale === null
                        ? "bg-cyan-600 text-white"
                        : "bg-zinc-200 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-300 dark:hover:bg-zinc-700"
                    }`}
                  >
                    All Whales
                  </button>
                  {whales.map((whale) => (
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
                </div>
                {significantTrades > 0 && (
                  <Badge className="bg-yellow-500/20 text-yellow-400">
                    {significantTrades} large trades
                  </Badge>
                )}
                <span className="text-xs text-zinc-600 dark:text-zinc-500">
                  {filteredTrades.length} total
                </span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="all">
              <TabsList className="bg-zinc-200 dark:bg-zinc-800 mb-4">
                <TabsTrigger value="all">All Trades</TabsTrigger>
                <TabsTrigger value="crypto">Crypto Only</TabsTrigger>
                <TabsTrigger value="large">Large (&gt;$1K)</TabsTrigger>
              </TabsList>
              <TabsContent value="all">
                <WhaleTradesTable trades={filteredTrades} />
              </TabsContent>
              <TabsContent value="crypto">
                <WhaleTradesTable
                  trades={filteredTrades.filter(
                    (t) =>
                      t.market.toLowerCase().includes("bitcoin") ||
                      t.market.toLowerCase().includes("btc") ||
                      t.market.toLowerCase().includes("eth") ||
                      t.market.toLowerCase().includes("sol") ||
                      t.market.toLowerCase().includes("xrp")
                  )}
                />
              </TabsContent>
              <TabsContent value="large">
                <WhaleTradesTable
                  trades={filteredTrades.filter((t) => t.usd_value > 1000)}
                />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* Tracked whales info */}
        <Card className="bg-white dark:bg-zinc-900 border-zinc-300 dark:border-zinc-800">
          <CardHeader>
            <CardTitle className="text-lg text-zinc-800 dark:text-zinc-200">
              Tracked Whales
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {whales.map((whale) => (
                <div
                  key={whale.name}
                  onClick={() => setSelectedWhale(whale.name === selectedWhale ? null : whale.name)}
                  className={`p-4 rounded-lg cursor-pointer transition-all ${
                    selectedWhale === whale.name
                      ? "ring-2 ring-cyan-500 bg-cyan-50 dark:bg-zinc-800"
                      : "bg-zinc-100 dark:bg-zinc-800/50 hover:bg-zinc-200 dark:hover:bg-zinc-800"
                  }`}
                >
                  <div className="font-medium text-cyan-600 dark:text-cyan-400">{whale.name}</div>
                  <div className="text-xs text-zinc-600 dark:text-zinc-500 font-mono mt-1">
                    {whale.address}
                  </div>
                  {whale.strategy && (
                    <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">
                      {whale.strategy}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-300 dark:border-zinc-800 px-6 py-4 mt-8">
        <div className="max-w-7xl mx-auto text-center text-xs text-zinc-600 dark:text-zinc-500">
          Real-time data from Binance + Polymarket | Not financial advice
        </div>
      </footer>
    </div>
  );
}
