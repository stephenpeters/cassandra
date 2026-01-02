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

export default function Home() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTC");

  const {
    isConnected,
    candles,
    orderbooks,
    momentum,
    whaleTrades,
    whales,
    symbols,
  } = useWebSocket();

  // Map symbols to Binance format for data lookup
  const symbolToBinance: Record<string, string> = {
    BTC: "BTCUSDT",
    ETH: "ETHUSDT",
    SOL: "SOLUSDT",
    XRP: "XRPUSDT",
    DOGE: "DOGEUSDT",
  };

  const currentCandles = candles[selectedSymbol] || [];
  const currentOrderbook = orderbooks[symbolToBinance[selectedSymbol]?.toLowerCase()];
  const currentMomentum = momentum[selectedSymbol];

  // Count significant whale trades (>$1000)
  const significantTrades = whaleTrades.filter((t) => t.usd_value > 1000).length;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4">
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
            <ConnectionStatus isConnected={isConnected} />
            <div className="text-xs text-zinc-500">
              Tracking {whales.length} whales
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Symbol selector */}
        <div className="flex gap-2">
          {(symbols.length > 0 ? symbols : ["BTC", "ETH", "SOL", "XRP", "DOGE"]).map(
            (sym) => (
              <button
                key={sym}
                onClick={() => setSelectedSymbol(sym)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedSymbol === sym
                    ? "bg-blue-600 text-white"
                    : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                }`}
              >
                {sym}
              </button>
            )
          )}
        </div>

        {/* Charts grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Price chart - takes 2 columns */}
          <Card className="lg:col-span-2 bg-zinc-900 border-zinc-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-zinc-200">
                {selectedSymbol}/USDT Price
              </CardTitle>
            </CardHeader>
            <CardContent>
              <PriceChart
                symbol={`${selectedSymbol}/USDT`}
                candles={currentCandles}
                height={350}
              />
            </CardContent>
          </Card>

          {/* Side panel */}
          <div className="space-y-6">
            {/* Momentum */}
            <Card className="bg-zinc-900 border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg text-zinc-200">
                  Momentum Signal
                </CardTitle>
              </CardHeader>
              <CardContent>
                <MomentumIndicator
                  symbol={selectedSymbol}
                  signal={currentMomentum}
                />
              </CardContent>
            </Card>

            {/* Order book */}
            <Card className="bg-zinc-900 border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg text-zinc-200">
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
          </div>
        </div>

        {/* All symbols momentum overview */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg text-zinc-200">
              All Signals Overview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {Object.entries(momentum).map(([sym, signal]) => (
                <div
                  key={sym}
                  className={`p-4 rounded-lg cursor-pointer transition-all ${
                    selectedSymbol === sym
                      ? "ring-2 ring-blue-500 bg-zinc-800"
                      : "bg-zinc-800/50 hover:bg-zinc-800"
                  }`}
                  onClick={() => setSelectedSymbol(sym)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">{sym}</span>
                    <Badge
                      className={`text-xs ${
                        signal.direction === "UP"
                          ? "bg-green-500/20 text-green-400"
                          : signal.direction === "DOWN"
                          ? "bg-red-500/20 text-red-400"
                          : "bg-zinc-600/20 text-zinc-400"
                      }`}
                    >
                      {signal.direction}
                    </Badge>
                  </div>
                  <div className="text-xs text-zinc-500">
                    Confidence: {(signal.confidence * 100).toFixed(0)}%
                  </div>
                  <div
                    className={`text-sm font-mono ${
                      signal.price_change_pct > 0
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    {signal.price_change_pct > 0 ? "+" : ""}
                    {signal.price_change_pct.toFixed(3)}%
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Whale trades */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg text-zinc-200">
                Whale Trades
              </CardTitle>
              <div className="flex items-center gap-2">
                {significantTrades > 0 && (
                  <Badge className="bg-yellow-500/20 text-yellow-400">
                    {significantTrades} large trades
                  </Badge>
                )}
                <span className="text-xs text-zinc-500">
                  {whaleTrades.length} total
                </span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="all">
              <TabsList className="bg-zinc-800 mb-4">
                <TabsTrigger value="all">All Trades</TabsTrigger>
                <TabsTrigger value="crypto">Crypto Only</TabsTrigger>
                <TabsTrigger value="large">Large (&gt;$1K)</TabsTrigger>
              </TabsList>
              <TabsContent value="all">
                <WhaleTradesTable trades={whaleTrades} />
              </TabsContent>
              <TabsContent value="crypto">
                <WhaleTradesTable
                  trades={whaleTrades.filter(
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
                  trades={whaleTrades.filter((t) => t.usd_value > 1000)}
                />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* Tracked whales */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-lg text-zinc-200">
              Tracked Whales
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {whales.map((whale) => (
                <div
                  key={whale.name}
                  className="p-4 bg-zinc-800/50 rounded-lg"
                >
                  <div className="font-medium text-cyan-400">{whale.name}</div>
                  <div className="text-xs text-zinc-500 font-mono mt-1">
                    {whale.address}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800 px-6 py-4 mt-8">
        <div className="max-w-7xl mx-auto text-center text-xs text-zinc-500">
          Real-time data from Binance + Polymarket | Not financial advice
        </div>
      </footer>
    </div>
  );
}
