"use client";

import { memo } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { WhaleTrade } from "@/types";

interface WhaleTradesTableProps {
  trades: WhaleTrade[];
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatMarket(market: string): string {
  // Shorten market title for display
  if (market.length > 50) {
    return market.substring(0, 47) + "...";
  }
  return market;
}

function WhaleTradesTableComponent({ trades }: WhaleTradesTableProps) {
  if (trades.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-zinc-500">
        Waiting for whale trades...
      </div>
    );
  }

  return (
    <ScrollArea className="h-[400px]">
      <Table>
        <TableHeader className="sticky top-0 bg-zinc-900">
          <TableRow className="border-zinc-800">
            <TableHead className="text-zinc-400">Time</TableHead>
            <TableHead className="text-zinc-400">Whale</TableHead>
            <TableHead className="text-zinc-400">Market</TableHead>
            <TableHead className="text-zinc-400">Side</TableHead>
            <TableHead className="text-zinc-400 text-right">Size</TableHead>
            <TableHead className="text-zinc-400 text-right">Price</TableHead>
            <TableHead className="text-zinc-400">Signal</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {trades.map((trade, idx) => {
            const isCryptoMarket =
              trade.market.toLowerCase().includes("bitcoin") ||
              trade.market.toLowerCase().includes("btc") ||
              trade.market.toLowerCase().includes("eth") ||
              trade.market.toLowerCase().includes("sol") ||
              trade.market.toLowerCase().includes("xrp");

            // Determine signal based on outcome and side
            const outcomeL = trade.outcome.toLowerCase();
            const isBullish =
              (trade.side === "BUY" &&
                (outcomeL.includes("up") || outcomeL.includes("above"))) ||
              (trade.side === "SELL" &&
                (outcomeL.includes("down") || outcomeL.includes("below")));

            return (
              <TableRow
                key={`${trade.tx_hash}-${idx}`}
                className={`border-zinc-800 ${
                  trade.usd_value > 1000 ? "bg-yellow-500/5" : ""
                }`}
              >
                <TableCell className="font-mono text-xs text-zinc-500">
                  {formatTime(trade.timestamp)}
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    {trade.icon && (
                      <img
                        src={trade.icon}
                        alt=""
                        className="w-5 h-5 rounded"
                      />
                    )}
                    <span className="text-cyan-400 font-medium">
                      {trade.whale}
                    </span>
                  </div>
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <span className="text-zinc-300 text-sm truncate block">
                    {formatMarket(trade.market)}
                  </span>
                  <span className="text-xs text-zinc-500">{trade.outcome}</span>
                </TableCell>
                <TableCell>
                  <Badge
                    variant="outline"
                    className={
                      trade.side === "BUY"
                        ? "border-green-500 text-green-500"
                        : "border-red-500 text-red-500"
                    }
                  >
                    {trade.side}
                  </Badge>
                </TableCell>
                <TableCell className="text-right font-mono">
                  ${trade.size.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </TableCell>
                <TableCell className="text-right font-mono text-zinc-400">
                  {trade.price.toFixed(2)}
                </TableCell>
                <TableCell>
                  {isCryptoMarket && (
                    <Badge
                      className={
                        isBullish
                          ? "bg-green-500/20 text-green-400 border-green-500"
                          : "bg-red-500/20 text-red-400 border-red-500"
                      }
                    >
                      {isBullish ? "BULL" : "BEAR"}
                    </Badge>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </ScrollArea>
  );
}

export const WhaleTradesTable = memo(WhaleTradesTableComponent);
