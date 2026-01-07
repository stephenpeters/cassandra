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
  if (market.length > 40) {
    return market.substring(0, 37) + "...";
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
        <TableHeader className="sticky top-0 bg-zinc-100 dark:bg-zinc-900">
          <TableRow className="border-zinc-300 dark:border-zinc-800">
            <TableHead className="text-zinc-400 w-20">Time</TableHead>
            <TableHead className="text-zinc-400 w-24">Whale</TableHead>
            <TableHead className="text-zinc-400">Market</TableHead>
            <TableHead className="text-zinc-400 w-16 text-center">Pos</TableHead>
            <TableHead className="text-zinc-400 w-20 text-right">Size</TableHead>
            <TableHead className="text-zinc-400 w-20 text-right">Value</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {trades.map((trade, idx) => {
            const outcomeL = trade.outcome.toLowerCase();
            const isBullish = outcomeL.includes("up") || outcomeL.includes("above");

            return (
              <TableRow
                key={`${trade.tx_hash}-${idx}`}
                className={`border-zinc-300 dark:border-zinc-800 ${
                  trade.usd_value > 1000 ? "bg-yellow-500/10 dark:bg-yellow-500/5" : ""
                }`}
              >
                <TableCell className="font-mono text-xs text-zinc-500 py-2">
                  {formatTime(trade.timestamp)}
                </TableCell>
                <TableCell className="py-2">
                  <div className="flex items-center gap-1">
                    {trade.icon && (
                      <img
                        src={trade.icon}
                        alt=""
                        className="w-4 h-4 rounded"
                      />
                    )}
                    <span className="text-cyan-600 dark:text-cyan-400 font-medium text-xs truncate">
                      {trade.whale}
                    </span>
                  </div>
                </TableCell>
                <TableCell className="py-2">
                  <span className="text-zinc-700 dark:text-zinc-300 text-xs truncate block">
                    {formatMarket(trade.market)}
                  </span>
                </TableCell>
                <TableCell className="py-2 text-center">
                  <Badge
                    className={`text-[10px] ${
                      isBullish
                        ? "bg-green-500/20 text-green-400"
                        : "bg-red-500/20 text-red-400"
                    }`}
                  >
                    {isBullish ? "UP" : "DN"}
                  </Badge>
                </TableCell>
                <TableCell className="text-right font-mono text-xs py-2">
                  {trade.size.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </TableCell>
                <TableCell className="text-right font-mono text-xs text-green-500 py-2">
                  ${trade.usd_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
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
