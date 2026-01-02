"use client";

import { memo } from "react";

interface ConnectionStatusProps {
  isConnected: boolean;
}

function ConnectionStatusComponent({ isConnected }: ConnectionStatusProps) {
  return (
    <div className="flex items-center gap-2">
      <div
        className={`w-2 h-2 rounded-full ${
          isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"
        }`}
      />
      <span className="text-xs text-zinc-500">
        {isConnected ? "Connected" : "Disconnected"}
      </span>
    </div>
  );
}

export const ConnectionStatus = memo(ConnectionStatusComponent);
