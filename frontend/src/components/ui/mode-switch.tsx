"use client";

import { memo } from "react";
import type { TradingModeValue } from "@/types";

interface ModeSwitchProps {
  mode: TradingModeValue;
  onChange: (mode: TradingModeValue) => void;
  disabled?: boolean;
  className?: string;
}

/**
 * 3-way mode switch: OFF - PAPER - LIVE
 *
 * OFF = Kill switch (stops all trading)
 * PAPER = Simulated trades
 * LIVE = Real trades (CAUTION!)
 */
function ModeSwitchComponent({ mode, onChange, disabled = false, className = "" }: ModeSwitchProps) {
  const modes: { value: TradingModeValue; label: string; color: string; activeColor: string }[] = [
    {
      value: "off",
      label: "OFF",
      color: "text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300",
      activeColor: "bg-zinc-500 text-white",
    },
    {
      value: "paper",
      label: "PAPER",
      color: "text-blue-400 hover:text-blue-500",
      activeColor: "bg-blue-500 text-white",
    },
    {
      value: "live",
      label: "LIVE",
      color: "text-red-400 hover:text-red-500",
      activeColor: "bg-red-500 text-white",
    },
  ];

  return (
    <div className={`flex items-center rounded-lg border border-zinc-300 dark:border-zinc-700 overflow-hidden ${className}`}>
      {modes.map((m) => (
        <button
          key={m.value}
          onClick={() => !disabled && onChange(m.value)}
          disabled={disabled}
          className={`
            px-3 py-1.5 text-xs font-semibold transition-all
            ${mode === m.value
              ? m.activeColor
              : `bg-zinc-100 dark:bg-zinc-800 ${m.color}`
            }
            ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
            first:rounded-l-md last:rounded-r-md
            border-r border-zinc-300 dark:border-zinc-700 last:border-r-0
          `}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

export const ModeSwitch = memo(ModeSwitchComponent);
