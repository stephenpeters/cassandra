"use client";

import { useState } from "react";
import { Info } from "lucide-react";

interface InfoTooltipProps {
  content: string;
  className?: string;
}

export function InfoTooltip({ content, className = "" }: InfoTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <span className={`relative inline-flex items-center ${className}`}>
      <button
        type="button"
        className="p-0.5 rounded-full text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        aria-label="More info"
      >
        <Info className="w-3.5 h-3.5" />
      </button>
      {isVisible && (
        <div className="absolute z-50 bottom-full left-0 mb-2 px-3 py-2 text-xs text-zinc-100 bg-zinc-800 dark:bg-zinc-700 rounded-lg shadow-lg w-64 whitespace-normal">
          {content}
          <div className="absolute top-full left-3 border-4 border-transparent border-t-zinc-800 dark:border-t-zinc-700" />
        </div>
      )}
    </span>
  );
}
