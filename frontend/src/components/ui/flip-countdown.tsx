"use client";

import { useEffect, useState } from "react";
import FlipNumbers from "react-flip-numbers";

interface FlipCountdownProps {
  seconds: number;
  className?: string;
  showLabels?: boolean;
}

export function FlipCountdown({ seconds, className = "", showLabels = false }: FlipCountdownProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;

  // Format as MM:SS
  const minsStr = mins.toString().padStart(2, "0");
  const secsStr = secs.toString().padStart(2, "0");

  // Don't render flip numbers on server
  if (!mounted) {
    return (
      <div className={`font-mono text-2xl ${className}`}>
        {minsStr}:{secsStr}
      </div>
    );
  }

  return (
    <div className={`flex items-center gap-1 ${className}`}>
      <div className="flex flex-col items-center">
        <FlipNumbers
          height={28}
          width={20}
          color="currentColor"
          background="transparent"
          play
          perspective={200}
          numbers={minsStr}
        />
        {showLabels && <span className="text-[10px] text-zinc-500 mt-1">MIN</span>}
      </div>
      <span className="text-2xl font-bold opacity-50">:</span>
      <div className="flex flex-col items-center">
        <FlipNumbers
          height={28}
          width={20}
          color="currentColor"
          background="transparent"
          play
          perspective={200}
          numbers={secsStr}
        />
        {showLabels && <span className="text-[10px] text-zinc-500 mt-1">SEC</span>}
      </div>
    </div>
  );
}

// Compact version for inline use
export function FlipCountdownCompact({ seconds, className = "" }: Omit<FlipCountdownProps, "showLabels">) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  const minsStr = mins.toString().padStart(2, "0");
  const secsStr = secs.toString().padStart(2, "0");

  if (!mounted) {
    return (
      <span className={`font-mono ${className}`}>
        {minsStr}:{secsStr}
      </span>
    );
  }

  return (
    <div className={`flex items-center gap-0.5 ${className}`}>
      <FlipNumbers
        height={18}
        width={12}
        color="currentColor"
        background="transparent"
        play
        perspective={150}
        numbers={minsStr}
      />
      <span className="text-sm font-bold opacity-50">:</span>
      <FlipNumbers
        height={18}
        width={12}
        color="currentColor"
        background="transparent"
        play
        perspective={150}
        numbers={secsStr}
      />
    </div>
  );
}
