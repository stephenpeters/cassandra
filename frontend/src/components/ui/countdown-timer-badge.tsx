"use client";

import { memo, useState, useEffect } from "react";
import { Badge } from "./badge";

interface CountdownTimerBadgeCompactProps {
  seconds: number;
  className?: string;
}

export const CountdownTimerBadgeCompact = memo(function CountdownTimerBadgeCompact({
  seconds,
  className = "",
}: CountdownTimerBadgeCompactProps) {
  const [remaining, setRemaining] = useState(seconds);

  useEffect(() => {
    setRemaining(seconds);
  }, [seconds]);

  useEffect(() => {
    if (remaining <= 0) return;

    const timer = setInterval(() => {
      setRemaining((prev) => Math.max(0, prev - 1));
    }, 1000);

    return () => clearInterval(timer);
  }, [remaining]);

  const mins = Math.floor(remaining / 60);
  const secs = remaining % 60;

  const isUrgent = remaining <= 120; // Last 2 minutes
  const isCritical = remaining <= 30; // Last 30 seconds

  return (
    <Badge
      className={`font-mono text-xs ${
        isCritical
          ? "bg-red-500/20 text-red-400 animate-pulse"
          : isUrgent
          ? "bg-orange-500/20 text-orange-400"
          : "bg-blue-500/20 text-blue-400"
      } ${className}`}
    >
      {mins}:{secs.toString().padStart(2, "0")}
    </Badge>
  );
});
