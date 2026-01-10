"use client";

import { memo, useState, useEffect } from "react";
import FlipNumbers from "react-flip-numbers";

interface FlipCountdownProps {
  seconds: number;
  className?: string;
}

export const FlipCountdown = memo(function FlipCountdown({
  seconds,
  className = "",
}: FlipCountdownProps) {
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

  const isLastMinute = remaining <= 60; // Last minute - red background
  const isCritical = remaining <= 30; // Last 30 seconds - pulse

  // Always white text, blue bg normally, red bg in last minute
  const textColor = "#ffffff";
  const bgClass = isLastMinute
    ? `bg-red-600 ${isCritical ? "animate-pulse" : ""}`
    : "bg-blue-600";
  const colonColor = isLastMinute ? "#ef4444" : "#3b82f6";

  return (
    <div className={`flex items-center gap-1 ${className}`}>
      <div className={`${bgClass} rounded px-1.5 py-0.5 transition-colors`}>
        <FlipNumbers
          height={20}
          width={14}
          color={textColor}
          background="transparent"
          play
          numbers={String(mins).padStart(2, "0")}
        />
      </div>
      <span className="font-bold text-lg" style={{ color: colonColor }}>:</span>
      <div className={`${bgClass} rounded px-1.5 py-0.5 transition-colors`}>
        <FlipNumbers
          height={20}
          width={14}
          color={textColor}
          background="transparent"
          play
          numbers={String(secs).padStart(2, "0")}
        />
      </div>
    </div>
  );
});
