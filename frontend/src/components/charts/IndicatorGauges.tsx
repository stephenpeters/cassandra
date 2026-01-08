"use client";

import { memo } from "react";
import { InfoTooltip } from "@/components/ui/info-tooltip";
import type { MomentumSignal } from "@/types";

interface IndicatorGaugesProps {
  momentum: MomentumSignal | undefined;
  compact?: boolean;
}

/**
 * Dial Gauge component - renders a semicircular gauge
 */
interface DialGaugeProps {
  value: number;        // Current value
  min: number;          // Minimum value
  max: number;          // Maximum value
  label: string;        // Label text
  zones?: { start: number; end: number; color: string }[];  // Colored zones
  valueColor?: string;  // Color for the value text
  size?: number;        // Size in pixels (default 100)
  showValue?: boolean;  // Whether to show the value
  tooltip?: string;     // Tooltip text on hover
}

function DialGauge({
  value,
  min,
  max,
  label,
  zones = [],
  valueColor = "text-zinc-200",
  size = 60,
  showValue = true,
  tooltip,
}: DialGaugeProps) {
  const normalizedValue = Math.max(min, Math.min(max, value));
  const valueRatio = (normalizedValue - min) / (max - min);

  const r = size * 0.38;          // Arc radius
  const strokeW = size * 0.1;     // Arc thickness
  const cx = size / 2;            // Center X
  const viewH = r + strokeW + 6;  // Total height
  const cy = viewH - (strokeW / 2 + 2);  // Pivot near BOTTOM
  const needleLen = r * 0.82;     // Needle length

  // Needle angle: 0° at min (left), 180° at max (right)
  // Needle points UP into the arc (flipped 180°)
  const needleAngle = Math.PI * valueRatio;
  const needleX = cx - needleLen * Math.cos(needleAngle);
  const needleY = cy - needleLen * Math.sin(needleAngle);  // Minus = points UP

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={viewH} viewBox={`0 0 ${size} ${viewH}`}>
        {/* Background arc - curves UP (smile shape, flipped 180°) */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeW}
          className="text-zinc-700"
        />

        {/* Colored zones */}
        {zones.map((zone, i) => {
          const startRatio = (zone.start - min) / (max - min);
          const endRatio = (zone.end - min) / (max - min);
          // 0° = left, 180° = right, arc curves UP (flipped 180°)
          const startAng = Math.PI * startRatio;
          const endAng = Math.PI * endRatio;

          const x1 = cx - r * Math.cos(startAng);
          const y1 = cy - r * Math.sin(startAng);  // Minus = curves UP
          const x2 = cx - r * Math.cos(endAng);
          const y2 = cy - r * Math.sin(endAng);    // Minus = curves UP

          const arcSpan = (endRatio - startRatio) * 180;
          const largeArc = arcSpan > 90 ? 1 : 0;

          return (
            <path
              key={i}
              d={`M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`}
              fill="none"
              stroke={zone.color}
              strokeWidth={strokeW}
              strokeOpacity={0.85}
            />
          );
        })}

        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={needleX}
          y2={needleY}
          stroke="#18181b"
          strokeWidth={1.5}
          strokeLinecap="round"
        />

        {/* Center dot */}
        <circle cx={cx} cy={cy} r={2.5} fill="#18181b" />
      </svg>

      {/* Label and value */}
      <div className="text-center">
        <div className="text-[9px] text-zinc-500 uppercase tracking-wide flex items-center justify-center gap-0.5">
          {label}
          {tooltip && <InfoTooltip content={tooltip} />}
        </div>
        {showValue && (
          <div className={`text-xs font-mono font-bold ${valueColor}`}>
            {value.toFixed(value >= 100 ? 0 : 1)}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Direction indicator - shows UP/DOWN/NEUTRAL with an arrow
 */
interface DirectionIndicatorProps {
  direction: "UP" | "DOWN" | "NEUTRAL";
  label: string;
  tooltip?: string;
}

function DirectionIndicator({ direction, label, tooltip }: DirectionIndicatorProps) {
  const color = direction === "UP" ? "text-green-500" : direction === "DOWN" ? "text-red-500" : "text-zinc-500";
  const bgColor = direction === "UP" ? "bg-green-500/10" : direction === "DOWN" ? "bg-red-500/10" : "bg-zinc-500/10";
  const arrow = direction === "UP" ? "▲" : direction === "DOWN" ? "▼" : "●";

  return (
    <div className={`flex flex-col items-center px-3 py-1.5 rounded-lg ${bgColor}`}>
      <div className={`text-base ${color}`}>{arrow}</div>
      <div className="text-[9px] text-zinc-500 uppercase tracking-wide flex items-center gap-0.5">
        {label}
        {tooltip && <InfoTooltip content={tooltip} />}
      </div>
      <div className={`text-[10px] font-medium ${color}`}>{direction}</div>
    </div>
  );
}

/**
 * Visual dial gauge display for technical indicators.
 * Shows RSI, ADX, VWAP, and Supertrend at a glance.
 */
function IndicatorGaugesComponent({ momentum, compact = false }: IndicatorGaugesProps) {
  if (!momentum) {
    return (
      <div className="text-zinc-500 text-xs text-center py-2">
        Loading indicators...
      </div>
    );
  }

  const rsi = momentum.rsi ?? 50;
  const adx = momentum.adx ?? 0;
  const vwapSignal = momentum.vwap_signal ?? "NEUTRAL";
  const supertrendDir = momentum.supertrend_direction ?? "NEUTRAL";

  // RSI color based on value
  const rsiColor = rsi < 30 ? "text-green-500" : rsi > 70 ? "text-red-500" : "text-amber-400";

  // ADX color based on strength
  const adxColor = adx < 20 ? "text-zinc-400" : adx < 40 ? "text-purple-400" : "text-purple-500";

  if (compact) {
    // Compact inline display for header/summary
    return (
      <div className="flex items-center gap-3 text-xs">
        {/* RSI Pill */}
        <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full ${
          rsi < 30 ? "bg-green-500/20" : rsi > 70 ? "bg-red-500/20" : "bg-zinc-500/20"
        }`}>
          <span className="text-zinc-500">RSI</span>
          <span className={`font-mono font-medium ${rsiColor}`}>{rsi.toFixed(0)}</span>
        </div>

        {/* ADX Pill */}
        <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full ${
          adx >= 25 ? "bg-purple-500/20" : "bg-zinc-500/20"
        }`}>
          <span className="text-zinc-500">ADX</span>
          <span className={`font-mono font-medium ${adxColor}`}>{adx.toFixed(0)}</span>
        </div>

        {/* Direction Arrows */}
        <div className="flex items-center gap-1">
          <span className={`text-lg ${
            vwapSignal === "UP" ? "text-green-500" :
            vwapSignal === "DOWN" ? "text-red-500" : "text-zinc-400"
          }`}>
            {vwapSignal === "UP" ? "▲" : vwapSignal === "DOWN" ? "▼" : "◆"}
          </span>
          <span className={`text-lg ${
            supertrendDir === "UP" ? "text-green-500" :
            supertrendDir === "DOWN" ? "text-red-500" : "text-zinc-400"
          }`}>
            {supertrendDir === "UP" ? "▲" : supertrendDir === "DOWN" ? "▼" : "◆"}
          </span>
        </div>
      </div>
    );
  }

  // Full dial gauge display - all 4 indicators on one line
  return (
    <div className="space-y-2">
      {/* All indicators in one row */}
      <div className="flex items-end justify-between gap-2">
        {/* RSI Gauge */}
        <DialGauge
          value={rsi}
          min={0}
          max={100}
          label="RSI"
          valueColor={rsiColor}
          size={56}
          zones={[
            { start: 0, end: 30, color: "#22c55e" },
            { start: 30, end: 70, color: "#f59e0b" },
            { start: 70, end: 100, color: "#ef4444" },
          ]}
          tooltip={`RSI (Relative Strength Index): ${rsi.toFixed(1)}\n${rsi < 30 ? "Oversold - potential reversal UP" : rsi > 70 ? "Overbought - potential reversal DOWN" : "Neutral zone"}`}
        />

        {/* ADX Gauge */}
        <DialGauge
          value={adx}
          min={0}
          max={60}
          label="ADX"
          valueColor={adxColor}
          size={56}
          zones={[
            { start: 0, end: 20, color: "#71717a" },
            { start: 20, end: 40, color: "#a855f7" },
            { start: 40, end: 60, color: "#9333ea" },
          ]}
          tooltip={`ADX (Average Directional Index): ${adx.toFixed(1)}\n${adx < 20 ? "Weak/No trend" : adx < 40 ? "Strong trend" : "Very strong trend"}`}
        />

        {/* VWAP Direction */}
        <DirectionIndicator
          direction={vwapSignal as "UP" | "DOWN" | "NEUTRAL"}
          label="VWAP"
          tooltip={`VWAP Signal: ${vwapSignal}\nPrice ${vwapSignal === "UP" ? "above" : vwapSignal === "DOWN" ? "below" : "near"} Volume Weighted Average Price`}
        />

        {/* Supertrend Direction */}
        <DirectionIndicator
          direction={supertrendDir as "UP" | "DOWN" | "NEUTRAL"}
          label="Supertrend"
          tooltip={`Supertrend: ${supertrendDir}\n${supertrendDir === "UP" ? "Bullish trend - support below price" : supertrendDir === "DOWN" ? "Bearish trend - resistance above price" : "No clear trend"}`}
        />
      </div>

      {/* Consensus Summary */}
      <div className="text-center text-[10px]">
        <SignalConsensus
          vwap={vwapSignal}
          rsi={rsi < 30 ? "UP" : rsi > 70 ? "DOWN" : "NEUTRAL"}
          adx={adx < 20 ? "weak" : adx < 40 ? "strong" : "extreme"}
          supertrend={supertrendDir}
        />
      </div>
    </div>
  );
}

interface SignalConsensusProps {
  vwap: string;
  rsi: string;
  adx: string;
  supertrend: string;
}

function SignalConsensus({ vwap, rsi, adx, supertrend }: SignalConsensusProps) {
  let bullishCount = 0;
  let bearishCount = 0;

  if (vwap === "UP") bullishCount++;
  else if (vwap === "DOWN") bearishCount++;

  if (rsi === "UP") bullishCount++;
  else if (rsi === "DOWN") bearishCount++;

  if (supertrend === "UP") bullishCount++;
  else if (supertrend === "DOWN") bearishCount++;

  const trendingStr = adx !== "weak" ? " (Trending)" : " (No Trend)";

  if (bullishCount >= 2 && bullishCount > bearishCount) {
    return (
      <span className="text-green-500 font-medium">
        Bullish Consensus ({bullishCount}/3){trendingStr}
      </span>
    );
  } else if (bearishCount >= 2 && bearishCount > bullishCount) {
    return (
      <span className="text-red-500 font-medium">
        Bearish Consensus ({bearishCount}/3){trendingStr}
      </span>
    );
  } else {
    return (
      <span className="text-zinc-500">
        Mixed Signals{trendingStr}
      </span>
    );
  }
}

export const IndicatorGauges = memo(IndicatorGaugesComponent);
