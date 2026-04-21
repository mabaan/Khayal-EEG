"use client";

import { useMemo } from "react";
import type { SimulationSnapshot } from "@/lib/types";

interface EegWavePanelProps {
  snapshot: SimulationSnapshot | null;
}

export function EegWavePanel({ snapshot }: EegWavePanelProps) {
  const preview = useMemo(() => {
    if (!snapshot || snapshot.points.length === 0) {
      return [] as string[];
    }

    const channelIndices = [0, 3, 6, 9, 12];
    return channelIndices.map((channelIndex, lineIndex) => {
      return snapshot.points
        .map((point, index) => {
          const x = (index / Math.max(snapshot.points.length - 1, 1)) * 100;
          const centered = 20 + lineIndex * 24;
          const y = centered - point.values[channelIndex] * 0.13;
          return `${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(" ");
    });
  }, [snapshot]);

  return (
    <section className="kh-panel p-4 md:p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="kh-kicker">Live Signals</p>
          <h3 className="mt-1 text-base font-bold text-slate-900">Session Wave Monitor</h3>
        </div>
        <span className="kh-chip">Simulated 14-channel EEG</span>
      </div>

      <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-950 p-2">
        <svg viewBox="0 0 100 120" className="h-72 w-full" preserveAspectRatio="none" aria-label="simulated eeg waveform">
          <defs>
            <pattern id="kh-grid" width="5" height="5" patternUnits="userSpaceOnUse">
              <path d="M 5 0 L 0 0 0 5" fill="none" stroke="#1e293b" strokeWidth="0.12" />
            </pattern>
          </defs>
          <rect x="0" y="0" width="100" height="120" fill="#020617" />
          <rect x="0" y="0" width="100" height="120" fill="url(#kh-grid)" opacity="0.5" />

          {preview.map((points, index) => (
            <polyline
              key={index}
              points={points}
              fill="none"
              stroke={index % 2 === 0 ? "#2dd4bf" : "#67e8f9"}
              strokeWidth="0.55"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
        </svg>
      </div>
    </section>
  );
}
