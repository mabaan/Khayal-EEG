import type { SimulatedSignalPoint, SimulationSnapshot } from "@/lib/types";

const CHANNEL_NAMES = [
  "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
];

function pseudoNoise(seed: number): number {
  const x = Math.sin(seed * 91.17) * 43758.5453;
  return x - Math.floor(x);
}

export function buildSignalPoints(count = 120): SimulatedSignalPoint[] {
  const points: SimulatedSignalPoint[] = [];
  for (let i = 0; i < count; i += 1) {
    const values = CHANNEL_NAMES.map((_, index) => {
      const base = Math.sin((i + index * 7) / 10) * 30;
      const noise = (pseudoNoise(i * (index + 1)) - 0.5) * 12;
      return Number((base + noise).toFixed(2));
    });
    points.push({ t: i, values });
  }
  return points;
}

export function makeSimulationSnapshot(step: string, status: SimulationSnapshot["signal_status"]): SimulationSnapshot {
  return {
    signal_status: status,
    current_step: step,
    channel_names: CHANNEL_NAMES,
    points: buildSignalPoints(140)
  };
}
