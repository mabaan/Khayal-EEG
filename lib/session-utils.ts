import type { ProfileManifest, SignalStatus } from "@/lib/types";

export function modelReadyLabel(profile: ProfileManifest | null): string {
  if (!profile) {
    return "No Active Profile";
  }
  return profile.model_ready ? "Model Ready" : "Needs Training";
}

export function signalStatusLabel(status: SignalStatus): string {
  switch (status) {
    case "idle":
      return "Idle";
    case "uploading":
      return "Uploading";
    case "preprocessing":
      return "Preprocessing";
    case "segmenting":
      return "Segmenting";
    case "running":
      return "Running";
    case "complete":
      return "Complete";
    case "error":
      return "Error";
    default:
      return "Unknown";
  }
}

export function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(date);
}
