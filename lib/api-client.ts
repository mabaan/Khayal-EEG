import type {
  InferenceResult,
  ProfileManifest,
  SessionRecord,
  SimulationSnapshot,
  TrainingResult
} from "@/lib/types";

async function parseResponse<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as { error?: string } & T;
  if (!response.ok) {
    throw new Error(payload.error ?? "Request failed.");
  }
  return payload;
}

export async function fetchProfiles(): Promise<{ active_profile: string | null; profiles: ProfileManifest[] }> {
  const response = await fetch("/api/profile", { method: "GET", cache: "no-store" });
  return parseResponse(response);
}

export async function createProfile(name: string): Promise<{ profile: ProfileManifest }> {
  const response = await fetch("/api/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "create", name })
  });
  return parseResponse(response);
}

export async function selectProfile(profileId: string): Promise<{ profile: ProfileManifest }> {
  const response = await fetch("/api/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "select", profile_id: profileId })
  });
  return parseResponse(response);
}

export async function updateBaseModelPath(path: string): Promise<{ profile: ProfileManifest }> {
  const response = await fetch("/api/model/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint_path: path })
  });
  return parseResponse(response);
}

export async function startTraining(calibrationEdfPaths: string[]): Promise<TrainingResult> {
  const response = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ calibration_edf_paths: calibrationEdfPaths })
  });
  return parseResponse(response);
}

export async function runInference(edfPath: string, simulated = false): Promise<InferenceResult> {
  const response = await fetch("/api/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ edf_path: edfPath, simulated })
  });
  return parseResponse(response);
}

export async function fetchHistory(): Promise<{ sessions: SessionRecord[] }> {
  const response = await fetch("/api/history", { method: "GET", cache: "no-store" });
  return parseResponse(response);
}

export async function fetchSimulation(): Promise<{
  simulation: SimulationSnapshot;
  inference_result?: InferenceResult;
  status: string;
  message: string;
}> {
  const response = await fetch("/api/simulate", { method: "POST" });
  return parseResponse(response);
}
