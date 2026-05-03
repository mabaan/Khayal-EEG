import type {
  DemoLoadResult,
  InferenceResult,
  InferenceRequest,
  ProfileManifest,
  SessionRecord,
  SimulationSnapshot,
  TrainingResult,
  UploadEdfResult,
  UploadModelResult
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

export async function runInference(payload: InferenceRequest): Promise<InferenceResult> {
  const response = await fetch("/api/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parseResponse(response);
}

export async function uploadModelFile(file: File): Promise<UploadModelResult> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/model/upload", { method: "POST", body: form });
  return parseResponse(response);
}

export async function uploadEegAsset(
  file: File,
  role: "edf" | "marker",
  purpose: "calibration" | "inference" = "inference",
  groupId?: string
): Promise<UploadEdfResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("file_role", role);
  form.append("purpose", purpose);
  if (groupId) {
    form.append("group_id", groupId);
  }
  const response = await fetch("/api/edf/upload", { method: "POST", body: form });
  return parseResponse(response);
}

export async function loadDemoCase(): Promise<DemoLoadResult> {
  const response = await fetch("/api/demo/load", { method: "POST" });
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
