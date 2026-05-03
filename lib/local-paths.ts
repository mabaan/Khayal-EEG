import { promises as fs } from "node:fs";
import path from "node:path";

const ROOT = process.cwd();

export const paths = {
  root: ROOT,
  storageRoot: path.join(ROOT, "storage"),
  demoModels: path.join(ROOT, "storage", "demo_models"),
  demoEdfRoot: path.join(ROOT, "storage", "demo_edf"),
  baseModels: path.join(ROOT, "storage", "base_models"),
  profilesRoot: path.join(ROOT, "storage", "profiles"),
  sessionsRoot: path.join(ROOT, "storage", "sessions"),
  logsRoot: path.join(ROOT, "storage", "logs"),
  activeProfileRef: path.join(ROOT, "storage", "profiles", "active_profile.json")
};

export function profileDir(profileId: string): string {
  return path.join(paths.profilesRoot, profileId);
}

export function profileFile(profileId: string): string {
  return path.join(profileDir(profileId), "profile.json");
}

export function profileRawEdfDir(profileId: string): string {
  return path.join(profileDir(profileId), "raw_edf");
}

export function profileInferenceUploadDir(profileId: string, groupId: string): string {
  return path.join(profileRawEdfDir(profileId), "inference_uploads", groupId);
}

export function profileProcessedDir(profileId: string): string {
  return path.join(profileDir(profileId), "preprocessed_edf");
}

export function profileSegmentedDir(profileId: string): string {
  return path.join(profileDir(profileId), "segmented_windows");
}

export function profileModelsDir(profileId: string): string {
  return path.join(profileDir(profileId), "models");
}

export function profileTrainingDir(profileId: string): string {
  return path.join(profileDir(profileId), "training");
}

export function profileInferenceDir(profileId: string): string {
  return path.join(profileDir(profileId), "inference");
}

export function sessionTypeDir(type: "calibration" | "training" | "inference" | "simulation"): string {
  return path.join(paths.sessionsRoot, type);
}

export async function ensureDir(dir: string): Promise<void> {
  await fs.mkdir(dir, { recursive: true });
}

export async function ensureStorageLayout(): Promise<void> {
  await Promise.all([
    ensureDir(paths.baseModels),
    ensureDir(paths.profilesRoot),
    ensureDir(paths.sessionsRoot),
    ensureDir(paths.logsRoot),
    ensureDir(sessionTypeDir("calibration")),
    ensureDir(sessionTypeDir("training")),
    ensureDir(sessionTypeDir("inference")),
    ensureDir(sessionTypeDir("simulation"))
  ]);
}

export async function ensureProfileLayout(profileId: string): Promise<void> {
  await Promise.all([
    ensureDir(profileDir(profileId)),
    ensureDir(profileRawEdfDir(profileId)),
    ensureDir(profileProcessedDir(profileId)),
    ensureDir(profileSegmentedDir(profileId)),
    ensureDir(profileModelsDir(profileId)),
    ensureDir(profileTrainingDir(profileId)),
    ensureDir(profileInferenceDir(profileId))
  ]);
}
