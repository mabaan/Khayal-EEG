import { promises as fs } from "node:fs";
import path from "node:path";
import { BASE_MODEL_DEFAULT_PATH } from "@/lib/constants";
import {
  ensureProfileLayout,
  ensureStorageLayout,
  paths,
  profileFile,
  profileModelsDir
} from "@/lib/local-paths";
import type { ActiveProfileRef, ProfileManifest } from "@/lib/types";

function nowIso(): string {
  return new Date().toISOString();
}

function profileIdFromName(name: string): string {
  return `${name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${Date.now()}`;
}

async function readJsonFile<T>(filePath: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

async function writeJsonFile(filePath: string, value: unknown): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export async function listProfiles(): Promise<ProfileManifest[]> {
  await ensureStorageLayout();
  const items = await fs.readdir(paths.profilesRoot, { withFileTypes: true });
  const profiles: ProfileManifest[] = [];

  for (const item of items) {
    if (!item.isDirectory()) {
      continue;
    }
    const filePath = profileFile(item.name);
    try {
      const raw = await fs.readFile(filePath, "utf8");
      profiles.push(JSON.parse(raw) as ProfileManifest);
    } catch {
      // ignore malformed profile folders
    }
  }

  return profiles.sort((a, b) => b.updated_at.localeCompare(a.updated_at));
}

export async function readActiveProfileRef(): Promise<ActiveProfileRef> {
  await ensureStorageLayout();
  return readJsonFile<ActiveProfileRef>(paths.activeProfileRef, { profile_id: null });
}

export async function writeActiveProfileRef(profileId: string | null): Promise<void> {
  await writeJsonFile(paths.activeProfileRef, { profile_id: profileId });
}

export async function getProfile(profileId: string): Promise<ProfileManifest | null> {
  const filePath = profileFile(profileId);
  return readJsonFile<ProfileManifest | null>(filePath, null);
}

export async function getActiveProfile(): Promise<ProfileManifest | null> {
  const active = await readActiveProfileRef();
  if (!active.profile_id) {
    return null;
  }
  return getProfile(active.profile_id);
}

export async function createProfile(name: string): Promise<ProfileManifest> {
  await ensureStorageLayout();
  const profileId = profileIdFromName(name);
  await ensureProfileLayout(profileId);

  const profile: ProfileManifest = {
    id: profileId,
    name: name.trim(),
    created_at: nowIso(),
    updated_at: nowIso(),
    base_model_path: BASE_MODEL_DEFAULT_PATH,
    user_model_path: null,
    status: "needs_calibration",
    model_ready: false,
    notes: ""
  };

  await writeJsonFile(profileFile(profileId), profile);
  await writeActiveProfileRef(profileId);

  return profile;
}

export async function setActiveProfile(profileId: string): Promise<ProfileManifest> {
  const profile = await getProfile(profileId);
  if (!profile) {
    throw new Error("Profile not found.");
  }
  await writeActiveProfileRef(profileId);
  return profile;
}

export async function updateProfile(profileId: string, patch: Partial<ProfileManifest>): Promise<ProfileManifest> {
  const profile = await getProfile(profileId);
  if (!profile) {
    throw new Error("Profile not found.");
  }

  const nextProfile: ProfileManifest = {
    ...profile,
    ...patch,
    id: profile.id,
    updated_at: nowIso()
  };

  await writeJsonFile(profileFile(profileId), nextProfile);
  return nextProfile;
}

export async function setModelReady(profileId: string, userModelPath: string): Promise<ProfileManifest> {
  return updateProfile(profileId, {
    user_model_path: userModelPath,
    model_ready: true,
    status: "ready"
  });
}

export function defaultUserModelPath(profileId: string): string {
  return path.join(profileModelsDir(profileId), "diff_e_user.pt");
}
