import { promises as fs } from "node:fs";
import path from "node:path";
import { ensureDir, ensureStorageLayout, profileInferenceDir, profileTrainingDir, sessionTypeDir } from "@/lib/local-paths";
import type { SessionRecord, SessionType } from "@/lib/types";

function nowIso(): string {
  return new Date().toISOString();
}

function sessionId(type: SessionType): string {
  return `${type}-${Date.now()}`;
}

async function writeJson(filePath: string, value: unknown): Promise<void> {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function readSessionFile(filePath: string): Promise<SessionRecord | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as SessionRecord;
  } catch {
    return null;
  }
}

export async function createSessionRecord(
  partial: Omit<SessionRecord, "session_id" | "created_at">
): Promise<SessionRecord> {
  await ensureStorageLayout();
  const record: SessionRecord = {
    ...partial,
    session_id: sessionId(partial.type),
    created_at: nowIso()
  };

  const flatSessionPath = path.join(sessionTypeDir(partial.type), `${record.session_id}.json`);
  await writeJson(flatSessionPath, record);

  if (partial.type === "training") {
    await writeJson(path.join(profileTrainingDir(partial.profile_id), "train_manifest.json"), record);
  }

  if (partial.type === "inference" || partial.type === "simulation") {
    await writeJson(path.join(profileInferenceDir(partial.profile_id), "inference_manifest.json"), record);
  }

  return record;
}

export async function listSessionRecords(profileId?: string): Promise<SessionRecord[]> {
  await ensureStorageLayout();
  const sessionTypes: SessionType[] = ["calibration", "training", "inference", "simulation"];
  const records: SessionRecord[] = [];

  for (const type of sessionTypes) {
    const folder = sessionTypeDir(type);
    const entries = await fs.readdir(folder, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith(".json")) {
        continue;
      }
      const session = await readSessionFile(path.join(folder, entry.name));
      if (!session) {
        continue;
      }
      if (profileId && session.profile_id !== profileId) {
        continue;
      }
      records.push(session);
    }
  }

  return records.sort((a, b) => b.created_at.localeCompare(a.created_at));
}
