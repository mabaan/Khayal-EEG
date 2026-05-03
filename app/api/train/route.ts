import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import path from "node:path";
import { createSessionRecord } from "@/lib/history-store";
import { PYTHON_SERVICE_URL } from "@/lib/constants";
import { getActiveProfile, setModelReady, updateProfile } from "@/lib/profile-store";
import { profileRawEdfDir } from "@/lib/local-paths";

async function listProfileEdfs(profileId: string): Promise<string[]> {
  const folder = profileRawEdfDir(profileId);
  const entries = await fs.readdir(folder, { withFileTypes: true }).catch(() => []);
  return entries
    .filter((entry: { isFile: () => boolean; name: string }) => entry.isFile() && entry.name.toLowerCase().endsWith(".edf"))
    .map((entry: { name: string }) => path.join(folder, entry.name));
}

export async function POST(request: NextRequest) {
  const active = await getActiveProfile();
  if (!active) {
    return NextResponse.json({ error: "No active profile." }, { status: 400 });
  }

  try {
    const payload = (await request.json()) as { calibration_edf_paths?: string[] };
    const edfPaths = payload.calibration_edf_paths?.filter(Boolean) ?? [];
    const calibrationPaths = edfPaths.length > 0 ? edfPaths : await listProfileEdfs(active.id);

    if (calibrationPaths.length === 0) {
      return NextResponse.json({ error: "No calibration EDF files found for training." }, { status: 400 });
    }

    await updateProfile(active.id, { status: "training", model_ready: false });

    const response = await fetch(`${PYTHON_SERVICE_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile_id: active.id,
        base_model_path: active.base_model_path,
        calibration_edf_paths: calibrationPaths
      })
    });

    const result = (await response.json()) as {
      status?: "success" | "failed";
      message?: string;
      model_path?: string;
      metrics_path?: string;
      session_id?: string;
    };

    if (!response.ok || result.status !== "success" || !result.model_path) {
      await updateProfile(active.id, { status: "needs_calibration", model_ready: false });
      await createSessionRecord({
        profile_id: active.id,
        type: "training",
        status: "failed",
        signal_status: "error",
        source: "edf_upload",
        details: result
      });
      return NextResponse.json({ error: result.message ?? "Training failed." }, { status: 500 });
    }

    const profile = await setModelReady(active.id, result.model_path);

    const session = await createSessionRecord({
      profile_id: active.id,
      type: "training",
      status: "success",
      signal_status: "complete",
      source: "edf_upload",
      output_path: result.model_path,
      details: {
        metrics_path: result.metrics_path,
        python_session_id: result.session_id
      }
    });

    return NextResponse.json({
      profile_id: active.id,
      session_id: session.session_id,
      model_path: result.model_path,
      metrics_path: result.metrics_path ?? "",
      status: "success",
      message: result.message ?? "Training completed.",
      profile
    });
  } catch (error) {
    await updateProfile(active.id, { status: "needs_calibration", model_ready: false }).catch(() => {
      // keep best effort rollback
    });
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Training route failed." },
      { status: 500 }
    );
  }
}
