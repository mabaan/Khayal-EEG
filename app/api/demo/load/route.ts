import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { PYTHON_SERVICE_URL } from "@/lib/constants";
import { paths } from "@/lib/local-paths";
import { getActiveProfile, setModelReady } from "@/lib/profile-store";
import type { ModelInfo } from "@/lib/types";

const DEMO_MODEL_PATH = path.join(paths.demoModels, "S5_classifier_A.pt");
const DEMO_EDF_PATH = path.join(
  paths.demoEdfRoot,
  "S5_C7_T2",
  "Sentence_7_EPOCX_570600_2025.11.12T04.15.10.08.00.edf"
);
const DEMO_MARKER_PATH = path.join(
  paths.demoEdfRoot,
  "S5_C7_T2",
  "Sentence_7_EPOCX_570600_2025.11.12T04.15.10.08.00_intervalMarker.csv"
);

export async function POST() {
  try {
    const profile = await getActiveProfile();
    if (!profile) {
      return NextResponse.json({ error: "No active profile." }, { status: 400 });
    }

    const requiredPaths = [DEMO_MODEL_PATH, DEMO_EDF_PATH, DEMO_MARKER_PATH];
    for (const requiredPath of requiredPaths) {
      const exists = await fs.access(requiredPath).then(() => true).catch(() => false);
      if (!exists) {
        return NextResponse.json({ error: `Missing bundled demo asset: ${requiredPath}` }, { status: 500 });
      }
    }

    const response = await fetch(`${PYTHON_SERVICE_URL}/validate-model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_path: DEMO_MODEL_PATH })
    });
    const payload = (await response.json()) as { status?: string; message?: string; model?: ModelInfo | null };
    if (!response.ok || payload.status !== "success" || !payload.model) {
      return NextResponse.json({ error: payload.message ?? "Bundled demo model validation failed." }, { status: 500 });
    }

    const updatedProfile = await setModelReady(profile.id, DEMO_MODEL_PATH);
    return NextResponse.json({
      profile: updatedProfile,
      model: payload.model,
      model_path: DEMO_MODEL_PATH,
      edf_path: DEMO_EDF_PATH,
      marker_csv_path: DEMO_MARKER_PATH
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Demo load failed." },
      { status: 500 }
    );
  }
}
