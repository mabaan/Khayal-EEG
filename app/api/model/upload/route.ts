import { promises as fs } from "node:fs";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { PYTHON_SERVICE_URL } from "@/lib/constants";
import { ensureDir, ensureProfileLayout, profileModelsDir } from "@/lib/local-paths";
import { getActiveProfile, setModelReady } from "@/lib/profile-store";
import { isModelFileName } from "@/lib/validators";
import type { ModelInfo } from "@/lib/types";

export async function POST(request: NextRequest) {
  try {
    const profile = await getActiveProfile();
    if (!profile) {
      return NextResponse.json({ error: "No active profile." }, { status: 400 });
    }

    const form = await request.formData();
    const file = form.get("file");
    if (!(file instanceof File)) {
      return NextResponse.json({ error: "Missing file field." }, { status: 400 });
    }
    if (!isModelFileName(file.name)) {
      return NextResponse.json({ error: "Only .pt or .pth checkpoints are accepted." }, { status: 400 });
    }

    await ensureProfileLayout(profile.id);
    const uploadDir = path.join(profileModelsDir(profile.id), "uploads");
    await ensureDir(uploadDir);
    const savedName = `${Date.now()}-${file.name.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
    const savePath = path.join(uploadDir, savedName);
    await fs.writeFile(savePath, Buffer.from(await file.arrayBuffer()));

    const response = await fetch(`${PYTHON_SERVICE_URL}/validate-model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_path: savePath })
    });
    const payload = (await response.json()) as { status?: string; message?: string; model?: ModelInfo | null };
    if (!response.ok || payload.status !== "success" || !payload.model) {
      await fs.unlink(savePath).catch(() => undefined);
      return NextResponse.json({ error: payload.message ?? "Model validation failed." }, { status: 400 });
    }

    const updatedProfile = await setModelReady(profile.id, savePath);
    return NextResponse.json({
      saved_path: savePath,
      model: payload.model,
      profile: updatedProfile
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Model upload failed." },
      { status: 500 }
    );
  }
}
