import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import { getActiveProfile, updateProfile } from "@/lib/profile-store";
import { isPtFilePath } from "@/lib/validators";

export async function POST(request: NextRequest) {
  try {
    const profile = await getActiveProfile();
    if (!profile) {
      return NextResponse.json({ error: "No active profile." }, { status: 400 });
    }

    const payload = (await request.json()) as { checkpoint_path?: string };
    const checkpointPath = payload.checkpoint_path?.trim() ?? "";

    if (!checkpointPath || !isPtFilePath(checkpointPath)) {
      return NextResponse.json({ error: "checkpoint_path must be a .pt file." }, { status: 400 });
    }

    const exists = await fs.access(checkpointPath).then(() => true).catch(() => false);
    if (!exists) {
      return NextResponse.json({ error: "Checkpoint path does not exist locally." }, { status: 400 });
    }

    const updated = await updateProfile(profile.id, {
      base_model_path: checkpointPath
    });

    return NextResponse.json({ profile: updated });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Model load route failed." },
      { status: 500 }
    );
  }
}
