import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import path from "node:path";
import { createSessionRecord } from "@/lib/history-store";
import { getActiveProfile } from "@/lib/profile-store";
import { ensureProfileLayout, profileRawEdfDir } from "@/lib/local-paths";
import { isEdfFileName } from "@/lib/validators";

export async function POST(request: NextRequest) {
  try {
    const profile = await getActiveProfile();
    if (!profile) {
      return NextResponse.json({ error: "No active profile." }, { status: 400 });
    }

    const form = await request.formData();
    const file = form.get("file");
    const purpose = (form.get("purpose")?.toString() ?? "inference") as "calibration" | "inference";

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "Missing file field." }, { status: 400 });
    }

    if (!isEdfFileName(file.name)) {
      return NextResponse.json({ error: "Only EDF files are accepted." }, { status: 400 });
    }

    await ensureProfileLayout(profile.id);

    const savedName = `${Date.now()}-${file.name.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
    const savePath = path.join(profileRawEdfDir(profile.id), savedName);
    const bytes = Buffer.from(await file.arrayBuffer());
    await fs.writeFile(savePath, bytes);

    if (purpose === "calibration") {
      await createSessionRecord({
        profile_id: profile.id,
        type: "calibration",
        status: "success",
        signal_status: "complete",
        source: "edf_upload",
        input_path: savePath,
        details: { upload_only: true }
      });
    }

    return NextResponse.json({
      saved_path: savePath,
      profile_id: profile.id,
      purpose
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "EDF upload failed." },
      { status: 500 }
    );
  }
}
