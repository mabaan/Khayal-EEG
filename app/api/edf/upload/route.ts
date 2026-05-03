import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { createSessionRecord } from "@/lib/history-store";
import { getActiveProfile } from "@/lib/profile-store";
import { ensureDir, ensureProfileLayout, profileInferenceUploadDir, profileRawEdfDir } from "@/lib/local-paths";
import { isCsvFileName, isEdfFileName } from "@/lib/validators";

export async function POST(request: NextRequest) {
  try {
    const profile = await getActiveProfile();
    if (!profile) {
      return NextResponse.json({ error: "No active profile." }, { status: 400 });
    }

    const form = await request.formData();
    const file = form.get("file");
    const purpose = (form.get("purpose")?.toString() ?? "inference") as "calibration" | "inference";
    const fileRole = (form.get("file_role")?.toString() ?? "edf") as "edf" | "marker";
    const groupId = form.get("group_id")?.toString().trim() || randomUUID();

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "Missing file field." }, { status: 400 });
    }

    if (fileRole === "edf" && !isEdfFileName(file.name)) {
      return NextResponse.json({ error: "Only EDF files are accepted for EEG uploads." }, { status: 400 });
    }
    if (fileRole === "marker" && !isCsvFileName(file.name)) {
      return NextResponse.json({ error: "Only CSV files are accepted for marker uploads." }, { status: 400 });
    }
    if (purpose === "calibration" && fileRole !== "edf") {
      return NextResponse.json({ error: "Calibration uploads only support EDF files." }, { status: 400 });
    }

    await ensureProfileLayout(profile.id);

    const saveDir =
      purpose === "calibration"
        ? profileRawEdfDir(profile.id)
        : profileInferenceUploadDir(profile.id, groupId);
    await ensureDir(saveDir);

    const savedName = `${Date.now()}-${file.name.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
    const savePath = path.join(saveDir, savedName);
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
      filename: file.name,
      group_id: groupId,
      role: fileRole,
      purpose,
      profile_id: profile.id
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "EDF upload failed." },
      { status: 500 }
    );
  }
}
