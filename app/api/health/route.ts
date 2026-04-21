import { NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import { PYTHON_SERVICE_URL } from "@/lib/constants";
import { ensureStorageLayout, paths } from "@/lib/local-paths";

export async function GET() {
  await ensureStorageLayout();

  const storageOk = await fs.access(paths.storageRoot).then(() => true).catch(() => false);

  let pythonOk = false;
  try {
    const response = await fetch(`${PYTHON_SERVICE_URL}/health`, { method: "GET" });
    pythonOk = response.ok;
  } catch {
    pythonOk = false;
  }

  return NextResponse.json({
    status: storageOk && pythonOk ? "ok" : "degraded",
    storage_ok: storageOk,
    python_service_ok: pythonOk,
    python_service_url: PYTHON_SERVICE_URL
  });
}
