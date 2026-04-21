import { NextRequest, NextResponse } from "next/server";
import { listSessionRecords } from "@/lib/history-store";

export async function GET(request: NextRequest) {
  const profileId = request.nextUrl.searchParams.get("profile_id") ?? undefined;
  const sessions = await listSessionRecords(profileId);
  return NextResponse.json({ sessions });
}
