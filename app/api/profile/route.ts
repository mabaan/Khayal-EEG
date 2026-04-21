import { NextRequest, NextResponse } from "next/server";
import { createProfile, getActiveProfile, listProfiles, setActiveProfile } from "@/lib/profile-store";
import { isSafeProfileName } from "@/lib/validators";

export async function GET() {
  const [profiles, active] = await Promise.all([listProfiles(), getActiveProfile()]);
  return NextResponse.json({ profiles, active_profile: active?.id ?? null });
}

export async function POST(request: NextRequest) {
  try {
    const payload = (await request.json()) as {
      action?: "create" | "select";
      name?: string;
      profile_id?: string;
    };

    if (payload.action === "create") {
      if (!payload.name || !isSafeProfileName(payload.name)) {
        return NextResponse.json({ error: "Invalid profile name." }, { status: 400 });
      }
      const profile = await createProfile(payload.name);
      return NextResponse.json({ profile });
    }

    if (payload.action === "select") {
      if (!payload.profile_id) {
        return NextResponse.json({ error: "profile_id is required." }, { status: 400 });
      }
      const profile = await setActiveProfile(payload.profile_id);
      return NextResponse.json({ profile });
    }

    return NextResponse.json({ error: "Unsupported profile action." }, { status: 400 });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Profile route failed." },
      { status: 500 }
    );
  }
}
