import { NextResponse } from "next/server";
import { createSessionRecord } from "@/lib/history-store";
import { PYTHON_SERVICE_URL } from "@/lib/constants";
import { getActiveProfile } from "@/lib/profile-store";

export async function POST() {
  const profile = await getActiveProfile();
  if (!profile) {
    return NextResponse.json({ error: "No active profile." }, { status: 400 });
  }

  try {
    const response = await fetch(`${PYTHON_SERVICE_URL}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile_id: profile.id,
        user_model_path: profile.user_model_path,
        model_ready: profile.model_ready
      })
    });

    const payload = (await response.json()) as {
      simulation?: unknown;
      inference_result?: {
        final_sentence?: string;
      };
      status?: string;
      message?: string;
    };

    if (!response.ok || payload.status === "failed") {
      await createSessionRecord({
        profile_id: profile.id,
        type: "simulation",
        status: "failed",
        signal_status: "error",
        source: "simulated",
        details: payload
      });
      return NextResponse.json({ error: payload.message ?? "Simulation failed." }, { status: 500 });
    }

    await createSessionRecord({
      profile_id: profile.id,
      type: "simulation",
      status: "success",
      signal_status: "complete",
      source: "simulated",
      predicted_sentence: payload.inference_result?.final_sentence,
      details: payload
    });

    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Simulation route failed." },
      { status: 500 }
    );
  }
}
