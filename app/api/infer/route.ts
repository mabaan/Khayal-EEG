import { NextRequest, NextResponse } from "next/server";
import { createSessionRecord } from "@/lib/history-store";
import { PYTHON_SERVICE_URL, READINESS_ERROR_MESSAGE } from "@/lib/constants";
import { getActiveProfile } from "@/lib/profile-store";

export async function POST(request: NextRequest) {
  const profile = await getActiveProfile();
  if (!profile) {
    return NextResponse.json({ error: "No active profile." }, { status: 400 });
  }

  if (!profile.model_ready || !profile.user_model_path) {
    return NextResponse.json({ error: READINESS_ERROR_MESSAGE }, { status: 400 });
  }

  try {
    const payload = (await request.json()) as { edf_path?: string; simulated?: boolean };
    const edfPath = payload.edf_path?.trim() ?? "";

    if (!edfPath) {
      return NextResponse.json({ error: "edf_path is required." }, { status: 400 });
    }

    const response = await fetch(`${PYTHON_SERVICE_URL}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile_id: profile.id,
        user_model_path: profile.user_model_path,
        edf_path: edfPath,
        simulated: Boolean(payload.simulated)
      })
    });

    const result = (await response.json()) as {
      status?: "success" | "failed";
      message?: string;
      final_sentence?: string;
      candidates?: unknown[];
      selected_sentence_id?: number;
      stage1_posteriors?: unknown[];
      used_fallback?: boolean;
      session_id?: string;
    };

    if (!response.ok || result.status !== "success") {
      await createSessionRecord({
        profile_id: profile.id,
        type: "inference",
        status: "failed",
        signal_status: "error",
        source: payload.simulated ? "simulated" : "edf_upload",
        input_path: edfPath,
        details: result
      });
      return NextResponse.json({ error: result.message ?? "Inference failed." }, { status: 500 });
    }

    const session = await createSessionRecord({
      profile_id: profile.id,
      type: "inference",
      status: "success",
      signal_status: "complete",
      source: payload.simulated ? "simulated" : "edf_upload",
      input_path: edfPath,
      predicted_sentence: result.final_sentence,
      details: {
        python_session_id: result.session_id,
        selected_sentence_id: result.selected_sentence_id,
        used_fallback: result.used_fallback
      }
    });

    return NextResponse.json({
      profile_id: profile.id,
      session_id: session.session_id,
      final_sentence: result.final_sentence,
      candidates: result.candidates ?? [],
      selected_sentence_id: result.selected_sentence_id ?? 0,
      stage1_posteriors: result.stage1_posteriors ?? [],
      used_fallback: Boolean(result.used_fallback),
      status: "success",
      message: result.message ?? "Inference complete."
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Inference route failed." },
      { status: 500 }
    );
  }
}
