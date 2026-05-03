import { NextRequest, NextResponse } from "next/server";
import { createSessionRecord } from "@/lib/history-store";
import { PYTHON_SERVICE_URL, READINESS_ERROR_MESSAGE } from "@/lib/constants";
import { getActiveProfile } from "@/lib/profile-store";
import type { InferenceResult, Stage2Mode } from "@/lib/types";

export async function POST(request: NextRequest) {
  const profile = await getActiveProfile();
  if (!profile) {
    return NextResponse.json({ error: "No active profile." }, { status: 400 });
  }

  if (!profile.model_ready || !profile.user_model_path) {
    return NextResponse.json({ error: READINESS_ERROR_MESSAGE }, { status: 400 });
  }

  try {
    const payload = (await request.json()) as {
      edf_path?: string;
      marker_csv_path?: string;
      top_k_words?: number;
      retrieval_topk?: number;
      stage2_mode?: Stage2Mode;
      use_demo_files?: boolean;
    };

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
        marker_csv_path: payload.marker_csv_path?.trim() || undefined,
        top_k_words: payload.top_k_words ?? 8,
        retrieval_topk: payload.retrieval_topk ?? 5,
        stage2_mode: payload.stage2_mode ?? "qwen",
        use_demo_files: Boolean(payload.use_demo_files)
      })
    });

    const result = (await response.json()) as InferenceResult;

    if (!response.ok || result.status !== "success") {
      await createSessionRecord({
        profile_id: profile.id,
        type: "inference",
        status: "failed",
        signal_status: "error",
        source: payload.use_demo_files ? "demo" : "edf_upload",
        input_path: edfPath,
        details: { result }
      });
      return NextResponse.json({ error: result.message ?? "Inference failed." }, { status: 500 });
    }

    const session = await createSessionRecord({
      profile_id: profile.id,
      type: "inference",
      status: "success",
      signal_status: "complete",
      source: payload.use_demo_files ? "demo" : "edf_upload",
      input_path: edfPath,
      predicted_sentence: result.prediction.arabic,
      details: {
        python_session_id: result.session_id,
        sentence_id: result.prediction.sentence_id,
        stage2_mode: result.stage2.mode,
        used_fallback: result.stage2.used_fallback,
        warning_count: result.preprocessing.warnings.length + result.stage2.warnings.length
      }
    });

    return NextResponse.json({
      ...result,
      history_session_id: session.session_id
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Inference route failed." },
      { status: 500 }
    );
  }
}
