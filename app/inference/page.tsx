"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { DebugDetailsPanel } from "@/components/debug-details-panel";
import { EdfFileCard } from "@/components/edf-file-card";
import { FinalPredictionCard } from "@/components/final-prediction-card";
import { InferenceTimeline } from "@/components/inference-timeline";
import { ModelFileCard } from "@/components/model-file-card";
import { SessionTable } from "@/components/session-table";
import { Stage1EvidencePanel } from "@/components/stage1-evidence-panel";
import { Stage2CandidatesPanel } from "@/components/stage2-candidates-panel";
import { StatusBadge } from "@/components/status-badge";
import {
  fetchHistory,
  fetchProfiles,
  runInference,
  uploadEegAsset,
  uploadModelFile
} from "@/lib/api-client";
import { READINESS_ERROR_MESSAGE, STAGE2_TOP_K } from "@/lib/constants";
import type {
  InferenceResult,
  ModelInfo,
  ProfileManifest,
  SessionRecord,
  Stage2Mode,
  TimelineStep
} from "@/lib/types";

type UploadAssetState = {
  path: string;
  filename: string;
  groupId: string;
  source: "upload";
  status: "loaded" | "uploaded" | "validated";
};

type HealthPayload = {
  python_service_ok?: boolean;
  python_service?: {
    message?: string;
    ollama_available?: boolean;
    transformers_available?: boolean;
    cuda_available?: boolean;
    supported_stage2_modes?: string[];
  } | null;
};

const OPTIMISTIC_STEPS: Array<Pick<TimelineStep, "id" | "label">> = [
  { id: "model_validated", label: "Model validated" },
  { id: "edf_loaded", label: "EDF loaded" },
  { id: "marker_detected", label: "Marker CSV detected" },
  { id: "preprocessing", label: "Preprocessing EEG" },
  { id: "segmenting", label: "Segmenting 3 imagination windows" },
  { id: "building_tensors", label: "Building 19-channel tensors" },
  { id: "stage1", label: "Running Stage 1 DiffE word classifier" },
  { id: "posterior_evidence", label: "Building word evidence" },
  { id: "retrieval", label: "Building sentence shortlist" },
  { id: "reranking", label: "Running Qwen/Ollama sentence selection" },
  { id: "decoded", label: "Final sentence decoded" }
];

function buildOptimisticTimeline(): TimelineStep[] {
  return OPTIMISTIC_STEPS.map((step) => ({
    ...step,
    status: "pending",
    detail: null,
    warnings: []
  }));
}

function pathFileName(filePath: string | null | undefined): string {
  if (!filePath) {
    return "No file selected";
  }
  return filePath.split(/[\\/]/).pop() ?? filePath;
}

function profileBackedModel(profile: ProfileManifest | null): ModelInfo | null {
  if (!profile?.user_model_path) {
    return null;
  }
  return {
    path: profile.user_model_path,
    filename: pathFileName(profile.user_model_path),
    arch: "diffe",
    subject: null,
    classifier: null,
    n_classes: 25,
    device: "cpu",
    window_size: 352,
    validated: profile.model_ready
  };
}

export default function InferencePage() {
  const [activeProfile, setActiveProfile] = useState<ProfileManifest | null>(null);
  const [history, setHistory] = useState<SessionRecord[]>([]);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [edfAsset, setEdfAsset] = useState<UploadAssetState | null>(null);
  const [markerAsset, setMarkerAsset] = useState<UploadAssetState | null>(null);
  const stage2Mode: Stage2Mode = "qwen";
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [timeline, setTimeline] = useState<TimelineStep[]>(buildOptimisticTimeline());
  const [pendingAction, setPendingAction] = useState<"model" | "edf" | "marker" | "infer" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const optimisticTimer = useRef<number | null>(null);

  async function refreshHistory() {
    const payload = await fetchHistory();
    setHistory(payload.sessions.slice(0, 6));
  }

  async function refreshProfileContext() {
    const payload = await fetchProfiles();
    const active = payload.profiles.find((item) => item.id === payload.active_profile) ?? null;
    setActiveProfile(active);
    setModel((current: ModelInfo | null) => current ?? profileBackedModel(active));
  }

  async function refreshHealth() {
    const response = await fetch("/api/health", { cache: "no-store" });
    const payload = (await response.json()) as HealthPayload;
    setHealth(payload);
  }

  useEffect(() => {
    Promise.all([refreshProfileContext(), refreshHistory(), refreshHealth()]).catch((err) => {
      setError(err instanceof Error ? err.message : "Failed to load inference context.");
    });
  }, []);

  useEffect(() => {
    return () => {
      if (optimisticTimer.current) {
        window.clearInterval(optimisticTimer.current);
      }
    };
  }, []);

  function startOptimisticTimeline() {
    if (optimisticTimer.current) {
      window.clearInterval(optimisticTimer.current);
    }
    setTimeline(() =>
      buildOptimisticTimeline().map((step, index) => {
        if (index === 0) {
          return { ...step, status: "complete", detail: "Validated local checkpoint", warnings: [] };
        }
        if (index === 1) {
          return { ...step, status: "running", detail: "Opening local EDF recording", warnings: [] };
        }
        return step;
      })
    );

    let index = 1;
    optimisticTimer.current = window.setInterval(() => {
      setTimeline((previous: TimelineStep[]) =>
        previous.map((step: TimelineStep, currentIndex: number) => {
          if (currentIndex < index) {
            return step.status === "running" ? { ...step, status: "complete" } : step;
          }
          if (currentIndex === index) {
            return step.status === "pending" ? { ...step, status: "running" } : step;
          }
          return step;
        })
      );
      index += 1;
      if (index >= OPTIMISTIC_STEPS.length) {
        if (optimisticTimer.current) {
          window.clearInterval(optimisticTimer.current);
        }
      }
    }, 850);
  }

  async function handleModelUpload(file: File) {
    setPendingAction("model");
    setError(null);
    setMessage(null);
    try {
      const payload = await uploadModelFile(file);
      setModel(payload.model);
      setActiveProfile(payload.profile);
      setResult(null);
      setMessage(`Validated checkpoint: ${payload.model.filename}`);
      await refreshHealth();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Model upload failed.");
    } finally {
      setPendingAction(null);
    }
  }

  async function handleEdfUpload(file: File) {
    setPendingAction("edf");
    setError(null);
    setMessage(null);
    try {
      const payload = await uploadEegAsset(file, "edf", "inference");
      setEdfAsset({
        path: payload.saved_path,
        filename: payload.filename,
        groupId: payload.group_id,
        source: "upload",
        status: "uploaded"
      });
      setMarkerAsset(null);
      setResult(null);
      setMessage(`Uploaded EEG recording: ${payload.filename}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "EDF upload failed.");
    } finally {
      setPendingAction(null);
    }
  }

  async function handleMarkerUpload(file: File) {
    if (!edfAsset?.groupId) {
      setError("Upload the EDF recording first so the marker CSV can be stored alongside it.");
      return;
    }
    setPendingAction("marker");
    setError(null);
    setMessage(null);
    try {
      const payload = await uploadEegAsset(file, "marker", "inference", edfAsset.groupId);
      setMarkerAsset({
        path: payload.saved_path,
        filename: payload.filename,
        groupId: payload.group_id,
        source: "upload",
        status: "uploaded"
      });
      setResult(null);
      setMessage(`Uploaded marker CSV: ${payload.filename}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Marker upload failed.");
    } finally {
      setPendingAction(null);
    }
  }

  async function handleRunInference() {
    if (!activeProfile?.model_ready || !activeProfile.user_model_path) {
      setError(READINESS_ERROR_MESSAGE);
      return;
    }
    if (!edfAsset?.path) {
      setError("Upload an EDF recording first.");
      return;
    }

    setPendingAction("infer");
    setError(null);
    setMessage(null);
    setResult(null);
    startOptimisticTimeline();

    try {
      const inference = await runInference({
        edf_path: edfAsset.path,
        marker_csv_path: markerAsset?.path,
        top_k_words: 8,
        retrieval_topk: STAGE2_TOP_K,
        stage2_mode: stage2Mode
      });
      if (optimisticTimer.current) {
        window.clearInterval(optimisticTimer.current);
      }
      setResult(inference);
      setTimeline(inference.timeline);
      setMessage(`Inference complete. Final sentence: ${inference.prediction.sentence_id}`);
      await refreshHistory();
      await refreshHealth();
    } catch (err) {
      if (optimisticTimer.current) {
        window.clearInterval(optimisticTimer.current);
      }
      setTimeline((previous: TimelineStep[]) =>
        previous.map((step: TimelineStep, index: number) =>
          index === previous.findIndex((item: TimelineStep) => item.status === "running")
            ? { ...step, status: "failed", detail: err instanceof Error ? err.message : "Inference failed." }
            : step
        )
      );
      setError(err instanceof Error ? err.message : "Inference failed.");
    } finally {
      setPendingAction(null);
    }
  }

  const warningMessages = useMemo(() => {
    const warnings = new Set<string>();
    if (result) {
      result.preprocessing.warnings.forEach((warning: string) => warnings.add(warning));
      result.stage2.warnings.forEach((warning: string) => warnings.add(warning));
    }
    if (health?.python_service_ok === false) {
      warnings.add("Python inference service is currently unavailable.");
    }
    return Array.from(warnings);
  }, [health, result]);

  const canRun = Boolean(activeProfile?.model_ready && activeProfile.user_model_path && edfAsset?.path) && pendingAction === null;
  const modelCardValue = model ?? profileBackedModel(activeProfile);
  const ollamaAvailable = health?.python_service?.ollama_available;
  const transformersAvailable = health?.python_service?.transformers_available;
  const cudaAvailable = health?.python_service?.cuda_available;
  const stage2StatusCards = [
    {
      label: "Qwen/Ollama",
      value: ollamaAvailable === undefined ? "Unchecked" : ollamaAvailable ? "Available" : "Unavailable",
      tone: ollamaAvailable ? "border-emerald-200 bg-emerald-50/60" : "border-slate-200 bg-white"
    },
    {
      label: "Local Retrieval",
      value: transformersAvailable === undefined ? "Unchecked" : transformersAvailable ? "Installed" : "Not installed",
      tone: transformersAvailable ? "border-emerald-200 bg-emerald-50/60" : "border-slate-200 bg-white"
    },
    {
      label: "CUDA",
      value: cudaAvailable === undefined ? "Unchecked" : cudaAvailable ? "Available" : "CPU fallback",
      tone: cudaAvailable ? "border-emerald-200 bg-emerald-50/60" : "border-slate-200 bg-white"
    },
    {
      label: "Fallback",
      value: "Top-1 retrieval",
      tone: "border-slate-200 bg-white"
    }
  ];

  return (
    <AppShell
      title="Khayal Session"
      subtitle="Decode imagined Arabic speech from EDF trials using a personalized Stage 1 checkpoint and local sentence selection."
    >
      <section className="kh-panel-strong mb-5 px-5 py-5 md:px-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-3xl">
            <p className="kh-kicker">Session Overview</p>
            <h2 className="mt-1 text-2xl font-extrabold tracking-tight text-slate-900">Local-only inference workspace</h2>
            <p className="mt-2 text-sm text-slate-600">
              The active profile stays in control. Upload a local checkpoint and EDF trial, then run sentence decoding end-to-end.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="kh-chip">12-Sentence Catalog</span>
            <span className="kh-chip">Retrieval + Qwen</span>
          </div>
        </div>
      </section>

      {message ? (
        <section className="mb-5 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-semibold text-emerald-800">
          {message}
        </section>
      ) : null}

      {error ? (
        <section className="mb-5 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-semibold text-rose-800">
          {error}
        </section>
      ) : null}

      {warningMessages.length > 0 ? (
        <section className="mb-5 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-sm font-bold text-amber-900">Warnings</p>
            <div className="mt-2 space-y-1 text-sm text-amber-800">
              {warningMessages.map((warning: string) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          </section>
      ) : null}

      <div className="grid gap-5 2xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <div className="space-y-5">
          <div className="grid gap-5 xl:grid-cols-2 xl:items-start">
            <ModelFileCard
              profile={activeProfile}
              model={modelCardValue}
              pending={pendingAction === "model"}
              onUpload={handleModelUpload}
            />

            <section className="kh-panel self-start p-5">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="min-w-0 flex-1">
                  <p className="kh-kicker">Stage 2 Decoder</p>
                  <h3 className="mt-1 max-w-[18ch] text-[1.7rem] font-extrabold leading-tight tracking-tight text-slate-900">
                    Retrieval + Qwen/Ollama Selection
                  </h3>
                  <p className="mt-3 max-w-[34ch] text-sm leading-6 text-slate-600">
                    Local retrieval ranks the 12-sentence catalog. Qwen/Ollama then chooses the final sentence, with rank-1 fallback if selection fails.
                  </p>
                </div>
                <StatusBadge label="Qwen Mode" tone="good" />
              </div>

              <div className="mt-5 grid gap-3 sm:grid-cols-2">
                {stage2StatusCards.map((item) => (
                  <div
                    key={item.label}
                    className={`min-w-0 rounded-2xl border px-3.5 py-3 ${item.tone}`}
                  >
                    <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">{item.label}</p>
                    <p className="mt-1 break-words text-sm font-semibold leading-5 text-slate-900">{item.value}</p>
                  </div>
                ))}
              </div>
            </section>
          </div>

          <div className="grid gap-5 xl:grid-cols-2">
            <EdfFileCard
              kicker="EEG Recording"
              title="EDF Trial"
              description="Upload one raw EDF sentence trial recorded with Emotiv EPOC X."
              accept=".edf"
              statusLabel={pendingAction === "edf" ? "Uploading" : edfAsset ? "Uploaded" : "Missing"}
              statusTone={edfAsset ? "good" : "warn"}
              fileName={edfAsset?.filename ?? "No EDF recording selected"}
              filePath={edfAsset?.path}
              helper="Expected: one 3-word Khayal trial EDF."
              pending={pendingAction === "edf"}
              onUpload={handleEdfUpload}
            />

            <EdfFileCard
              kicker="Marker CSV"
              title="Marker File"
              description="Upload the optional interval marker CSV."
              accept=".csv"
              statusLabel={pendingAction === "marker" ? "Uploading" : markerAsset ? "Uploaded" : "Optional"}
              statusTone={markerAsset ? "good" : "neutral"}
              fileName={markerAsset?.filename ?? "No marker CSV selected"}
              filePath={markerAsset?.path}
              helper="Expecting exactly 3 usable phase_Imagine markers."
              pending={pendingAction === "marker"}
              disabled={!edfAsset}
              onUpload={handleMarkerUpload}
            />
          </div>

          <section className="kh-panel-strong p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="kh-kicker">Run Controls</p>
                <h3 className="mt-1 text-lg font-bold text-slate-900">Execute the local inference workflow</h3>
                <p className="mt-2 text-sm text-slate-600">
                  The run button stays locked until the active profile has a validated Stage 1 checkpoint and an EDF recording is available.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={handleRunInference} disabled={!canRun} className="kh-btn">
                  {pendingAction === "infer" ? "Running Khayal Inference..." : "Run Khayal Inference"}
                </button>
              </div>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-4">
              <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Active Profile</p>
                <p className="mt-1 text-sm font-bold text-slate-900">{activeProfile?.name ?? "No profile selected"}</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Model Ready</p>
                <p className="mt-1 text-sm font-bold text-slate-900">{activeProfile?.model_ready ? "Yes" : "No"}</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Stage 2 Mode</p>
                <p className="mt-1 text-sm font-bold text-slate-900">qwen</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Retrieval Top-k</p>
                <p className="mt-1 text-sm font-bold text-slate-900">{STAGE2_TOP_K}</p>
              </div>
            </div>
          </section>

          <InferenceTimeline timeline={timeline} />
          <Stage1EvidencePanel stage1={result?.stage1 ?? null} />
          <Stage2CandidatesPanel stage2={result?.stage2 ?? null} prediction={result?.prediction ?? null} />
          <DebugDetailsPanel profile={activeProfile} result={result} />
        </div>

        <div className="space-y-5">
          <FinalPredictionCard result={result} />

          <section>
            <div className="mb-2">
              <p className="kh-kicker">Recent Sessions</p>
            </div>
            <SessionTable sessions={history} />
          </section>
        </div>
      </div>
    </AppShell>
  );
}
