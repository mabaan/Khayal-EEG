"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { EegWavePanel } from "@/components/eeg-wave-panel";
import { FilePicker } from "@/components/file-picker";
import { ResultPanel } from "@/components/result-panel";
import { StatusBadge } from "@/components/status-badge";
import { SessionTable } from "@/components/session-table";
import { fetchHistory, fetchSimulation, runInference } from "@/lib/api-client";
import type { InferenceResult, SessionRecord, SimulationSnapshot } from "@/lib/types";

export default function InferencePage() {
  const [file, setFile] = useState<File | null>(null);
  const [signalStatus, setSignalStatus] = useState("idle");
  const [currentStep, setCurrentStep] = useState("Waiting for input");
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [history, setHistory] = useState<SessionRecord[]>([]);
  const [snapshot, setSnapshot] = useState<SimulationSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refreshHistory() {
    const payload = await fetchHistory();
    setHistory(payload.sessions.slice(0, 6));
  }

  useEffect(() => {
    refreshHistory().catch((err) => setError(err instanceof Error ? err.message : "Failed to load recent sessions."));
  }, []);

  async function runSession() {
    if (!file) {
      setError("Please choose an EDF file.");
      return;
    }

    setError(null);
    setSignalStatus("uploading");
    setCurrentStep("Uploading EEG Recording");

    const form = new FormData();
    form.append("file", file);
    form.append("purpose", "inference");

    try {
      const uploadResp = await fetch("/api/edf/upload", { method: "POST", body: form });
      const uploadPayload = (await uploadResp.json()) as { saved_path?: string; error?: string };
      if (!uploadResp.ok || !uploadPayload.saved_path) {
        throw new Error(uploadPayload.error ?? "Upload failed");
      }

      setSignalStatus("running");
      setCurrentStep("Running Stage 1 + Stage 2 pipeline");
      const inference = await runInference(uploadPayload.saved_path, false);
      setResult(inference);
      setSignalStatus("complete");
      setCurrentStep("Session complete");
      await refreshHistory();
    } catch (err) {
      setSignalStatus("error");
      setCurrentStep("Session failed");
      setError(err instanceof Error ? err.message : "Session failed.");
    }
  }

  async function runSimulated() {
    setError(null);
    setSignalStatus("running");
    setCurrentStep("Simulated live replay");

    try {
      const simPayload = await fetchSimulation();
      setSnapshot(simPayload.simulation);
      if (simPayload.inference_result) {
        setResult(simPayload.inference_result);
      }
      setSignalStatus("complete");
      setCurrentStep("Simulation complete");
      await refreshHistory();
    } catch (err) {
      setSignalStatus("error");
      setCurrentStep("Simulation failed");
      setError(err instanceof Error ? err.message : "Simulation failed.");
    }
  }

  return (
    <AppShell title="Session" subtitle="Run local EEG decoding with fixed Diff-E + Retrieval + Qwen reranking">
      <div className="grid gap-5 xl:grid-cols-3">
        <div className="space-y-5 xl:col-span-2">
          <EegWavePanel snapshot={snapshot} />

          <div className="grid gap-3 md:grid-cols-3">
            <div className="kh-panel p-4">
              <p className="kh-kicker">Current Step</p>
              <p className="mt-2 text-sm font-bold text-slate-800">{currentStep}</p>
            </div>

            <div className="kh-panel p-4">
              <p className="kh-kicker">Signal Status</p>
              <div className="mt-2">
                <StatusBadge
                  label={signalStatus}
                  tone={signalStatus === "complete" ? "good" : signalStatus === "error" ? "error" : "neutral"}
                />
              </div>
            </div>

            <div className="kh-panel p-4">
              <p className="kh-kicker">Model Ready</p>
              <p className="mt-2 text-sm font-bold text-slate-800">Required before Session inference</p>
            </div>
          </div>
        </div>

        <div className="space-y-5">
          <section className="kh-panel p-4 md:p-5">
            <p className="kh-kicker">Session Controls</p>
            <div className="mt-3">
              <FilePicker label="Upload EEG Recording" accept=".edf" onSelect={setFile} />
            </div>
            <button type="button" onClick={runSession} className="kh-btn mt-3 w-full">
              Run Session
            </button>
            <button type="button" onClick={runSimulated} className="kh-btn kh-btn-secondary mt-2 w-full">
              Run Simulated Live Session
            </button>
            {error ? <p className="mt-2 text-sm font-semibold text-rose-600">{error}</p> : null}
          </section>

          <ResultPanel result={result} />
        </div>
      </div>

      <section className="mt-6">
        <div className="mb-2">
          <p className="kh-kicker">Recent Sessions</p>
        </div>
        <SessionTable sessions={history} />
      </section>
    </AppShell>
  );
}
