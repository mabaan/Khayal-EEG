"use client";

import { useState } from "react";
import { AppShell } from "@/components/app-shell";
import { startTraining } from "@/lib/api-client";

export default function TrainingPage() {
  const [edfPaths, setEdfPaths] = useState("");
  const [status, setStatus] = useState("Ready to start Diff-E Personalization.");
  const [error, setError] = useState<string | null>(null);

  async function runTraining() {
    setError(null);
    setStatus("Training in progress...");

    try {
      const calibrationPaths = edfPaths
        .split("\n")
        .map((item) => item.trim())
        .filter(Boolean);

      const result = await startTraining(calibrationPaths);
      setStatus(`Training finished: ${result.message}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed.");
      setStatus("Training failed.");
    }
  }

  return (
    <AppShell title="Training" subtitle="Prepare one personalized Stage 1 model per profile using fixed Diff-E personalization">
      <section className="kh-panel-strong mb-5 p-5">
        <p className="kh-kicker">Pipeline</p>
        <h2 className="mt-1 text-lg font-bold text-slate-900">Diff-E Personalization</h2>
        <p className="mt-2 text-sm text-slate-600">Training consumes uploaded calibration EDFs and writes a user-specific checkpoint to local storage.</p>
      </section>

      <section className="kh-panel p-5">
        <p className="kh-kicker">Training Inputs</p>
        <p className="mt-2 text-sm text-slate-600">Optional: enter one EDF path per line. Leave empty to auto-use all profile calibration EDF files.</p>

        <textarea
          value={edfPaths}
          onChange={(event) => setEdfPaths(event.target.value)}
          rows={9}
          className="kh-textarea mt-3"
          placeholder="storage/profiles/<profile_id>/raw_edf/file1.edf"
        />

        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button type="button" onClick={runTraining} className="kh-btn">
            Start Diff-E Personalization
          </button>
          <span className="kh-chip">Local-only checkpoint output</span>
        </div>

        <p className="mt-3 text-sm font-semibold text-slate-800">{status}</p>
        {error ? <p className="mt-1 text-sm font-semibold text-rose-600">{error}</p> : null}
      </section>
    </AppShell>
  );
}
