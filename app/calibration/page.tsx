"use client";

import { useState } from "react";
import { AppShell } from "@/components/app-shell";
import { FilePicker } from "@/components/file-picker";
import { SentenceGrid } from "@/components/sentence-grid";
import { Stepper } from "@/components/stepper";
import { CALIBRATION_FULL_REPETITIONS, CALIBRATION_STARTER_REPETITIONS } from "@/lib/constants";

const STEPS = ["Profile", "Record", "Upload EDF", "Verify"];

export default function CalibrationPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [message, setMessage] = useState("Upload a raw EDF calibration file.");
  const [error, setError] = useState<string | null>(null);

  async function upload() {
    if (!selectedFile) {
      setError("Please choose an EDF file first.");
      return;
    }

    const form = new FormData();
    form.append("file", selectedFile);
    form.append("purpose", "calibration");

    setError(null);
    try {
      const response = await fetch("/api/edf/upload", {
        method: "POST",
        body: form
      });
      const payload = (await response.json()) as { saved_path?: string; error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? "Upload failed.");
      }
      setMessage(`Saved calibration EDF: ${payload.saved_path}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  return (
    <AppShell title="Calibration" subtitle="Follow the fixed protocol schedule and upload raw EDF recordings for personalization">
      <div className="space-y-5">
        <Stepper steps={STEPS} current={2} />

        <section className="kh-panel p-5">
          <p className="kh-kicker">Calibration Policy</p>
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            <p className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
              Starter calibration: 12 sentences x {CALIBRATION_STARTER_REPETITIONS} repetitions
            </p>
            <p className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
              Full calibration: 12 sentences x {CALIBRATION_FULL_REPETITIONS} repetitions
            </p>
          </div>
          <p className="mt-3 text-sm text-slate-600">
            Per word event: rest 5s, stimulus 5s, imagination 6s. Keep only imagination window after discarding first 0.5s.
          </p>
        </section>

        <section className="kh-panel p-5">
          <p className="kh-kicker">Fixed Sentence Catalog</p>
          <div className="mt-3">
            <SentenceGrid />
          </div>
        </section>

        <section className="kh-panel p-5">
          <FilePicker label="Upload EEG Recording (.edf)" accept=".edf" onSelect={setSelectedFile} />
          <button onClick={upload} className="kh-btn mt-3" type="button">
            Save Calibration EDF
          </button>
          <p className="mt-2 text-sm text-slate-700">{message}</p>
          {error ? <p className="mt-1 text-sm font-semibold text-rose-600">{error}</p> : null}
        </section>
      </div>
    </AppShell>
  );
}
