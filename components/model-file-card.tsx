"use client";

import type { ModelInfo, ProfileManifest } from "@/lib/types";
import { StatusBadge } from "@/components/status-badge";

interface ModelFileCardProps {
  profile: ProfileManifest | null;
  model: ModelInfo | null;
  pending: boolean;
  onUpload: (file: File) => void;
}

export function ModelFileCard({ profile, model, pending, onUpload }: ModelFileCardProps) {
  const status = pending
    ? { label: "Uploading", tone: "neutral" as const }
    : model?.validated || profile?.model_ready
      ? { label: "Validated", tone: "good" as const }
      : { label: "Missing", tone: "warn" as const };

  const fileName = model?.filename ?? profile?.user_model_path?.split(/[\\/]/).pop() ?? "No checkpoint selected";

  return (
    <section className="kh-panel overflow-hidden p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="kh-kicker">Stage 1 Model</p>
          <h3 className="mt-1 text-lg font-bold text-slate-900">DiffE Personalized Checkpoint</h3>
          <p className="mt-2 text-sm text-slate-600">Load a local `.pt` or `.pth` checkpoint for the active profile.</p>
        </div>
        <StatusBadge label={status.label} tone={status.tone} />
      </div>

      <div className="mt-4 min-w-0 overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
        <p className="break-all text-sm font-semibold text-slate-800">{fileName}</p>
        <p className="mt-1 break-all text-xs leading-5 text-slate-500">{model?.path ?? profile?.user_model_path ?? "No local checkpoint path yet."}</p>
      </div>

      <dl className="mt-4 grid gap-3 text-sm text-slate-600 sm:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
          <dt className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Model Type</dt>
          <dd className="mt-1 font-semibold text-slate-800">{model?.arch?.toUpperCase() ?? "DiffE"}</dd>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
          <dt className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Classes</dt>
          <dd className="mt-1 font-semibold text-slate-800">{model?.n_classes ?? 25}</dd>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
          <dt className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Subject</dt>
          <dd className="mt-1 font-semibold text-slate-800">{model?.subject ?? "Profile-attached"}</dd>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
          <dt className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Classifier</dt>
          <dd className="mt-1 font-semibold text-slate-800">{model?.classifier ?? "Unknown"}</dd>
        </div>
      </dl>

      <label className="mt-4 flex min-w-0 cursor-pointer flex-col gap-2 rounded-2xl border border-dashed border-teal-200 bg-teal-50/50 px-4 py-3 text-sm transition hover:border-teal-300">
        <span className="font-semibold text-slate-700">{pending ? "Uploading checkpoint..." : "Upload Stage 1 checkpoint"}</span>
        <input
          type="file"
          accept=".pt,.pth"
          disabled={pending}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onUpload(file);
              event.currentTarget.value = "";
            }
          }}
          className="min-w-0 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
        />
        <span className="text-xs text-slate-500">Accepted extensions: `.pt`, `.pth`.</span>
      </label>
    </section>
  );
}
