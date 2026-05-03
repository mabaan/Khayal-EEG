"use client";

import { StatusBadge } from "@/components/status-badge";

interface EdfFileCardProps {
  kicker: string;
  title: string;
  description: string;
  accept: string;
  statusLabel: string;
  statusTone: "neutral" | "good" | "warn" | "error";
  fileName: string;
  filePath?: string | null;
  pending: boolean;
  disabled?: boolean;
  helper?: string;
  onUpload: (file: File) => void;
}

export function EdfFileCard({
  kicker,
  title,
  description,
  accept,
  statusLabel,
  statusTone,
  fileName,
  filePath,
  pending,
  disabled = false,
  helper,
  onUpload
}: EdfFileCardProps) {
  return (
    <section className="kh-panel overflow-hidden p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="kh-kicker">{kicker}</p>
          <h3 className="mt-1 text-lg font-bold text-slate-900">{title}</h3>
          <p className="mt-2 text-sm text-slate-600">{description}</p>
        </div>
        <StatusBadge label={statusLabel} tone={statusTone} />
      </div>

      <div className="mt-4 min-w-0 overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
        <p className="break-all text-sm font-semibold text-slate-800">{fileName}</p>
        <p className="mt-1 break-all text-xs leading-5 text-slate-500">{filePath ?? "No local file selected yet."}</p>
      </div>

      {helper ? <p className="mt-3 break-words text-xs leading-5 text-slate-500">{helper}</p> : null}

      <label className="mt-4 flex min-w-0 cursor-pointer flex-col gap-2 rounded-2xl border border-dashed border-sky-200 bg-sky-50/50 px-4 py-3 text-sm transition hover:border-sky-300">
        <span className="font-semibold text-slate-700">{pending ? "Uploading..." : `Upload ${title}`}</span>
        <input
          type="file"
          accept={accept}
          disabled={disabled || pending}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onUpload(file);
              event.currentTarget.value = "";
            }
          }}
          className="min-w-0 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
        />
      </label>
    </section>
  );
}
